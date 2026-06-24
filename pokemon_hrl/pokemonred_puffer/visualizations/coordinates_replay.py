import argparse
from datetime import datetime, timedelta
import os

import cv2
import mediapy
from tqdm import tqdm

from pokemonred_puffer.eval import BACKGROUND
from pokemonred_puffer.global_map import local_to_global

PLAYER_PATH = os.path.join(os.path.dirname(__file__), "player.png")


def _load_player_image(*, downsample: int) -> "cv2.typing.MatLike":
    # BACKGROUND와 동일하게 RGB로 맞춰야 mediapy 내보내기에서 색 반전이 없다.
    image = cv2.imread(PLAYER_PATH, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"player image not found: {PLAYER_PATH}")
    player = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return player[::downsample, ::downsample]


def _collect_coord_files(coords_dir: str, coords_file: str | None) -> list[str]:
    files: list[str] = []
    for path in os.listdir(str(coords_dir)):
        if coords_file and path != coords_file:
            continue
        if not path.endswith("coords.csv"):
            continue
        if len(path.split("-")) != 3:
            continue
        files.append(path)
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("coords_dir")
    parser.add_argument("output_dir")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride when reading the coordinates array"
        " or how to bucket coordinates by timestamp in seconds in seconds mode.",
    )
    parser.add_argument(
        "--left-crop",
        type=int,
        default=0,
        help="Amount of steps or seconds relative to the "
        "earliest timestamp detected to to crop from the left side.",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=None,
        help="Amount of steps or seconds to read not including stride. 0 means read all.",
    )
    parser.add_argument(
        "--downsample", type=int, default=1, help="Amount to downsample the video by. Max is 16"
    )
    parser.add_argument(
        "--image-crop",
        default=[0, 0, 0, 0],
        type=lambda x: [int(y) for y in x.split(",")],
        help="top,vlength,left,hlength cropping in pixels off the original image",
    )
    parser.add_argument(
        "--sync-method",
        choices=["time", "steps"],
        default="steps",
        help="How to synchronize the coordinates files",
    )
    parser.add_argument("--coords-file", help="Only render this file in coords_dir.")
    args = parser.parse_args()

    # step_index -> set[(map_id, y, x)]
    steps: dict[int, set[tuple[int, int, int]]] = {}
    max_length = None if args.length in (None, 0) else int(args.length)
    left_crop = int(args.left_crop)

    files = _collect_coord_files(str(args.coords_dir), args.coords_file)
    if args.sync_method == "time":
        if not files:
            raise ValueError("No coordinate files found for time sync")
        earliest_time: datetime | None = None
        for path in files:
            date_string = path.split("-")[1]
            ts = datetime.strptime(date_string, "%Y%m%d%H%M%S")
            earliest_time = min(earliest_time or ts, ts)
        assert earliest_time is not None
        left_crop = int((earliest_time + timedelta(seconds=args.left_crop)).timestamp())

    for path in tqdm(files):
        with open(os.path.join(args.coords_dir, path)) as f:
            for i, line in enumerate(f):
                timestamp, map_n, y_pos, x_pos = line.strip(" \n").split(",")
                if args.sync_method == "steps":
                    if i < left_crop:
                        continue
                    offset = i - left_crop
                    if offset % args.stride != 0:
                        continue
                    if max_length is not None and offset > max_length:
                        break
                    key = offset // args.stride
                else:
                    ts_value = int(datetime.strptime(timestamp, "%Y%m%d%H%M%S").timestamp())
                    if ts_value < left_crop:
                        continue
                    offset = ts_value - left_crop
                    if offset % args.stride != 0:
                        continue
                    if max_length is not None and offset > max_length:
                        break
                    key = offset // args.stride
                if key not in steps:
                    steps[key] = set()
                steps[key].add((int(map_n), int(y_pos), int(x_pos)))

    sorted_steps = sorted(steps.items(), key=lambda k: k[0])

    top, vlength, left, hlength = args.image_crop
    if not vlength:
        vlength = BACKGROUND.shape[0] - top
    if not hlength:
        hlength = BACKGROUND.shape[1] - left
    background = BACKGROUND.copy()[
        top : top + vlength : args.downsample, left : left + hlength : args.downsample
    ]
    player = _load_player_image(downsample=args.downsample)
    os.makedirs(args.output_dir, exist_ok=True)
    with mediapy.VideoWriter(
        os.path.join(args.output_dir, "coords.mp4"),
        background.shape[:2],
        fps=24,
    ) as writer:
        for _, step in tqdm(sorted_steps):
            # This is slow. See if we can make all the frames in a threadpool
            frame = background.copy()
            for map_n, y_pos, x_pos in step:
                y, x = local_to_global(y_pos, x_pos, map_n)
                y *= 16
                x *= 16
                y = (y - top) // args.downsample
                x = (x - left) // args.downsample
                # check if the image is in frame
                if (
                    0 <= y
                    and 0 <= x
                    and y + player.shape[0] <= background.shape[0]
                    and x + player.shape[1] <= background.shape[1]
                ):
                    frame[
                        y : y + player.shape[0],
                        x : x + player.shape[1],
                    ] = player
            writer.add_image(frame)
    if hasattr(os, "sync"):
        os.sync()


if __name__ == "__main__":
    main()
