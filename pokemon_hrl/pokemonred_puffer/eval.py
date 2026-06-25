import os

import cv2
import numpy as np
from numba import jit

from pokemonred_puffer.global_map import PAD


KANTO_MAP_PATH = os.path.join(os.path.dirname(__file__), "kanto_map_dsv.png")


def _imread_unicode(path: str) -> np.ndarray:
    """Load an image from disk; cv2.imread fails on Windows non-ASCII paths."""
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def _load_background() -> np.ndarray:
    # cv2.imread()는 BGR로 로드되지만 downstream(wandb.Image, mediapy, matplotlib)은 RGB를 기대함.
    # 한 번만 변환해 두어 채널 스왑으로 인한 색 반전(빨간 지붕이 파랗게 보이는 등)을 방지한다.
    background = cv2.cvtColor(_imread_unicode(KANTO_MAP_PATH), cv2.COLOR_BGR2RGB)
    return np.pad(background, ((PAD * 16, PAD * 16), (PAD * 16, PAD * 16), (0, 0)))


BACKGROUND = _load_background()



def make_agent_memory_grid(counts: np.ndarray, scale: int = 4) -> np.ndarray:
    """Render full global agent-memory grid without Kanto background.

    Input:
      - (N, H, W, 3): RGB memory grids from env infos
      - (N, H, W): scalar visit maps

    Output:
      - upscaled uint8 RGB image for wandb.Image
    """
    counts = np.asarray(counts)
    if counts.ndim == 4 and counts.shape[-1] == 3:
        rgb = np.max(counts.astype(np.float32), axis=0)
    elif counts.ndim == 3:
        visit = np.max(counts.astype(np.float32), axis=0)
        if np.max(visit) > 0:
            visit = visit / np.max(visit)
        rgb = np.zeros((*visit.shape, 3), dtype=np.float32)
        rgb[..., 1] = visit
    else:
        raise ValueError(f"Unsupported grid shape for make_agent_memory_grid: {counts.shape}")

    rgb = np.clip(rgb, 0.0, 1.0)
    image = (255.0 * rgb).astype(np.uint8)

    scale = max(1, int(scale))
    if scale > 1:
        image = np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)
    return image


def make_pokemon_red_overlay(counts: np.ndarray):
    """Aggregate env exploration maps onto the Kanto background.

    - Legacy: ``counts`` shape ``(n_env, H, W)`` scalar visit density (HSV colormap).
    - RGB: ``counts`` shape ``(n_env, H, W, 3)`` pre-colored tiles (visit / stuck / high reward).
    """
    if counts.ndim == 4 and counts.shape[-1] == 3:
        return _make_pokemon_red_overlay_rgb(counts)
    return _make_pokemon_red_overlay_hsv(counts)


def _make_pokemon_red_overlay_rgb(counts: np.ndarray):
    # 각 env 맵은 이미 카테고리별 색(방문=연두, stuck=빨강, 고보상=파랑, 건물=보라)으로
    # 칠해져 있다. 산술 평균을 쓰면 "1개 env만 방문한 타일"이 0으로 희석돼 검게 뭉개지고
    # 0 초과 마스크에는 걸려 배경을 덮어 얼룩처럼 보인다. 채널별 max(= 하이라이트 합집합)로
    # 바꾸면 소수 env만 방문한 타일도 원색이 유지된다.
    rgb = np.max(counts, axis=0).astype(np.float32)
    rgb = np.clip(rgb, 0.0, 1.0)
    overlay = (255.0 * rgb).astype(np.uint8)
    # 색이 극히 작은 잔여 노이즈까지 오버레이되지 않도록 채널 최대값 기준으로 살짝 컷오프.
    nonzero = np.ascontiguousarray(
        np.where(np.max(rgb, axis=-1) > 1e-2, 1.0, 0.0).astype(np.float32)
    )

    kernel = np.ascontiguousarray(np.ones((16, 16), dtype=np.uint8))
    r = np.ascontiguousarray(overlay[..., 0])
    g = np.ascontiguousarray(overlay[..., 1])
    b = np.ascontiguousarray(overlay[..., 2])
    r = np.kron(r, kernel).astype(np.uint8)
    g = np.kron(g, kernel).astype(np.uint8)
    b = np.kron(b, kernel).astype(np.uint8)
    overlay = np.stack((r, g, b), axis=-1)
    mask = np.kron(nonzero, kernel).astype(np.uint8)
    mask = np.stack((mask, mask, mask), axis=-1) != 0

    # BACKGROUND는 로드 시 이미 RGB로 변환되어 있어 추가 변환이 필요 없다.
    render = BACKGROUND.copy().astype(np.int32)
    render_shape = render.shape
    render = render.ravel()
    render[mask.ravel()] = 0.2 * render[mask.ravel()] + 0.8 * overlay.ravel()[mask.ravel()]
    render = render.reshape(render_shape)
    render = np.clip(render, 0, 255).astype(np.uint8)
    return render


@jit(nopython=True, nogil=True)
def _make_pokemon_red_overlay_hsv(counts: np.ndarray):
    # TODO: Rethink how this scaling works
    # Divide by number of elements > 0
    # The clip scaling needs to be re-calibrated since my
    # overlay is from the global map with fading
    scaled = np.ascontiguousarray(np.sum(counts, axis=0).astype(np.float32))
    scaled = scaled / np.max(scaled)
    nonzero = np.ascontiguousarray(np.where(scaled > 0, 1, 0).astype(np.float32))
    # scaled = np.clip(counts, 0, 1000) / 1000.0

    # Convert counts to hue map
    hsv = np.stack((2 * (1 - scaled) / 3, nonzero, nonzero), axis=-1)

    # Convert the HSV image to RGB
    overlay = 255 * hsv_to_rgb(hsv)

    # Upscale to 16x16
    kernel = np.ascontiguousarray(np.ones((16, 16), dtype=np.uint8))
    r = np.ascontiguousarray(overlay[..., 0])
    g = np.ascontiguousarray(overlay[..., 1])
    b = np.ascontiguousarray(overlay[..., 2])
    r = np.kron(r, kernel).astype(np.uint8)
    g = np.kron(g, kernel).astype(np.uint8)
    b = np.kron(b, kernel).astype(np.uint8)
    overlay = np.stack((r, g, b), axis=-1)
    mask = np.kron(nonzero, np.ascontiguousarray(kernel)).astype(np.uint8)
    mask = np.stack((mask, mask, mask), axis=-1) != 0

    # Combine with background
    render = BACKGROUND.copy().astype(np.int32)
    render_shape = render.shape
    render = render.ravel()
    render[mask.ravel()] = 0.2 * render[mask.ravel()] + 0.8 * overlay.ravel()[mask.ravel()]
    render = render.reshape(render_shape)
    render = np.clip(render, 0, 255).astype(np.uint8)

    return render


@jit(nopython=True, nogil=True)
def hsv_to_rgb(hsv):
    """
    Copied from matplotlib for numba
    Convert HSV values to RGB.

    Parameters
    ----------
    hsv : (..., 3) array-like
       All values assumed to be in range [0, 1]

    Returns
    -------
    (..., 3) `~numpy.ndarray`
       Colors converted to RGB values in range [0, 1]
    """
    # hsv = np.asarray(hsv)

    # check length of the last dimension, should be _some_ sort of rgb
    if hsv.shape[-1] != 3:
        raise ValueError(
            "Last dimension of input array must be 3; " f"shape {hsv.shape} was found."
        )

    in_shape = hsv.shape
    # hsv = np.array(
    #     hsv, copy=False,
    #     dtype=np.float32,  # Don't work on ints.
    #     ndmin=2,  # In case input was 1D.
    # )
    hsv = hsv.astype(np.float32)

    h = hsv[..., 0].ravel()
    s = hsv[..., 1].ravel()
    v = hsv[..., 2].ravel()

    r = np.empty_like(h)
    g = np.empty_like(h)
    b = np.empty_like(h)

    i = (h * 6.0).astype(np.int32)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    idx = i % 6 == 0
    r[idx] = v[idx]
    g[idx] = t[idx]
    b[idx] = p[idx]

    idx = i == 1
    r[idx] = q[idx]
    g[idx] = v[idx]
    b[idx] = p[idx]

    idx = i == 2
    r[idx] = p[idx]
    g[idx] = v[idx]
    b[idx] = t[idx]

    idx = i == 3
    r[idx] = p[idx]
    g[idx] = q[idx]
    b[idx] = v[idx]

    idx = i == 4
    r[idx] = t[idx]
    g[idx] = p[idx]
    b[idx] = v[idx]

    idx = i == 5
    r[idx] = v[idx]
    g[idx] = p[idx]
    b[idx] = q[idx]

    idx = s == 0
    r[idx] = v[idx]
    g[idx] = v[idx]
    b[idx] = v[idx]

    r = r.reshape(hsv[..., 0].shape)
    g = g.reshape(hsv[..., 0].shape)
    b = b.reshape(hsv[..., 0].shape)
    rgb = np.stack((r, g, b), axis=-1)

    return rgb.reshape(in_shape)
