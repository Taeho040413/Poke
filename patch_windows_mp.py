from pathlib import Path

path = Path("pokemon_hrl/training/engine.py")
text = path.read_text(encoding="utf-8")

if "from functools import partial" not in text:
    old_import = """from argparse import Namespace
from pathlib import Path
from typing import Any
"""
    new_import = """from argparse import Namespace
from functools import partial
from pathlib import Path
from typing import Any
"""
    if old_import not in text:
        raise SystemExit("import block not found")
    text = text.replace(old_import, new_import, 1)

helper = '''def _make_interactive_env_for_vector(
    config_container: dict[str, Any],
    scenario_index: int,
    *_args: Any,
    **_kwargs: Any,
):
    """Pickle-safe env creator for Windows multiprocessing spawn."""
    cfg = OmegaConf.create(config_container)
    shared_plan = get_shared_plan_store()
    bootstrap_shared_planner(cfg, scenario_index=scenario_index, shared_plan=shared_plan)
    return make_interactive_env(
        cfg,
        scenario_index=scenario_index,
        puffer_wrapper=True,
        shared_plan=shared_plan,
    )


'''

anchor = "def resolve_vector_backend(train_cfg: Namespace):\n"
if "_make_interactive_env_for_vector" not in text:
    if anchor not in text:
        raise SystemExit("resolve_vector_backend anchor not found")
    text = text.replace(anchor, helper + anchor, 1)

old_local_creator = """    def env_creator(*_args, **_kwargs):
        return make_interactive_env(
            cfg,
            scenario_index=scenario_index,
            puffer_wrapper=True,
            shared_plan=shared_plan,
        )

"""

new_local_creator = """    env_creator = partial(
        _make_interactive_env_for_vector,
        OmegaConf.to_container(cfg, resolve=True),
        int(scenario_index),
    )

"""

if old_local_creator in text:
    text = text.replace(old_local_creator, new_local_creator, 1)
elif "env_creator = partial(" not in text:
    raise SystemExit("local env_creator block not found")

path.write_text(text, encoding="utf-8")
print("patched Windows-safe multiprocessing env_creator")
