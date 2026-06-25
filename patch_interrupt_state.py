from pathlib import Path

path = Path("pokemon_hrl/pokemonred_puffer/cleanrl_puffer.py")
text = path.read_text(encoding="utf-8")

old = '''    def save_policy_only(self, filename: str = "model_interrupt.pt") -> str:
        """Save policy weights only — not used for auto-resume (goal/interval checkpoints)."""
        path = self._checkpoint_run_dir()
        out_path = os.path.join(path, filename)
        tmp = out_path + ".tmp"
        torch.save(self.uncompiled_policy, tmp)
        os.replace(tmp, out_path)
        return out_path
'''

new = '''    def save_policy_only(self, filename: str = "model_interrupt.pt") -> str:
        """Save policy weights only plus a game-state sidecar for restart.

        The .pt file remains policy-only. The emulator state is written separately
        as game_latest.state in the same run directory so resume can start from
        the same game point instead of the ROM intro.
        """
        path = self._checkpoint_run_dir()
        out_path = os.path.join(path, filename)
        tmp = out_path + ".tmp"
        torch.save(self.uncompiled_policy, tmp)
        os.replace(tmp, out_path)

        game_path = self._save_game_checkpoint(path)
        if game_path:
            print(
                "[checkpoint] Ctrl+C — 재시작용 게임 state sidecar 저장: "
                f"{game_path}",
                flush=True,
            )

        return out_path
'''

if old not in text:
    raise SystemExit("save_policy_only block not found")

path.write_text(text.replace(old, new, 1), encoding="utf-8")
print("patched policy-only checkpoint to save game_latest.state sidecar")
