# Patches against vendored dependencies

## `lerobot-no-torch.patch`

Applies to [Seeed-Projects/lerobot](https://github.com/Seeed-Projects/lerobot)
at commit `0f392484458cb5ebca0310c0c4c47390a31c80ed` (branch `main`).

**What it does.** Makes `lerobot.robots` importable without torch.

`lerobot.motors.motors_bus` imports exactly two names from
`lerobot/utils/utils.py` — `enter_pressed` and `move_cursor_up` — and both are
pure stdlib. But that module imported `torch`, `accelerate` and `datasets` at
module scope, and `robot.py` / `so_follower.py` imported `RobotAction` and
`RobotObservation` from `lerobot.processor`, which is built on torch. The
effect was that driving a servo over a serial port pulled in the entire ML
stack — and on Windows, the MSVC runtime that torch's DLLs need and a stock
Python install does not have.

The patch moves those imports to be lazy (inside the functions that use them)
or annotation-only (`if TYPE_CHECKING:` plus `from __future__ import
annotations`). `RobotAction` and `RobotObservation` are both plain
`TypeAlias = dict[str, Any]` and appear only in annotations, so nothing needs
them at runtime. No behaviour changes; nothing on the motors path calls any of
the lazily-imported names.

Three files, ~57 lines: `robots/robot.py`, `robots/so_follower/so_follower.py`,
`utils/utils.py`. Each carries a `LOCAL PATCH (capstone_project)` comment
explaining itself in place, so the reason survives even if this file does not.

**Applying it**

```bash
cd lerobot
git apply ../patches/lerobot-no-torch.patch
```

Verified to apply cleanly to the pristine upstream commit above, and to
reverse cleanly off the patched tree.

**If you update the lerobot checkout,** re-apply this and regenerate the patch:

```bash
cd lerobot
git diff -- src/lerobot/robots/robot.py \
            src/lerobot/robots/so_follower/so_follower.py \
            src/lerobot/utils/utils.py \
  --output=../patches/lerobot-no-torch.patch
```

Do not add `--ignore-all-space` when regenerating. It omits whitespace from
the context lines and produces a patch that looks right, reverses cleanly
against the tree it came from, and then fails to apply to a fresh checkout.

## Note on `Seeed_RoboController`

Its only local difference from upstream is a file-permission bit on `setup.py`
(`755` → `644`), which Windows produces spuriously. There is nothing to patch.

## Licence

lerobot is Apache-2.0 (Copyright 2024 The Hugging Face team). The modifications
described here are marked in place in the source, per section 4(b) of that
licence.
