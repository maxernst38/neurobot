# Camera-based teleoperation of a LeRobot SO-101

Two ChArUco-calibrated webcams track an operator's arm with MediaPipe,
triangulate the pose in 3D, and drive a LeRobot SO-101 follower arm from the
measured joint angles. Developed alongside an EMG-based control path as a
capstone project.

## Setup

The two vendored dependencies are git submodules, so clone recursively:

```bash
git clone --recurse-submodules <this repo>
cd capstone_project
```

Already cloned without `--recurse-submodules`? Fetch them with:

```bash
git submodule update --init
```

### Apply the lerobot patch

`lerobot` needs one local patch before it will run here. Without it, importing
the robot driver pulls in torch — and on Windows, the MSVC runtime that torch's
DLLs need and a stock Python install does not have.

```bash
cd lerobot
git apply ../patches/lerobot-no-torch.patch
cd ..
```

See [patches/README.md](patches/README.md) for what it changes and why.

### Install

```bash
pip install -e lerobot
pip install -r arm_motion_tracker/requirements.txt
```

## Running

Tracker alone, in a browser (no robot needed):

```bash
python arm_motion_tracker/tracker.py --web
```

Then open <http://localhost:8080/>. The page binds to localhost by default
because it commands a robot arm; `--web-host 0.0.0.0` is the deliberate opt-in
for reaching it from another device.

Tracker plus the robot bridge:

```bash
python run_teleop.py
```

Recalibrate the robot's motor ranges:

```bash
python main.py --recalibrate
```

## Recording sessions

Captures are written to `recordings/` as JSON Lines — one header, one row per
frame — holding three layers of each instant: what each camera saw (2D
landmarks and confidence), where that put each joint (triangulated xyz), and
what that measured (joint angles). Keeping the raw detections is what lets an
old capture be re-processed against a new calibration or corrected code.

```bash
python arm_motion_tracker/tracker.py --web --record --record-note "trial 3"
python arm_motion_tracker/tracker.py --web --record --record-video   # + raw footage
python arm_motion_tracker/tracker.py --web --record --mask-face block  # anonymise
```

Inspect one without starting the tracker:

```bash
python arm_motion_tracker/recording.py                      # list
python arm_motion_tracker/recording.py session-<stamp>.jsonl  # summary
```

Recordings are gitignored — roughly 8.8 MB per minute, or several times that
with video. Archive the ones that matter deliberately.

## Calibration

Stereo calibration lives in `arm_motion_tracker/calibration/` and is produced
by `arm_motion_tracker/calibrate_cameras.py`. Your own arm's ranges of motion
are recorded from the tracker page and stored in `config/arm_ranges.json`;
they carry a geometry version, and a range recorded against an older
definition of an angle is refused rather than silently misapplied.

## Licence

lerobot is Apache-2.0 (Copyright 2024 The Hugging Face team); the local
modifications are marked in place in its source. Project code is this
repository's own.
