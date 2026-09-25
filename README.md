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
python run_teleop.py --no-robot        # cameras only, no arm attached
python run_teleop.py --num-cameras 1   # one camera
```

`--num-cameras` takes 2 (the default, and what triangulation needs), 1, or 0.
With one camera there is nothing to triangulate, so depth comes from
MediaPipe's monocular guess rather than from measurement — usable for
gestures, unreliable for anything needing real depth. With 0 the tracker opens
no cameras at all and serves the UI for replaying recordings.

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

## Using the web tool

`--web` serves the UI at <http://localhost:8080/>. Everything textual is DOM
driven by a websocket feed; the camera panes and the 3D projections are MJPEG
streams still rendered by OpenCV. Every control is a button, and the
single-key shortcuts work on the page too.

**Camera panes** — one per camera, each with its resolution, health, and
whether *that* camera can see a pose right now. Both cameras drawing a
skeleton is not the same claim as both contributing to triangulation, which
is why they are separate panes rather than one composed picture.

**3D projections** (`V`) — front, side and top views of the fused skeleton.
Before engaging, a yellow ghost shows the posture matching the arm's current
pose, which is how you line yourself up.

**Metrics** — the five tracked quantities: shoulder heading, shoulder flex,
elbow flex, wrist flex, grip.

**Triangulation** (`T`) — per-landmark, the answer to "both cameras can see
me, so why is nothing triangulating". Each joint says which gate it failed:
seen by neither camera, or seen by both and placed somewhere the two views
cannot reconcile. `I` keeps joints that fail those gates, labelled as
inferred rather than dropped — useful when the cameras cannot both see an arm
held across the body, but a reading built on an inferred joint can be
perfectly steady and wrong.

**Robot** (`C` to connect) — deviation bars per joint showing how far you are
from the arm's pose, with the engage tolerance shaded. `E` engages once
matched, `R` releases. While engaged, chips show which joints are actually
being driven; a joint whose metric is missing holds its last target, which
from outside looks identical to a dead servo, so it is named rather than left
ambiguous.

**Jog the arm** — `a/z` pan, `s/x` lift, `d/c` elbow, `f/v` wrist flex,
`g/b` wrist roll, `h/n` gripper. `P` saves the current pose as the startup
pose. Only available while not engaged.

**My range of motion** — records what your arm actually covers, so a degree
of your movement means the right amount of robot movement. Start it, move
every joint through its full comfortable travel including opening and closing
your hand for ~30 s, then save. Writes `config/arm_ranges.json`.

**Record this session** / **Replay a recording** — see above. Replay offers
three ways to derive the angles: *As recorded* (what the robot was told at the
time), *From 3D points* (recompute with today's metric code), and
*Re-triangulate* (rebuild the 3D from the recorded 2D using the calibration
loaded now). The third is how a capture gets re-processed after recalibrating,
and is unavailable when no calibration is loaded.

The page binds to localhost by default because it commands a robot arm. Use
`--web-host 0.0.0.0` to open it on a phone propped where you can see it while
standing in front of the cameras.

## Calibrating the cameras

Triangulation is the whole point of two cameras — it makes depth measured
rather than guessed — and it needs a stereo calibration. The result lands in
`arm_motion_tracker/calibration/stereo_calibration.json`.

### 1. Print a board

```bash
python arm_motion_tracker/make_calibration_board.py --cols 6 --rows 4 --square-mm 45
```

Writes a PDF with the artwork at exact millimetre dimensions plus a 100 mm
ruler. **Print at 100% / "actual size", never "fit to page"** — fit-to-page
silently rescales the pattern and every distance computed later inherits the
error. Measure the printed ruler to confirm, then mount it on something rigid
and flat. A curled sheet is not a plane, and the maths assumes a plane.

Bigger squares beat more squares for a widely separated pair: both cameras
must resolve the *same* corners at once, and a board held between two cameras
far apart is foreshortened in both views, so small markers stop decoding well
before they stop being visible.

### 2. Capture

```bash
python arm_motion_tracker/calibrate_cameras.py capture --square-mm 45
```

Use whatever you actually measured on the printout, not what you asked for.
Controls: `SPACE` capture, `A` auto-capture, `U` undo, `Q` done.

Hold the board where **both** cameras see it, and vary the pose a lot —
distance, tilt, and position across each frame, including the corners. Views
where only one camera sees the board are not wasted: they still feed that
camera's own intrinsics. Only the extrinsics need simultaneous views, so
collect plenty of those specifically.

### 3. Solve

```bash
python arm_motion_tracker/calibrate_cameras.py calibrate
```

Capture and solve are separate steps on purpose: images can be re-processed
with different board or detector settings without standing in front of the
cameras again, and a disappointing result stays diagnosable because the
evidence is still on disk. Capture records its board settings into
`session.json` beside the images and calibrate reads them back, so the two
cannot silently disagree about the board — which would otherwise produce a
confidently wrong result.

Read the reported RMS. Sub-pixel stereo RMS is good; a large one means the
solve did not converge on a consistent geometry, and the tracker will happily
triangulate nonsense from it. The number of **stereo** views matters more than
the total — those are the only ones constraining where the cameras are
relative to each other.

`--pattern checkerboard` is also supported. It is simpler to print, but
all-or-nothing: every inner corner must be visible for a view to count, so it
discards far more views on a widely separated pair, and its corners carry no
identity — if the two cameras ever order them differently the stereo solve
fails with a huge RMS.

### Afterwards

The tracker adapts intrinsics if you capture at a different resolution than
you calibrated at, but that adaptation is a model of how the camera changes
mode, not a measurement — prefer matching resolutions. **Moving either camera
invalidates the calibration**; the extrinsics describe where they were.

## Your arm's ranges

Recorded from the tracker page (see *My range of motion*) into
`config/arm_ranges.json`. They carry a geometry version, and a range recorded
against an older definition of an angle is refused per metric rather than
silently misapplied — a stale span is a wrong gain, and too large a gain is an
arm that crosses its travel on a small movement of yours.

The robot's own motor ranges are separate, and recorded with
`python main.py --recalibrate`.

## Licence

lerobot is Apache-2.0 (Copyright 2024 The Hugging Face team); the local
modifications are marked in place in its source. Project code is this
repository's own.
