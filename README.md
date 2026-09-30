# Camera-based teleoperation of a LeRobot SO-101

Two ChArUco-calibrated webcams track an operator's arm with MediaPipe,
triangulate the pose in 3D, and drive a LeRobot SO-101 follower arm from the
measured joint angles. Developed alongside an EMG-based control path as a
capstone project.

## Setup: from a fresh machine to teleop with one camera

Every command runs from a terminal: PowerShell on Windows, Terminal on macOS
and Linux. Where the platforms differ, the step shows a block for each.
Everywhere else the same commands work on all three.

### 1. Install Python 3.12 and git

Use **Python 3.10, 3.11 or 3.12**. lerobot pins `torch<2.8.0`, and there are
no builds of those torch versions for Python 3.13 or later. On a newer Python,
`pip install -e lerobot` fails with
`No matching distribution found for torch`, and running anything afterwards
fails with `No module named 'lerobot.robots'`.

Windows (PowerShell):

```powershell
winget install Python.Python.3.12
winget install Git.Git
```

Close and reopen PowerShell afterwards so it picks up the new commands.

macOS (with [Homebrew](https://brew.sh)):

```bash
brew install python@3.12 git
```

Linux (Debian/Ubuntu):

```bash
sudo apt install git python3.12 python3.12-venv
sudo usermod -aG dialout $USER    # lets you open the arm's serial port
```

Log out and back in for the `dialout` group to take effect. Ubuntu 22.04 has
no `python3.12` package: install `python3.10 python3.10-venv` instead, and
write `python3.10` wherever the steps below say `python3.12`.

### 2. Clone the repo

```bash
git clone --recurse-submodules https://github.com/maxernst38/neurobot.git
cd neurobot
```

lerobot and Seeed_RoboController are git submodules, which is why the clone
needs `--recurse-submodules`. If you already cloned without it, fetch them
with `git submodule update --init`.

### 3. Apply the lerobot patch

```bash
cd lerobot
git apply ../patches/lerobot-no-torch.patch
cd ..
```

Do this once per clone. Without the patch, importing the robot driver pulls in
torch. On Windows, torch's DLLs also need an MSVC runtime that a stock Python
install doesn't have. See [patches/README.md](patches/README.md) for what the
patch changes.

### 4. Create and activate a virtual environment

Windows (PowerShell):

```powershell
py -3.12 -m venv neurobot
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned    # once per machine; answer Y
.\neurobot\Scripts\Activate.ps1
```

PowerShell won't run scripts until you change the execution policy, and that
includes the venv's activation script.

macOS / Linux:

```bash
python3.12 -m venv neurobot
source neurobot/bin/activate
```

Your prompt now starts with `(neurobot)`. In every new terminal, `cd` into the
repo and run the activation line again before doing anything else.

### 5. Install the dependencies

With the venv active:

```bash
python -m pip install --upgrade pip
pip install -e lerobot
pip install -r arm_motion_tracker/requirements.txt
```

The first install downloads torch, so it takes a while.

### 6. Connect the hardware and check the arm

Plug in one webcam. Plug the arm in over USB and switch on its power supply.
Then:

```bash
python robot.py --check
```

This finds the arm's serial port, confirms every motor responds, and reports
whether the arm is calibrated. It doesn't move the arm. If it can't pick a
port, it lists the ones it found. Pass the right one with `--port`, e.g.
`--port COM3` on Windows, `--port /dev/ttyACM0` on Linux, or
`--port /dev/tty.usbmodem...` on macOS. Use the same `--port` in step 7.

### 7. Run teleop with one camera

```bash
python main.py --num-cameras 1
```

- **First run on an uncalibrated arm:** the terminal walks you through
  lerobot's motor calibration before anything else starts. Follow its prompts.
- Open <http://localhost:8080/> in a browser.
- Press `C` to connect to the arm. Line your arm up with the yellow ghost pose
  in the 3D view, then press `E` to engage and `R` to release.
- Press `Ctrl+C` in the terminal to stop. That shuts down both processes and
  lets the arm go limp.

If the wrong webcam opens, pick one by index with `--camera 1`. On macOS, the
first run asks for camera access for your terminal. Allow it, or turn it on
under System Settings → Privacy & Security → Camera. To try the tracker
without the arm, add `--no-robot`.

With one camera, depth is MediaPipe's estimate rather than a measurement, so
it's fine for gestures but unreliable for anything that needs real depth. Two
calibrated cameras measure depth properly. See
[Calibrating the cameras](#calibrating-the-cameras).

## Running

`main.py` is the entry point: it starts the robot bridge (`robot.py`) and the
camera tracker (`arm_motion_tracker/tracker.py`) together, and stops both on
Ctrl+C.

```bash
python main.py
python main.py --no-robot        # cameras only, no arm attached
python main.py --num-cameras 1   # one camera
```

Then open <http://localhost:8080/>. The page binds to localhost by default
because it commands a robot arm; `--web-host 0.0.0.0` is the deliberate opt-in
for reaching it from another device.

`--num-cameras` takes 2 (the default, and what triangulation needs), 1, or 0.
With one camera there is nothing to triangulate, so depth comes from
MediaPipe's monocular guess rather than from measurement — usable for
gestures, unreliable for anything needing real depth. With 0 the tracker opens
no cameras at all and serves the UI for replaying recordings.

Either half also runs on its own, which is what to do when you are working on
that half. The tracker needs no robot:

```bash
python arm_motion_tracker/tracker.py --web
python robot.py                  # arm only, jogged from the keyboard
```

Recalibrate the robot's motor ranges:

```bash
python robot.py --recalibrate
```

## Recording sessions

Captures are written to `recordings/` as JSON Lines — one header, one row per
frame — holding three layers of each instant: what each camera saw (2D
landmarks and confidence), where that put each joint (triangulated xyz), and
what that measured (joint angles). Keeping the raw detections is what lets an
old capture be re-processed against a new calibration or corrected code.

```bash
python main.py --record --record-note "trial 3"
python main.py --record --record-video              # + raw footage
python main.py --record --mask-face block           # anonymise
python main.py --record --record-compress           # ~20x smaller on disk
```

`main.py` passes every `--record*` flag through to the tracker, so the same
capture works with the arm attached or with `--no-robot`. Recording can also
be started and stopped from the page mid-session, and
`python arm_motion_tracker/tracker.py --web --record` records the tracker on
its own.

Play one back:

```bash
python main.py --replay session-<stamp>.jsonl
python main.py --replay session-<stamp>.jsonl --derive cameras
```

`--replay` implies `--no-robot` and opens no cameras, so a capture is reviewed
on whatever machine you have rather than at the rig. The page gains a
transport: scrub, play/pause (`K`), step a frame either way, 0.25x–4x, loop. A
recording can also be opened from the page mid-session, which is the way to
watch one while the cameras are still running.

`--derive` chooses how a replayed frame is read — `stored` as it was recorded,
`points` recomputed from the recorded 3D, `cameras` re-triangulated from the
recorded 2D with the calibration loaded now. It defaults to `points`, and the
three are switchable on the page while a replay is open; see
[Using the web tool](#using-the-web-tool).

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
`python robot.py --recalibrate`.

## Licence

lerobot is Apache-2.0 (Copyright 2024 The Hugging Face team); the local
modifications are marked in place in its source. Project code is this
repository's own.
