"""Generate print-ready camera-calibration boards at exact physical size.

Printing is where calibration usually goes wrong: "fit to page" silently
rescales the pattern, and every distance you later compute inherits that
error. So each board is emitted as a PDF whose artwork is placed at exact
millimetre dimensions, alongside a printed 100 mm ruler you can measure to
confirm the print came out 1:1.

    python make_calibration_board.py --cols 6 --rows 4 --square-mm 45

Bigger squares beat more squares for a widely-separated camera pair. Both
cameras must resolve the SAME corners at once, and a board held between two
cameras 90 degrees apart is foreshortened in both views, so small markers
stop decoding well before they stop being visible.

Print at 100% / "actual size" (never "fit to page"), on matte paper, and
mount it on something rigid and flat -- foam board, clipboard, glass. A
curled sheet is not a plane, and the maths assumes a plane.
"""

import argparse
import os

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

PAPER_MM = {"A4": (297.0, 210.0), "A3": (420.0, 297.0), "LETTER": (279.4, 215.9)}
MM_PER_INCH = 25.4


def checkerboard_image(cols, rows, px_per_square=100):
    """Classic checkerboard, `cols` x `rows` *squares* (so cols-1 x rows-1
    inner corners, which is what OpenCV's findChessboardCorners wants)."""
    board = np.indices((rows, cols)).sum(axis=0) % 2
    img = np.kron(board, np.ones((px_per_square, px_per_square), dtype=np.uint8)) * 255
    return 255 - img


def charuco_board(cols, rows, square_mm, marker_ratio=0.72, px_per_square=100):
    """ChArUco: a checkerboard with ArUco markers in the white squares. Each
    marker is individually identifiable, so the board still calibrates when
    it is partly out of frame or steeply oblique -- which is what makes it
    the better choice for widely-separated cameras."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    board = cv2.aruco.CharucoBoard(
        (cols, rows), float(square_mm), float(square_mm) * marker_ratio, dictionary
    )
    img = board.generateImage((cols * px_per_square, rows * px_per_square), marginSize=0)
    return img, dictionary


def emit_pdf(path, img, board_w_mm, board_h_mm, paper, caption_lines):
    """Place `img` on the page at exactly board_w_mm x board_h_mm."""
    page_w, page_h = PAPER_MM[paper]
    if board_w_mm > page_w - 10 or board_h_mm > page_h - 25:
        raise SystemExit(
            f"Board {board_w_mm:.0f}x{board_h_mm:.0f} mm does not fit on {paper} "
            f"({page_w:.0f}x{page_h:.0f} mm). Reduce --square-mm or use --paper A3."
        )

    fig = plt.figure(figsize=(page_w / MM_PER_INCH, page_h / MM_PER_INCH))
    fig.patch.set_facecolor("white")

    left = (page_w - board_w_mm) / 2 / page_w
    bottom = (page_h - board_h_mm) / 2 / page_h + 0.035
    ax = fig.add_axes([left, bottom, board_w_mm / page_w, board_h_mm / page_h])
    ax.imshow(img, cmap="gray", interpolation="nearest", vmin=0, vmax=255, aspect="auto")
    ax.set_xticks([]), ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 100 mm verification ruler: if this does not measure 100 mm on the
    # printout, the print was scaled and every dimension below is wrong.
    ruler_mm = 100.0
    rx = (page_w - ruler_mm) / 2 / page_w
    rax = fig.add_axes([rx, 0.032, ruler_mm / page_w, 0.012])
    rax.set_xlim(0, 1), rax.set_ylim(0, 1)
    rax.set_xticks([]), rax.set_yticks([])
    for spine in rax.spines.values():
        spine.set_visible(False)
    rax.plot([0, 1], [0.5, 0.5], color="black", lw=1.0)
    for x in (0.0, 0.5, 1.0):
        rax.plot([x, x], [0.0, 1.0], color="black", lw=1.0)
    fig.text(0.5, 0.020, "^ this line must measure exactly 100 mm ^",
             ha="center", va="top", fontsize=7, family="monospace")

    fig.text(0.5, 0.965, caption_lines[0], ha="center", va="top",
             fontsize=10, family="monospace")
    fig.text(0.5, 0.945, caption_lines[1], ha="center", va="top",
             fontsize=7.5, family="monospace", color="0.35")

    with PdfPages(path) as pdf:
        pdf.savefig(fig)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--square-mm", type=float, default=45.0, help="side of one square, in mm")
    p.add_argument("--cols", type=int, default=6, help="squares across")
    p.add_argument("--rows", type=int, default=4, help="squares down")
    p.add_argument("--paper", choices=sorted(PAPER_MM), default="A4")
    p.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration"))
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    w_mm, h_mm = args.cols * args.square_mm, args.rows * args.square_mm
    inner = f"{args.cols - 1}x{args.rows - 1}"
    marker_mm = args.square_mm * 0.72

    chk = checkerboard_image(args.cols, args.rows)
    chk_path = os.path.join(args.out_dir, f"checkerboard_{args.cols}x{args.rows}_{args.square_mm:g}mm.pdf")
    emit_pdf(chk_path, chk, w_mm, h_mm, args.paper, [
        f"Checkerboard  {args.cols}x{args.rows} squares  |  {args.square_mm:g} mm squares",
        f"OpenCV pattern size (INNER corners) = {inner}    board {w_mm:g} x {h_mm:g} mm",
    ])

    cha_img, _ = charuco_board(args.cols, args.rows, args.square_mm)
    cha_path = os.path.join(args.out_dir, f"charuco_{args.cols}x{args.rows}_{args.square_mm:g}mm.pdf")
    emit_pdf(cha_path, cha_img, w_mm, h_mm, args.paper, [
        f"ChArUco  {args.cols}x{args.rows}  |  square {args.square_mm:g} mm  marker {marker_mm:.1f} mm  |  DICT_5X5_100",
        f"OpenCV pattern size (INNER corners) = {inner}    board {w_mm:g} x {h_mm:g} mm",
    ])

    for path in (chk_path, cha_path):
        print(f"wrote {path}")
    print(f"\nBoard {w_mm:g} x {h_mm:g} mm on {args.paper}.")
    print(f"Calibrate with:  --cols {args.cols} --rows {args.rows} "
          f"--square-mm {args.square_mm:g} --marker-mm {marker_mm:.1f}")
    print("(after measuring the printed ruler; if it is not 100 mm, scale both numbers by what you measured / 100)")


if __name__ == "__main__":
    main()
