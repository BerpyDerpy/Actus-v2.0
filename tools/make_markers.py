"""Printable ArUco marker sheets and a board layout guide, from ``configs/rack.yaml``.

    uv run python tools/make_markers.py                    # -> data/markers/
    uv run python tools/make_markers.py --out prints/ --page letter

Writes SVG (vector, physical millimetre units), so a marker prints at exactly
``aruco.marker_size`` when printed at 100 % / "actual size" (not "fit to page"):

* ``markers_<page>_p<N>.svg``: the markers at true size, each with a white quiet zone, a dashed
  cut line, its id, where its centre goes on the board, and an arrow marking the top edge (stick
  it upright: arrow toward the board's top, rack −y). Each page has a 100 mm scale bar; measure
  it after printing.
* ``board_layout.svg``: a scaled plan of the board (markers, protocol zones, the outer-box
  footprint, rack axes) for building the rig. Not to scale.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from actus.spatial.rack_frame import DEFAULT_RACK_CONFIG, RackGeometry
from actus.spatial.zones import as_polygon

PAGES_MM = {"a4": (210.0, 297.0), "letter": (215.9, 279.4)}
MARGIN_MM = 12.0
LABEL_MM = 16.0


def marker_cells(dictionary: str, marker_id: int) -> np.ndarray:
    """Marker bit grid including the black border, 1 = white. Shape (n+2, n+2)."""
    import cv2

    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    n = int(d.markerSize) + 2
    img = cv2.aruco.generateImageMarker(d, int(marker_id), n)
    return (img > 127).astype(np.uint8)


def render_cells(cells: np.ndarray, cell_px: int = 20, quiet_cells: int = 1) -> np.ndarray:
    """Raster a cell grid (white quiet zone around it) — used to check a sheet decodes."""
    img = np.kron(cells, np.ones((cell_px, cell_px), np.uint8)) * 255
    q = quiet_cells * cell_px
    return np.pad(img, q, constant_values=255)


def _marker_svg(cells: np.ndarray, x: float, y: float, size: float) -> str:
    n = cells.shape[0]
    c = size / n
    parts = [f'<rect x="{x:.3f}" y="{y:.3f}" width="{size:.3f}" height="{size:.3f}" fill="#000"/>']
    for r in range(n):
        for k in range(n):
            if cells[r, k]:
                parts.append(
                    f'<rect x="{x + k * c:.3f}" y="{y + r * c:.3f}" width="{c:.3f}" '
                    f'height="{c:.3f}" fill="#fff"/>'
                )
    return "".join(parts)


def _svg(width_mm: float, height_mm: float, body: list[str]) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_mm}mm" height="{height_mm}mm" '
        f'viewBox="0 0 {width_mm} {height_mm}" shape-rendering="crispEdges" '
        'font-family="Helvetica, Arial, sans-serif">\n'
        f'<rect width="{width_mm}" height="{height_mm}" fill="#fff"/>\n'
        + "\n".join(body)
        + "\n</svg>\n"
    )


def marker_pages(geo: RackGeometry, page: str = "a4") -> list[str]:
    pw, ph = PAGES_MM[page]
    size = geo.marker_size * 1000.0
    cells = {mid: marker_cells(geo.dictionary, mid) for mid in sorted(geo.markers)}
    quiet = size / next(iter(cells.values())).shape[0]  # one cell of white around the marker
    cell_w = size + 2 * quiet + 8
    cell_h = size + 2 * quiet + LABEL_MM + 8
    cols = max(1, int((pw - 2 * MARGIN_MM) // cell_w))
    rows = max(1, int((ph - 2 * MARGIN_MM - 20) // cell_h))
    per_page = cols * rows
    ids = sorted(cells)
    pages = []
    for p0 in range(0, len(ids), per_page):
        body = [
            f'<text x="{MARGIN_MM}" y="{MARGIN_MM - 3}" font-size="4">ACTUS rack markers '
            f"{escape(geo.dictionary)}, {size:g} mm. Print at 100% (actual size).</text>"
        ]
        for j, mid in enumerate(ids[p0 : p0 + per_page]):
            cx = MARGIN_MM + (j % cols) * cell_w + 4
            cy = MARGIN_MM + (j // cols) * cell_h + 4
            body.append(
                f'<rect x="{cx:.2f}" y="{cy:.2f}" width="{size + 2 * quiet:.2f}" '
                f'height="{size + 2 * quiet:.2f}" fill="none" stroke="#999" stroke-width="0.2" '
                'stroke-dasharray="1.5,1.5"/>'
            )
            body.append(_marker_svg(cells[mid], cx + quiet, cy + quiet, size))
            mx, my = geo.markers[mid]
            ax = cx + quiet + size / 2
            body.append(
                f'<path d="M {ax - 3:.2f} {cy - 0.5:.2f} L {ax:.2f} {cy - 3.5:.2f} '
                f'L {ax + 3:.2f} {cy - 0.5:.2f} Z" fill="#c00"/>'
            )
            ty = cy + size + 2 * quiet + 5
            body.append(
                f'<text x="{cx:.2f}" y="{ty:.2f}" font-size="4" font-weight="bold">id {mid}</text>'
            )
            body.append(
                f'<text x="{cx:.2f}" y="{ty + 5:.2f}" font-size="3">centre at x={mx * 1000:.0f} '
                f"mm, y={my * 1000:.0f} mm; red arrow = top</text>"
            )
        sy = ph - MARGIN_MM
        body.append(
            f'<line x1="{MARGIN_MM}" y1="{sy}" x2="{MARGIN_MM + 100}" y2="{sy}" stroke="#000" '
            'stroke-width="0.5"/>'
        )
        for x in (MARGIN_MM, MARGIN_MM + 100):
            body.append(
                f'<line x1="{x}" y1="{sy - 2}" x2="{x}" y2="{sy + 2}" stroke="#000" '
                'stroke-width="0.5"/>'
            )
        body.append(
            f'<text x="{MARGIN_MM + 104}" y="{sy + 1.2}" font-size="3.5">100 mm: check with a '
            "ruler</text>"
        )
        pages.append(_svg(pw, ph, body))
    return pages


def board_layout(geo: RackGeometry, zones: dict[str, np.ndarray]) -> str:
    """A scaled plan of the board on an A4 landscape page."""
    pw, ph = 297.0, 210.0
    bw, bh = (v * 1000 for v in geo.board_size)
    s = min((pw - 40) / bw, (ph - 50) / bh)
    ox, oy = 20.0, 30.0

    def P(x_m: float, y_m: float) -> tuple[float, float]:
        return ox + x_m * 1000 * s, oy + y_m * 1000 * s

    def poly(pts, stroke, fill, label):
        pts_s = " ".join(f"{a:.2f},{b:.2f}" for a, b in (P(x, y) for x, y in pts))
        lx, ly = P(*np.min(pts, axis=0))
        return (
            f'<polygon points="{pts_s}" fill="{fill}" stroke="{stroke}" stroke-width="0.5"/>'
            f'<text x="{lx + 1.5:.2f}" y="{ly + 5:.2f}" font-size="4" fill="{stroke}">'
            f"{escape(label)}</text>"
        )

    body = [
        f'<text x="20" y="14" font-size="6" font-weight="bold">ACTUS rack board layout '
        f"({bw:.0f} x {bh:.0f} mm){'' if geo.measured else ' - PLACEHOLDER GEOMETRY'}</text>",
        '<text x="20" y="21" font-size="3.5">Not to scale. Rack frame: origin top-left, +x '
        "right, +y down, metres. Marker positions are centres.</text>",
        f'<rect x="{ox}" y="{oy}" width="{bw * s:.2f}" height="{bh * s:.2f}" fill="#f4f4f0" '
        'stroke="#000" stroke-width="0.6"/>',
    ]
    for name, pts in zones.items():
        body.append(poly(pts, "#1a7f37", "#1a7f3722", f"Zone {name}"))
    if geo.outer_box_footprint:
        body.append(poly(as_polygon(geo.outer_box_footprint), "#b35c00", "#b35c0022", "outer box"))
    for mid, (mx, my) in sorted(geo.markers.items()):
        h = geo.marker_size / 2
        x0, y0 = P(mx - h, my - h)
        side = geo.marker_size * 1000 * s
        body.append(f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{side:.2f}" height="{side:.2f}"/>')
        body.append(
            f'<text x="{x0:.2f}" y="{y0 + side + 4:.2f}" font-size="3.5">id {mid} '
            f"({mx * 1000:.0f}, {my * 1000:.0f})</text>"
        )
    x0, y0 = P(0, 0)
    body.append(
        f'<line x1="{x0}" y1="{y0}" x2="{x0 + 25}" y2="{y0}" stroke="#c00" stroke-width="0.8"/>'
        f'<text x="{x0 + 26}" y="{y0 + 1.5}" font-size="3.5" fill="#c00">+x</text>'
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + 25}" stroke="#080" stroke-width="0.8"/>'
        f'<text x="{x0 - 5}" y="{y0 + 30}" font-size="3.5" fill="#080">+y</text>'
    )
    return _svg(pw, ph, body)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Write printable ArUco marker sheets (SVG).")
    ap.add_argument("--rack", default=DEFAULT_RACK_CONFIG)
    ap.add_argument("--protocol", default="protocols/bas_boxes.yaml", help="zones for the layout")
    ap.add_argument("--out", type=Path, default=Path("data/markers"))
    ap.add_argument("--page", choices=sorted(PAGES_MM), default="a4")
    args = ap.parse_args(argv)

    from actus.core.config import resolve_path
    from actus.protocol.schema import load_protocol

    geo = RackGeometry.load(args.rack)
    zones = {k: as_polygon(v) for k, v in load_protocol(args.protocol).zones.items()}
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for i, svg in enumerate(marker_pages(geo, args.page), start=1):
        p = out / f"markers_{args.page}_p{i}.svg"
        p.write_text(svg, encoding="utf-8")
        written.append(p)
    p = out / "board_layout.svg"
    p.write_text(board_layout(geo, zones), encoding="utf-8")
    written.append(p)
    for p in written:
        print(f"wrote {p}")
    if not geo.measured:
        print("NOTE: configs/rack.yaml is placeholder geometry (measured: false).")
    print("Print the marker pages at 100% / actual size and check the 100 mm scale bar.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
