"""T-SPATIAL tools/make_markers.py: sheets decode, are true size, and are valid SVG."""

import importlib.util
import xml.etree.ElementTree as ET

import pytest

from actus.core.config import REPO_ROOT
from actus.spatial.rack_frame import RackGeometry

cv2 = pytest.importorskip("cv2")

_spec = importlib.util.spec_from_file_location(
    "make_markers", REPO_ROOT / "tools" / "make_markers.py"
)
mm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mm)

SVG = "{http://www.w3.org/2000/svg}"


@pytest.mark.parametrize("marker_id", [0, 1, 2, 3, 17])
def test_marker_cells_decode_to_their_id(marker_id):
    cells = mm.marker_cells("DICT_4X4_50", marker_id)
    assert cells.shape == (6, 6)
    assert cells[0].sum() == 0 and cells[:, 0].sum() == 0  # black border
    img = mm.render_cells(cells)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    _, ids, _ = cv2.aruco.ArucoDetector(d).detectMarkers(img)
    assert ids is not None and ids.ravel().tolist() == [marker_id]


def test_marker_page_is_true_size_svg():
    geo = RackGeometry.load()
    pages = mm.marker_pages(geo, "a4")
    assert len(pages) == 1  # 4 markers of 60 mm fit on one A4 page
    root = ET.fromstring(pages[0].split("\n", 1)[1])
    assert root.get("width") == "210.0mm" and root.get("viewBox") == "0 0 210.0 297.0"
    size = geo.marker_size * 1000
    black = [r for r in root.iter(f"{SVG}rect") if r.get("fill") == "#000"]
    assert len(black) == 4
    assert all(float(r.get("width")) == pytest.approx(size) for r in black)
    text = " ".join(t.text or "" for t in root.iter(f"{SVG}text"))
    assert "id 3" in text and "100 mm" in text


def test_cli_writes_sheets_and_layout(tmp_path, capsys):
    assert mm.main(["--out", str(tmp_path), "--page", "letter"]) == 0
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["board_layout.svg", "markers_letter_p1.svg"]
    for p in tmp_path.iterdir():
        ET.parse(p)  # well-formed XML
    assert "Zone A" in (tmp_path / "board_layout.svg").read_text(encoding="utf-8")
    assert "actual size" in capsys.readouterr().out
