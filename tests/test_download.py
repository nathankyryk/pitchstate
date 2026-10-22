import importlib.util
import zipfile
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "download_gsr.py"
spec = importlib.util.spec_from_file_location("download_gsr", SCRIPT)
download_gsr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download_gsr)


def _archive(tmp_path: Path) -> Path:
    archive = tmp_path / "valid.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("valid/SNGS-001/img1/000001.jpg", b"frame")
        zf.writestr("valid/SNGS-001/Labels-GameState.json", b"{}")
    return archive


def test_extract_marks_a_complete_unpack(tmp_path):
    archive = _archive(tmp_path)
    download_gsr.extract(archive)
    assert (tmp_path / "valid" / "SNGS-001" / "img1" / "000001.jpg").read_bytes() == b"frame"
    assert (tmp_path / "valid" / download_gsr.MARKER).exists()


def test_an_interrupted_unpack_is_redone(tmp_path):
    archive = _archive(tmp_path)
    partial = tmp_path / "valid" / "SNGS-001"
    partial.mkdir(parents=True)  # folder exists, but the unpack never finished (no marker)
    download_gsr.extract(archive)
    assert (partial / "Labels-GameState.json").exists()
    assert (tmp_path / "valid" / download_gsr.MARKER).exists()
