import importlib.util
import zipfile
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "download_gsr.py"
spec = importlib.util.spec_from_file_location("download_gsr", SCRIPT)
download_gsr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download_gsr)


def _archive(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / "valid.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("valid/SNGS-001/img1/000001.jpg", b"frame")
        zf.writestr("valid/SNGS-001/Labels-GameState.json", b"{}")
    return archive


def test_extract_to_another_disk_and_mark_it_complete(tmp_path):
    archive = _archive(tmp_path / "volume")
    target = tmp_path / "local" / "gamestate-2024" / "valid"
    download_gsr.extract(archive, target)
    assert (target / "SNGS-001" / "img1" / "000001.jpg").read_bytes() == b"frame"
    assert (target / download_gsr.MARKER).exists()
    assert archive.exists()  # the zip stays where it is kept


def test_an_interrupted_unpack_is_redone(tmp_path):
    archive = _archive(tmp_path)
    target = tmp_path / "valid"
    (target / "SNGS-001").mkdir(parents=True)  # folder exists, but the unpack never finished
    download_gsr.extract(archive, target)
    assert (target / "SNGS-001" / "Labels-GameState.json").exists()
    assert (target / download_gsr.MARKER).exists()
