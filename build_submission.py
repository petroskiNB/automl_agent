"""Build an allowlisted submission archive; no keys, models or data included."""
import hashlib
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def main():
    root = Path(__file__).resolve().parent
    names = ("solution.py", "research.py", "requirements.txt")
    for name in names:
        path = root / name
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Missing or empty file: {name}")
        if path.suffix == ".py":
            compile(path.read_text(encoding="utf-8-sig"), name, "exec")
    destination = root / "outputs" / "automl_submission.zip"
    destination.parent.mkdir(exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(root / name, arcname=name)
    with ZipFile(temporary) as archive:
        assert archive.namelist() == list(names)
        assert archive.testzip() is None
        for name in names:
            assert archive.read(name) == (root / name).read_bytes()
    if temporary.stat().st_size > 100 * 1024 * 1024:
        raise ValueError("Archive exceeds 100 MiB")
    temporary.replace(destination)
    print(f"Created: {destination}")
    print(f"Files: {', '.join(names)}")
    print(f"Size: {destination.stat().st_size} bytes")
    print(f"SHA256: {hashlib.sha256(destination.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
