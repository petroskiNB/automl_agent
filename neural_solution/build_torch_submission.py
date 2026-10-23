"""Build a separate allowlisted PyTorch ZIP, never overwrite an archive."""
# Source checkout delegates to the complete, versioned runtime.
if __name__ == "__main__":
    import sys as _sys
    from pathlib import Path as _Path
    if not _Path(__file__).with_name("solution.py").exists():
        from runtime import main as _launch
        _sys.argv.insert(1, 'build')
        _launch()

import argparse
import hashlib
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


BOOTSTRAP = '''"""Experimental PyTorch entry point in a clean evaluation container."""
import sys
from solution_torch import main

if __name__ == "__main__":
    sys.argv.append("--platform")
    main()
'''


def build(destination):
    root = Path(__file__).resolve().parent
    names = ["solution_torch.py", "torch_resources.py", "torch_model.py", "torch_backend.py", "torch_suites.py",
             "research.py", "xgb_backend.py", "diagnostics.py", "temporal_research.py", "classification_research.py",
             "classification_checks.py", "final_output.py", "parallel_research.py", "research_team.py", "requirements.txt"]
    contents = {"solution.py": BOOTSTRAP.encode(), "legacy_solution.py": (root/"solution.py").read_bytes()}
    contents.update({name: (root/name).read_bytes() for name in names})
    for name, source in contents.items():
        if name.endswith(".py"):
            compile(source.decode("utf-8-sig"), name, "exec")
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "x", compression=ZIP_DEFLATED) as archive:
        for name, source in contents.items():
            archive.writestr(name, source)
    with ZipFile(destination) as archive:
        if archive.testzip() is not None or archive.namelist() != list(contents):
            raise RuntimeError("Archive verification failed")
        if any(archive.read(name) != data for name, data in contents.items()):
            raise RuntimeError("Archive source mismatch")
    return dict(path=str(destination), bytes=destination.stat().st_size,
                sha256=hashlib.sha256(destination.read_bytes()).hexdigest())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent/"outputs"/"automl_pytorch_v1.zip")
    print(build(parser.parse_args().output))
