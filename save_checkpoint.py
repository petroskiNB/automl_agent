"""Archive an entire completed run, predictions and current sources without secrets."""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parent
    run = args.run.resolve()
    if not run.is_relative_to(project / "tasks") or run.parent.name != "agent_runs":
        raise ValueError("Expected a run under tasks/<task>/agent_runs")
    folder = run.parent.parent
    if (folder / ".agent.lock").exists():
        raise ValueError("Stop the running agent before taking a checkpoint")
    state = json.loads((run / "research/state.json").read_text(encoding="utf-8"))
    if not state.get("finalized"):
        raise ValueError("Checkpoint requires a completed final refit")
    from solution import discover_files, validate_submission
    valid, detail = validate_submission(folder, discover_files(folder))
    if not valid:
        raise ValueError(detail)
    out = project / "outputs"
    out.mkdir(exist_ok=True)
    name = f"{folder.name}_checkpoint_{datetime.now():%Y%m%d_%H%M%S_%f}"
    archive = out / (name + ".zip")
    paths = list(run.rglob("*")) + [folder / "submission.csv"]
    paths += list(project.glob("*.py")) + [project / "requirements.txt", project / "README.md", project / ".gitignore"]
    paths = sorted({p for p in paths if p.is_file()})
    manifest = {"run": str(run.relative_to(project)), "best": state["best"],
                "finalized": state["finalized"], "files": {}}
    with ZipFile(archive, "x", compression=ZIP_DEFLATED, compresslevel=1) as zipped:
        for path in paths:
            relative = path.relative_to(project).as_posix()
            zipped.write(path, relative)
            manifest["files"][relative] = {"sha256": digest(path), "bytes": path.stat().st_size}
        zipped.writestr("checkpoint_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    with ZipFile(archive) as zipped:
        corrupt = zipped.testzip()
        if corrupt:
            raise ValueError(f"Corrupt archive member: {corrupt}")
    manifest["archive_sha256"] = digest(archive)
    manifest["archive"] = archive.name
    (out / (name + ".json")).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"archive": str(archive), "files": len(paths), "best_score": state["best"]["score"],
                      "sha256": manifest["archive_sha256"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
