"""Assemble a public release directory from verified local runtime outputs.

Copies only allowlisted subtrees, refuses symlinks and private file names,
verifies the final Qwen retrieval against the controller's receipt, and writes
MANIFEST.json (path, bytes, SHA-256 for every file). It never edits frozen
inputs and never overwrites an existing release directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from selfref_scaling.common import canonical, sha  # noqa: E402

PRIVATE = re.compile(r"(^|/)(\.env(\..*)?|hf\.env|id_[a-z0-9]+|known_hosts|APPROVE-.*|.*\.pem|.*\.key|"
                     r"\.worker\.lock|\.judge\.lock|\.mini\.lock|.*\.lock)$")


def copy_tree(source, target, skipped):
    for path in sorted(Path(source).rglob("*")):
        relative = path.relative_to(source).as_posix()
        if path.is_symlink():
            raise ValueError("Symlink in release input: " + relative)
        if path.is_dir():
            continue
        if PRIVATE.search(relative):
            skipped.append(relative)
            continue
        destination = Path(target) / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)


def verified_final_retrieval(controller_dir):
    receipt = json.loads((Path(controller_dir) / "final-retrieval.json").read_text())
    data = receipt["data"]
    directory = Path(data["directory"])
    for name, digest in data["artifacts"].items():
        path = directory / name
        if path.is_symlink() or not path.is_file() or sha(path) != digest:
            raise ValueError("Retrieved artifact missing or changed: " + name)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=Path("out/scaling-qwen-20261003"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, action="append", default=[],
                        help="additional analysis output directories (name=path)")
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Release directory already exists; releases are never overwritten")
    skipped = []
    sources = {}
    for kind in ("cheap", "main"):
        controller = args.runtime / "controller" / kind
        if (controller / "final-retrieval.json").exists():
            sources[f"qwen_{kind}_pod"] = verified_final_retrieval(controller)
            sources[f"controller_{kind}"] = controller / "events.jsonl"
    for name in ("api", "judges"):
        if (args.runtime / name).is_dir():
            sources[name] = args.runtime / name
    for item in args.analysis:
        name, _, path = str(item).partition("=")
        sources[name] = Path(path)
    for name, source in sources.items():
        target = args.out / name
        if source.is_file():
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target / source.name)
        else:
            copy_tree(source, target, skipped)
    files = sorted(p for p in args.out.rglob("*") if p.is_file())
    manifest = {"schema": "selfref_scaling_release_v1", "skipped_private": sorted(skipped),
                "files": [{"path": p.relative_to(args.out).as_posix(), "bytes": p.stat().st_size,
                           "sha256": sha(p)} for p in files]}
    (args.out / "MANIFEST.json").write_text(canonical(manifest) + "\n")
    print(canonical({"files": len(files), "skipped_private": len(skipped)}))


if __name__ == "__main__":
    main()
