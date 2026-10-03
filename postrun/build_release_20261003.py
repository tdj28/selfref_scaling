"""Post-run release assembly for the 2026-10-03 run (original cheap attempt plus A1).

Reuses the frozen helpers in scripts/build_release.py (allowlisted copy,
private-name exclusion, verified final retrieval) for the three owned pods:
the failed original cheap attempt, the A1 cheap pod and the A1 main pod. Adds
the paid-call ledgers, the derived judgments, analysis and lens outputs, and
writes MANIFEST.json. Never overwrites an existing release directory.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from selfref_scaling.common import canonical, sha  # noqa: E402

spec = importlib.util.spec_from_file_location("frozen_release", ROOT / "scripts" / "build_release.py")
frozen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)

RUNTIME = ROOT / "out" / "scaling-qwen-20261003"
PODS = {"original_cheap": RUNTIME / "controller" / "cheap",
        "a1_cheap": RUNTIME / "a1" / "controller" / "cheap",
        "a1_main": RUNTIME / "a1" / "controller" / "main"}


REDACTED = "[redacted: account or SSH material]"


def _scrub(value):
    """Remove foreign-pod inventory, SSH endpoints and public keys; keep everything else."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key == "blocked" and isinstance(item, list):
                ids = sorted(map(str, item))
                out[key] = {"count": len(ids), "sha256_sorted_ids": hashlib.sha256("\n".join(ids).encode()).hexdigest()}
            elif key in ("ssh", "PUBLIC_KEY", "preexisting_ids"):
                out[key] = REDACTED
            else:
                out[_scrub(key)] = _scrub(item)  # keys can be local artifact paths
        return out
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    if isinstance(value, str):
        return value.replace(str(ROOT), "<repo>").replace(str(Path.home()), "<home>")
    return value


def redacted_ledger(source, target):
    """A redacted view (hash chain not preserved) plus the original file's SHA-256."""
    rows = [json.loads(line) for line in Path(source).read_text().splitlines() if line.strip()]
    view = [dict(deepcopy(r), data=_scrub(r["data"])) for r in rows]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(canonical(r) + "\n" for r in view))
    (target.parent / "ledger-redaction.json").write_text(canonical({
        "original_sha256": sha(source), "rows": len(rows),
        "note": "Redacted view: foreign pod IDs (count and digest kept), SSH endpoints and public key removed; "
                "row hashes refer to the unredacted local original."}) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--extra", action="append", default=[], help="name=path of an analysis output directory")
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Release directory already exists; releases are never overwritten")
    skipped = []
    for name, controller in PODS.items():
        if not (controller / "final-retrieval.json").exists():
            raise ValueError("Missing verified final retrieval: " + name)
        frozen.copy_tree(frozen.verified_final_retrieval(controller), args.out / "pods" / name / "outputs", skipped)
        target = args.out / "pods" / name / "controller"
        target.mkdir(parents=True, exist_ok=True)
        redacted_ledger(controller / "events.jsonl", target / "events.redacted.jsonl")
        receipt = json.loads((controller / "final-retrieval.json").read_text())
        receipt["data"] = dict(receipt["data"], directory="../outputs")  # local absolute path removed
        (target / "final-retrieval.relative.json").write_text(canonical(receipt) + "\n")
    basis = RUNTIME / "a1" / "approval-cheap-basis.json"
    shutil.copy2(basis, args.out / "pods" / "a1_cheap" / "approval-cheap-basis.json")
    for name in ("api", "judges"):
        if (RUNTIME / name).is_dir():
            frozen.copy_tree(RUNTIME / name, args.out / name, skipped)
    for item in args.extra:
        name, _, path = item.partition("=")
        frozen.copy_tree(Path(path), args.out / name, skipped)
    compressed = []
    for path in sorted(p for p in args.out.rglob("*") if p.is_file() and p.stat().st_size > 20 * 1024 ** 2):
        original = {"path": path.relative_to(args.out).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)}
        with path.open("rb") as source, gzip.GzipFile(path.with_name(path.name + ".gz"), "wb", mtime=0) as target:
            shutil.copyfileobj(source, target)
        path.unlink()
        compressed.append(original)
    files = sorted(p for p in args.out.rglob("*") if p.is_file())
    manifest = {"schema": "selfref_scaling_release_20261003", "skipped_private": sorted(skipped),
                "gzipped_originals": compressed,
                "files": [{"path": p.relative_to(args.out).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)}
                          for p in files]}
    (args.out / "MANIFEST.json").write_text(canonical(manifest) + "\n")
    print(canonical({"files": len(files), "skipped_private": len(skipped)}))


if __name__ == "__main__":
    main()
