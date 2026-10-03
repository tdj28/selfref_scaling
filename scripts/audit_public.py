"""Pre-commit public-release audit. Prints findings, never secret values.

Checks every file in the Git index (or the paths given) for: credential-like
patterns, exact matches of the local secret values named in an env file read
by path, private-material file names, oversize files, and symlinks.
Exit status 1 on any finding.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "openai_key": re.compile(rb"sk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    "anthropic_key": re.compile(rb"sk-ant-[A-Za-z0-9_-]{20,}"),
    "hf_token": re.compile(rb"\bhf_[A-Za-z0-9]{30,}"),
    "runpod_key": re.compile(rb"\brpa_[A-Za-z0-9]{20,}"),
    "github_token": re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{30,}"),
    "private_key": re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "aws_key": re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
}
NAMES = re.compile(r"(^|/)(\.env(\..*)?|id_[a-z0-9]+|known_hosts|APPROVE-.*|.*\.pem|.*\.key)$")
SECRET_NAMES = ("RUNPOD_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "HF_TOKEN",
                "GOODFIRE_API_KEY", "STEERING_API_KEY")
MAX_BYTES = 50 * 1024 * 1024


def secret_values(env_path):
    values = []
    if env_path and Path(env_path).is_file():
        for line in Path(env_path).read_text().splitlines():
            key, _, value = line.partition("=")
            value = value.strip().strip('"').strip("'")
            if key.strip() in SECRET_NAMES and len(value) >= 12:
                values.append(value.encode())
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default=str(Path.home() / "PROJECTS/CONSCIOUS/.env"))
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()
    files = args.paths or subprocess.check_output(["git", "ls-files", "--cached"], cwd=ROOT, text=True).split()
    secrets = secret_values(args.env)
    findings = []
    for name in files:
        path = ROOT / name
        if NAMES.search(name):
            findings.append((name, "private_file_name"))
        if path.is_symlink():
            findings.append((name, "symlink"))
            continue
        if not path.is_file():
            continue
        if path.stat().st_size > MAX_BYTES:
            findings.append((name, "oversize"))
        raw = path.read_bytes()
        for label, pattern in PATTERNS.items():
            if pattern.search(raw):
                findings.append((name, label))
        if any(value in raw for value in secrets):
            findings.append((name, "local_secret_value"))
    for name, label in findings:
        print(f"FINDING {label}: {name}")
    print(f"audited {len(files)} files against {len(PATTERNS)} patterns and {len(secrets)} local secret values; "
          f"{len(findings)} findings")
    sys.exit(1 if findings else 0)


if __name__ == "__main__":
    main()
