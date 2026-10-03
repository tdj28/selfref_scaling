"""Post-run public audit of a release directory or file list. Prints findings, never values.

Like scripts/audit_public.py (frozen), plus: key patterns require a token
boundary (Astra's opaque encrypted reasoning blobs can contain "sk-" mid-string),
and local absolute paths are findings. Exact local secret values are checked.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

B = rb"(?<![A-Za-z0-9_-])"
PATTERNS = {
    "openai_key": re.compile(B + rb"sk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    "anthropic_key": re.compile(B + rb"sk-ant-[A-Za-z0-9_-]{20,}"),
    "hf_token": re.compile(B + rb"hf_[A-Za-z0-9]{30,}"),
    "runpod_key": re.compile(B + rb"rpa_[A-Za-z0-9]{20,}"),
    "github_token": re.compile(B + rb"gh[pousr]_[A-Za-z0-9]{30,}"),
    "private_key": re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "ssh_public_key": re.compile(rb"ssh-(?:ed25519|rsa) AAAA"),
    "local_path": re.compile(re.escape(str(Path.home()).encode())),
}
NAMES = re.compile(r"(^|/)(\.env(\..*)?|hf\.env|id_[a-z0-9]+|known_hosts|APPROVE-.*|.*\.pem|.*\.key)$")
SECRET_NAMES = ("RUNPOD_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "HF_TOKEN", "GOODFIRE_API_KEY", "STEERING_API_KEY")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--env", default=str(Path.home() / "PROJECTS/CONSCIOUS/.env"))
    parser.add_argument("roots", nargs="+", type=Path)
    args = parser.parse_args()
    secrets = []
    if Path(args.env).is_file():
        for line in Path(args.env).read_text().splitlines():
            key, _, value = line.partition("=")
            value = value.strip().strip('"').strip("'")
            if key.strip() in SECRET_NAMES and len(value) >= 12:
                secrets.append(value.encode())
    files = [p for root in args.roots for p in ([root] if root.is_file() else sorted(root.rglob("*"))) if p.is_file() or p.is_symlink()]
    findings = []
    for path in files:
        if path.is_symlink():
            findings.append((path, "symlink"))
            continue
        if NAMES.search(path.as_posix()):
            findings.append((path, "private_file_name"))
        raw = path.read_bytes()
        findings += [(path, label) for label, pattern in PATTERNS.items() if pattern.search(raw)]
        if any(value in raw for value in secrets):
            findings.append((path, "local_secret_value"))
    for path, label in findings:
        print(f"FINDING {label}: {path}")
    print(f"audited {len(files)} files against {len(PATTERNS)} patterns and {len(secrets)} local secret values; "
          f"{len(findings)} findings")
    sys.exit(1 if findings else 0)


if __name__ == "__main__":
    main()
