#!/usr/bin/env python3
"""Find data that must not be pushed to GitHub: internal IPs, credentials, tokens, keys.

    python3 tools/check_sensitive.py [REPO_DIR ...]

Scans tracked and staged text files (git ls-files) of each repo and exits 1 if anything is found.
Allow a known-safe line by adding a regex to `.sensitive-allow` at the repo root (one per line),
e.g. a fake token in a unit test. Python 3.6+, stdlib only.
"""
import os
import re
import subprocess
import sys

PATTERNS = [
    ("internal IPv4", re.compile(r"(?<![\d.])(?:10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])|192\.168|129\.254)\.\d{1,3}\.\d{1,3}(?![\d.])")),
    ("GitHub token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}")),
    ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{30,}")),
    ("API key", re.compile(r"\b(?:sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16})\b")),
    ("private key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |)PRIVATE KEY-----")),
    ("kubeconfig credential", re.compile(r"\b(?:client-key-data|client-certificate-data|token):\s*[A-Za-z0-9+/=_-]{20,}\s*$")),
    ("password", re.compile(r"(?i)\b(?:password|passwd|pwd)\b\s*[:=]\s*['\"]?(?!\$|<|\{|changeme|example|your)[^\s'\"]{4,}")),
    ("URL with credentials", re.compile(r"https?://[^/\s:@]+:(?!\$|<|\{)[^/\s@]+@")),
]
# Documentation and unit-test ranges are safe to publish (RFC 5737 / RFC 3849).
SAFE = re.compile(r"\b(?:192\.0\.2|198\.51\.100|203\.0\.113)\.\d+")
BINARY = re.compile(r"\.(png|jpe?g|gif|webp|pdf|pptx?|docx?|xlsx?|hwp|zip|gz|tgz|tar|whl|so|a|o|bin|rbln|fxb|edf|pt|safetensors|lock|ico|svgz)$", re.I)


def tracked(repo):
    out = subprocess.run(["git", "-C", repo, "ls-files", "-z", "--cached"], stdout=subprocess.PIPE)
    return [p for p in out.stdout.decode("utf-8", "replace").split("\0") if p]


def allowlist(repo):
    path = os.path.join(repo, ".sensitive-allow")
    if not os.path.exists(path):
        return []
    return [re.compile(l.strip()) for l in open(path, encoding="utf-8") if l.strip() and not l.startswith("#")]


def scan(repo):
    allow = allowlist(repo)
    findings = []
    for rel in tracked(repo):
        if BINARY.search(rel):
            continue
        path = os.path.join(repo, rel)
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                lines = f.read().splitlines()
        except (OSError, IsADirectoryError):
            continue
        for n, line in enumerate(lines, 1):
            if any(a.search(line) for a in allow):
                continue
            for kind, pat in PATTERNS:
                m = pat.search(line)
                if m and not SAFE.search(m.group(0)):
                    findings.append((rel, n, kind, m.group(0)[:48]))
    return findings


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repos = sys.argv[1:] or [here]
    total = 0
    for repo in repos:
        found = scan(repo)
        total += len(found)
        print("%s: %s" % (repo, "clean" if not found else "%d finding(s)" % len(found)))
        for rel, n, kind, value in found[:200]:
            print("  %s:%d  %-22s %s" % (rel, n, kind, value))
    if total:
        print("\nMove site-specific values (IPs, hostnames with IPs, endpoints, credentials) into an "
              "untracked file such as cluster.local.env and read them from the environment; keep "
              "placeholders in cluster.example.env. Use 192.0.2.x for documentation examples.")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
