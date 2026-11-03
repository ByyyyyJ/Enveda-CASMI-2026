"""Wait for a kernel version to finish, submit its output, then wait for the public score.

Usage: python scripts/submit_when_done.py KERNEL VERSION MESSAGE   (kaggle CLI on PATH, e.g. conda run -n casmi2026)
"""
import subprocess
import sys
import time

COMP = "enveda-CASMI26-molecule-id-mass-spectra"
kernel, version, msg = sys.argv[1], sys.argv[2], sys.argv[3]


def run(*args):
    r = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    return (r.stdout or "") + (r.stderr or "")


def log(s):
    print(time.strftime("%H:%M:%S"), s, flush=True)


while True:
    out = run("kernels", "status", kernel)
    if "COMPLETE" in out.upper():
        log("kernel complete"); break
    if "ERROR" in out.upper() or "CANCEL" in out.upper():
        log("kernel failed: " + out.strip()); sys.exit(1)
    time.sleep(120)

log("submit: " + run("competitions", "submit", COMP, "-k", kernel, "-v", str(version), "-f", "submission.csv",
                     "-m", msg).strip().splitlines()[-1])
time.sleep(60)
while True:
    out = run("competitions", "submissions", COMP)
    line = next((l for l in out.splitlines() if msg[:40] in l), "")
    if line and "PENDING" not in line.upper():
        log(line); break
    time.sleep(300)
