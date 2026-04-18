"""Smoke test: boot streamlit headless, GET /, assert HTTP 200, no traceback in log."""
import subprocess, sys, time, urllib.request, urllib.error, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

log_path = ROOT / "_smoke_stdout.log"
port = 8521

with open(log_path, "wb") as log:
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py",
         "--server.headless", "true",
         "--server.port", str(port),
         "--browser.gatherUsageStats", "false"],
        stdout=log, stderr=subprocess.STDOUT,
    )

try:
    url = f"http://127.0.0.1:{port}/_stcore/health"
    # streamlit exposes /_stcore/health for HTTP liveness (returns 200 "ok")
    ok = False
    err = None
    for i in range(40):  # up to ~40 seconds
        time.sleep(1)
        if proc.poll() is not None:
            err = f"streamlit exited early with code {proc.returncode}"
            break
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                status = r.status
                body = r.read(128).decode("utf-8", errors="replace")
                print(f"HTTP {status} {url} after {i+1}s : {body!r}")
                if status == 200:
                    ok = True
                    break
        except urllib.error.URLError as e:
            err = f"{type(e).__name__}: {e}"
            continue
    if not ok:
        print("SMOKE FAIL:", err)
finally:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()

# Scan log for tracebacks
with open(log_path, encoding="utf-8", errors="replace") as f:
    log_text = f.read()

has_trace = "Traceback" in log_text or "Exception" in log_text
print("\n### streamlit log (last 60 lines) ###")
for line in log_text.splitlines()[-60:]:
    print(line)

print("\n### RESULT ###")
print("boot OK:", ok)
print("traceback in log:", has_trace)
sys.exit(0 if (ok and not has_trace) else 1)
