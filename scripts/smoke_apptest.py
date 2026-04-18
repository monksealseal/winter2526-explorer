"""Second-level smoke test: run the full streamlit script via AppTest,
which actually executes app.py (including the cube sanity check and all
tab bodies at the default query_params). Reports any exceptions."""
from pathlib import Path
import os, sys, traceback

# Run from repo root so relative data paths resolve, and add repo root
# to sys.path so ``from indices import ...`` in app.py works.
REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)
sys.path.insert(0, str(REPO))

from streamlit.testing.v1 import AppTest
at = AppTest.from_file("app.py", default_timeout=60)
try:
    at.run()
except Exception:
    print("AppTest threw:")
    traceback.print_exc()
    sys.exit(2)

# Collect any exceptions streamlit captured during the run
excs = at.exception
print(f"captured exceptions: {len(excs)}")
for e in excs:
    print("  --", e.value)
    if hasattr(e, "stack_trace"):
        print("     stack:", e.stack_trace)

# Also check for st.error calls (app-level soft failures)
errs = at.error
print(f"st.error calls: {len(errs)}")
for e in errs:
    print("  --", getattr(e, "value", e))

# Count main title occurrences to confirm render
titles = at.title
print(f"titles rendered: {len(titles)}")
for t in titles:
    print("  --", getattr(t, "value", t))

sys.exit(0 if (len(excs) == 0 and len(errs) == 0) else 1)
