"""Read-only, standard-library artifact verification; run from any working directory."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / "provenance/delivery_hashes.json").read_text())
for relative, expected in manifest.items():
    actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    assert actual == expected, relative
status = json.loads((root / "verification.json").read_text())
assert status["all_passed"]
for batch in status["completed_batches"]:
    assert json.loads((root / batch / "independent_verification.json").read_text())["all_passed"]
print(json.dumps({"all_passed": True, "verified_files": len(manifest),
                  "completed_batches": status["completed_batches"]}))
