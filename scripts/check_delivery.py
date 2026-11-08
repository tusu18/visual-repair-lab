"""Offline release integrity verification; no model calls."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'SHA256SUMS.json').read_text())
errors=[]
for name,expected in manifest.items():
 path=root/name
 if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:errors.append(name)
print(json.dumps({'passed':not errors,'files':len(manifest),'mismatches':errors},indent=2))
raise SystemExit(bool(errors))
