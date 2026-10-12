"""List HF repo files/sizes/licence/sha for candidate parity systems (data only)."""
import json
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")
for rid in sys.argv[1:]:
    url = f"https://huggingface.co/api/models/{rid}?blobs=true"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            m = json.load(r)
    except Exception as e:  # noqa: BLE001
        print(f"## {rid} FAIL {e}")
        continue
    sib = m.get("siblings", [])
    tot = sum((s.get("size") or 0) for s in sib if s["rfilename"].endswith((".safetensors", ".gguf", ".bin")))
    lic = (m.get("cardData") or {}).get("license")
    base = (m.get("cardData") or {}).get("base_model")
    print(f"## {rid} sha={m.get('sha')} gated={m.get('gated')} lic={lic} base={base} weights_GB={tot/1e9:.2f} lastMod={m.get('lastModified')}")
    print("   files:", ", ".join(f"{s['rfilename']}({(s.get('size') or 0)//1000000}MB)" for s in sib)[:1500])
