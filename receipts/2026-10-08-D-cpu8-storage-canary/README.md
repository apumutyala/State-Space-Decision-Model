# Private storage transfer canary

**Observed:** 2026-10-08, Autoresearch workspace `default`, `autor` 0.28.0.  
**Question:** does the Autoresearch object-storage `autor storage cp` path complete a minimal upload/download round-trip after the earlier direct node `autor cp` upload failure?  
**Scope:** synthetic 28-byte text only. This does not test large or multipart transfers, node-side `autor import`, the CPU-8 package, or a validation run.

The canary payload was exactly `ssdm-cpu8-storage-canary-v1\n`, 28 UTF-8 bytes, SHA-256
`a598f009d319f1d7ad723b591b50503a4b588ee8582fb329c80b8ee2bf1c98e1`.

Commands:

```sh
autor storage bucket create ssdm-cpu8-canary-20261008 --workspace default --json
printf 'ssdm-cpu8-storage-canary-v1\n' > /tmp/ssdm-cpu8-storage-canary.txt
sha256sum /tmp/ssdm-cpu8-storage-canary.txt
autor storage cp /tmp/ssdm-cpu8-storage-canary.txt ssdm-cpu8-canary-20261008/canary-v1/ssdm-cpu8-storage-canary.txt --workspace default --json
autor storage ls ssdm-cpu8-canary-20261008/canary-v1/ --workspace default --json
autor storage cp ssdm-cpu8-canary-20261008/canary-v1/ssdm-cpu8-storage-canary.txt /tmp/ssdm-cpu8-storage-canary-down.txt --workspace default --json
sha256sum /tmp/ssdm-cpu8-storage-canary-down.txt
cmp /tmp/ssdm-cpu8-storage-canary.txt /tmp/ssdm-cpu8-storage-canary-down.txt
autor storage rm ssdm-cpu8-canary-20261008/canary-v1/ssdm-cpu8-storage-canary.txt --workspace default --json
autor storage bucket rm ssdm-cpu8-canary-20261008 --workspace default --json
rm -f /tmp/ssdm-cpu8-storage-canary.txt /tmp/ssdm-cpu8-storage-canary-down.txt
```

**Result:** upload and download each reported 28 bytes; the remote listing contained exactly one object at 28 bytes; the downloaded SHA-256 matched and `cmp` exited 0. The object and temporary bucket were deleted. Audit recorded bucket create at `2026-10-08T02:33:31.836823Z`, delete at `2026-10-08T02:33:52.913191Z`, and 0 objects at deletion. A subsequent live billing read reported object-storage MTD $0, 0 active storage bytes, and no buckets. No node was created/woken, no model or Index/held-out inputs were used, and no compute or storage charge occurred.

This is evidence only for the tiny object-storage round-trip. It does not establish multi-part reliability for the roughly 500 MB package or prove that node-side `autor import` succeeds. Those require the separately reviewed CPU-8 attempt.
