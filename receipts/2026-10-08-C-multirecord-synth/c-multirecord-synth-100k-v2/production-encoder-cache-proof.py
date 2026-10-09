import builtins
import ctypes
from ctypes import wintypes
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sys
import time

REPO = Path('C:/Users/apumu/research/CHLabs/ssdm-wt/agent-c-multirecord-synth')
TOK = Path('C:/Users/apumu/research/CHLabs/ssdm-data/tokenizers/RWKV--RWKV7-G1k-13.3B-20260930@ce3630b854cedbd673569f4600c6cf744bca9dba')
sys.path.insert(0, str(REPO))
sys.path.insert(1, 'C:/Users/apumu/research/CHLabs/ssdm-data/decision-index')
os.environ.update(USE_TORCH='0', USE_TF='0', USE_FLAX='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
original_import = builtins.__import__
def no_torch(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise RuntimeError('Torch forbidden in tokenizer-only benchmark')
    return original_import(name, *args, **kwargs)
builtins.__import__ = no_torch

from data.c_multirecord_synth import _load_encoder, _load_layout_without_package_initializers, make_record, canonical, TEMPLATE, LAYOUT, BoundedEncodingCache
build_row = _load_layout_without_package_initializers(REPO)
tokenizer, underlying = _load_encoder(TOK)

class CountEncoder:
    def __init__(self):
        self.calls = 0
    def __call__(self, text, trusted=True):
        self.calls += 1
        return underlying(text, trusted)

class CacheEncoder:
    def __init__(self, maxsize):
        self.underlying = CountEncoder()
        self.calls = 0
        self.cached = lru_cache(maxsize=maxsize)(lambda text, trusted: tuple(self.underlying(text, trusted)))
    def __call__(self, text, trusted=True):
        self.calls += 1
        return list(self.cached(text, trusted))

class ProcessMemory(ctypes.Structure):
    _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
               ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
               ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
               ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
               ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
def memory():
    data = ProcessMemory()
    data.cb = ctypes.sizeof(data)
    ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemory), wintypes.DWORD]
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    if not ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(data), data.cb):
        raise ctypes.WinError()
    return {'rss_bytes': data.WorkingSetSize, 'peak_rss_bytes': data.PeakWorkingSetSize}

# Spread over the K sweep, ID styles, and later gold-position cycles.
indices = [style * 76 + (i * 75 // 10) + 228 * (i % 3)
           for style in range(3) for i in range(11)][:32]
records = [make_record(20261008, index, revision=2) for index in indices]
expected = {}
reports = {}
for name, encoder in [('baseline', CountEncoder()), ('production', BoundedEncodingCache(CountEncoder()))]:
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    rows = 0
    digest = hashlib.sha256()
    for index, record in zip(indices, records):
        request = {'model': 'm2', 'state': record['state'], 'questions': record['questions']}
        for qid in record['questions']:
            row = asdict(build_row(request, qid, encoder, template=TEMPLATE, layout=LAYOUT))
            key = (index, qid)
            if name == 'baseline':
                expected[key] = row
            else:
                assert row == expected[key], (name, key)
            digest.update(canonical(row))
            rows += 1
        serialized_ids = encoder(canonical(record).decode('utf-8'))
        key = (index, '__serialized_record__')
        if name == 'baseline':
            expected[key] = serialized_ids
        else:
            assert serialized_ids == expected[key], (name, key)
        digest.update(canonical(serialized_ids))
    report = {'wall_seconds': time.perf_counter()-wall_start, 'cpu_seconds': time.process_time()-cpu_start,
              'rows': rows, 'encoder_calls': encoder.calls if isinstance(encoder, CountEncoder) else encoder.hits + encoder.misses, 'complete_outputs_sha256': digest.hexdigest(), **memory()}
    if isinstance(encoder, BoundedEncodingCache):
        info = __import__('types').SimpleNamespace(hits=encoder.hits, misses=encoder.misses, maxsize=encoder.max_entries, currsize=len(encoder.entries))
        report.update(tokenizer_calls=encoder.encode.calls, accounted_cache_bytes=encoder.bytes, byte_bound=encoder.max_bytes, cache_hits=info.hits, cache_misses=info.misses,
                      cache_bound=info.maxsize, cache_entries=info.currsize, hit_rate=info.hits/(info.hits+info.misses))
        # Returned lists cannot mutate cached tuples; trusted/untrusted requests have distinct keys.
        first = encoder('hello benchmark', True)
        saved = list(first)
        first.append(987654321)
        assert encoder('hello benchmark', True) == saved
        before = encoder.misses
        assert encoder('hello benchmark', False) == saved
        assert encoder.misses == before + 1
    else:
        report['tokenizer_calls'] = encoder.calls
    reports[name] = report
assert len({r['complete_outputs_sha256'] for r in reports.values()}) == 1
for name in ('production',):
    reports[name]['wall_speedup_vs_baseline'] = reports['baseline']['wall_seconds']/reports[name]['wall_seconds']
    reports[name]['cpu_speedup_vs_baseline'] = reports['baseline']['cpu_seconds']/reports[name]['cpu_seconds']
report = {'scope':'bounded tokenizer-only production build_row + serialized-record census; no model/Index claim',
          'seed':20261008, 'indices':indices, 'records':len(records), 'q_rows':len(records)*7,
          'k_values':sorted({len(r['questions'][next(q for q,m in r['aug']['question_roles'].items() if m['kind']=='top_profile')]['criteria']) for r in records}),
          'id_styles':sorted({m['id_style'] for r in records for m in r['aug']['question_roles'].values() if m['kind']=='top_profile'}),
          'all_complete_layout_rows_equal':True, 'serialized_record_encodings_equal':True,
          'cache_return_alias_and_trust_key_checks_passed':True, 'reports':reports,
          'torch_loaded':any(n=='torch' or n.startswith('torch.') for n in sys.modules)}
path = Path(os.environ['TEMP'])/'c-production-encoder-cache-proof.json'
path.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,sort_keys=True,indent=2),flush=True)

