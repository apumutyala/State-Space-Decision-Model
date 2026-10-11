"""Shim-prompt token lengths: public items (lengths only, eval-only, nothing else printed) vs jev1 train rows."""
import sys, json, gzip
from collections import defaultdict
sys.path.insert(0, r'C:\Users\apumu\research\CHLabs\ssdm-wt\jev1-diag')
import third_party.h2o_lightning_shim as H
from tokenizers import Tokenizer
mod, c = H.load_contract()
tok = Tokenizer.from_file(r'C:\Users\apumu\research\CHLabs\ssdm-data\tokenizers\Qwen--Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a\tokenizer.json')
def n(state, q):
    text = c.question_prompt(mod.render_state(state, c.state_compact), q)[0]
    return len(tok.encode(text, add_special_tokens=False).ids)
def q5(xs):
    xs = sorted(xs); k = len(xs)
    return dict(n=k, min=xs[0], p50=xs[k // 2], p90=xs[min(k - 1, int(.9 * k))], max=xs[-1], over2048=sum(x > 2048 for x in xs))
D = r'C:\Users\apumu\research\CHLabs\ssdm-data\jevbench\b6b8fff7e345b98c060ad26c13308860ddc67004'
pub = defaultdict(list)
for f in ('original', 'easy', 'hard'):
    for l in open(D + '\\' + f + '.jsonl', encoding='utf-8'):
        if l.strip():
            r = json.loads(l); pub[r['family']].append(n(r['state'], r['question']))
print('PUBLIC shim tokens by family')
for k, v in sorted(pub.items()): print(' ', k, q5(v))
allp = [x for v in pub.values() for x in v]; print('  ALL', q5(allp))
tr = defaultdict(list); cnt = defaultdict(int)
for l in gzip.open(r'C:\Users\apumu\research\CHLabs\ssdm-data\m2\build\bschema-draw-jev1\records.jsonl.gz', 'rt', encoding='utf-8'):
    r = json.loads(l); f8 = r['aug']['jev1']['family8']; cnt[f8] += 1
    if f8 == 'long_policy' or cnt[f8] % 10 == 0:
        tr[f8].append(n(r['state'], r['questions']['q']))
print('JEV1 shim tokens by family (long_policy all rows, others every 10th)')
for k, v in sorted(tr.items()): print(' ', k, q5(v))

