import json,sys
j=json.load(open(sys.argv[1],encoding='utf-8')); v=j['v16']
for k in ('long_items_note','item_sets','counts','a2_supplement','set_manifest_sha256','packaging_partial','release'):
    print('##',k, json.dumps(v.get(k),ensure_ascii=False))
print('## sample',json.dumps(j['sample'])); print('## types',json.dumps(j['types'])); print('## protocol',j['protocol'],j['revision'])
for s in j['systems']:
    cb=s['cost'].get('common_basis') or {}
    if cb.get('rule'): print('## common_basis example',s['key'],json.dumps(cb)); break
import collections
print('## v16.n_items/n_public/n_sealed_side', collections.Counter((s['v16']['n_items'],s['v16']['n_public'],s['v16']['n_sealed_side'],s['v16']['set']) for s in j['systems']).most_common(10))
for s in j['systems']:
    if s['key'] in ('h2o-lightning-4b','h2o-lightning-4b-v1-1') or 'h2o' in s['key']: print('## h2o row', s['key'], s['status'], s['v16']['n_items'], s['v16']['set'], s['axes'])
for a in j.get('amendments',[]): print('## amend',a['date'],a['text'][:600])
