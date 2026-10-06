import json, hashlib, sys, os
m = json.load(open('/mnt/c/Users/apumu/research/CHLabs/ssdm-data/scripts/hub-manifest-0.1.json'))
root = os.path.expanduser('~/ssdm-data/work')
bad = 0
for b in m['benchmarks']:
    for s in b['sources']:
        p = os.path.join(root, s['path'])
        h = hashlib.sha256(open(p, 'rb').read()).hexdigest() if os.path.exists(p) else 'MISSING'
        if h != s['sha256']:
            bad += 1
            print(b['catalog_id'], b['dataset'], s['path'].split('/')[-1], 'local', h[:12], 'ref', s['sha256'][:12])
print('mismatched files', bad, 'of', sum(len(b['sources']) for b in m['benchmarks']))
# also compare counts
