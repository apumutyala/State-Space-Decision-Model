import json,sys,os,collections
d=sys.argv[1]; c=collections.Counter(); e=collections.Counter()
for f in ('original.jsonl','easy.jsonl','hard.jsonl'):
    for line in open(os.path.join(d,f),encoding='utf-8'):
        r=json.loads(line); q=r['question']
        if q['type']!='score': continue
        cr=q['criteria']; c[(f,r['family'],type(cr).__name__,len(cr),tuple(r['labels']), type(cr[0]).__name__ if isinstance(cr,list) else 'keys='+','.join(sorted(cr)))]+=1
        e[(len(r['labels']),r['expected'])]+=1
for k,v in sorted(c.items()): print(k,v)
print('expected level dist (K,level):',sorted(e.items()))
