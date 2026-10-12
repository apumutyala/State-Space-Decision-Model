import json,sys,os,collections
d=sys.argv[1]
ty=collections.Counter(); ft=collections.Counter(); tt=collections.Counter(); k=collections.Counter(); exp=collections.Counter(); keys=collections.Counter(); qkeys=collections.Counter()
lens=[]
for f,tier in (('original.jsonl','standard'),('easy.jsonl','easy'),('hard.jsonl','hard')):
    for line in open(os.path.join(d,f),encoding='utf-8'):
        if not line.strip(): continue
        r=json.loads(line); t=r['question']['type']
        ty[t]+=1; ft[(r.get('family'),t)]+=1; tt[(tier,t)]+=1; k[(t,len(r['labels']))]+=1
        exp[(t,type(r.get('expected')).__name__)]+=1; keys.update(r.keys()); qkeys.update(r['question'].keys())
        lens.append(len(json.dumps(r['state'],ensure_ascii=False)))
print('types',dict(ty)); print('tier x type',sorted(tt.items()))
print('family x type'); [print(' ',a,b,n) for (a,b),n in sorted(ft.items(),key=lambda x:(x[0][0] or '',x[0][1]))]
print('labels K by type',sorted(k.items())); print('expected pytype',dict(exp)); print('record keys',dict(keys)); print('question keys',dict(qkeys))
lens.sort(); print('state chars min/med/max',lens[0],lens[len(lens)//2],lens[-1], 'n>40000chars',sum(l>40000 for l in lens))
