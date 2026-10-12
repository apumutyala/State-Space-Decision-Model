import json,hashlib,sys,os
d=sys.argv[1]
for f in sorted(os.listdir(d)):
    b=open(os.path.join(d,f),'rb').read(); print(f,len(b),hashlib.sha256(b).hexdigest())
for f in ('h2o_api.json','qwen_api.json'):
    j=json.load(open(os.path.join(d,f),encoding='utf-8'))
    print(f, j.get('id'), j.get('sha'), j.get('lastModified'), [t for t in j.get('tags',[]) if 'licen' in t or 'base_model' in t], json.dumps(j.get('cardData')), j.get('gated'))
l=open(os.path.join(d,'h2o_LICENSE'),encoding='utf-8').read()
print(l[:200].replace('\n',' | ')); print('...', l[-400:].replace('\n',' | '))
