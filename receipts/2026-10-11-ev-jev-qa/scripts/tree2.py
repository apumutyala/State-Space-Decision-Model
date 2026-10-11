import json,sys,re
j=json.load(open(sys.argv[1],encoding='utf-8'))
for e in j['tree']:
    p=e['path']
    if e['type']=='blob' and (p.endswith('.jsonl') or re.search(r'(router|judge|v1\.6|v16|public|sealed|draw|P300)',p,re.I)): print(p,e.get('size'))
