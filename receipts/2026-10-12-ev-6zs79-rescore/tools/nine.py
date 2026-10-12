import sys,re,html
t=open(sys.argv[1],encoding='utf-8',errors='replace').read()
for pat in ('Nine A4','LONGITEM-TOKENS','77,857'):
    for m in list(re.finditer(re.escape(pat),t))[:2]:
        s=re.sub(r'<[^>]+>','',t[max(0,m.start()-900):m.end()+300]); print('---',pat,'::',html.unescape(re.sub(r'\s+',' ',s))[-1100:])
