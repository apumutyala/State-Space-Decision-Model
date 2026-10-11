import json,sys,re
for f in sys.argv[1:]:
    j=json.load(open(f,encoding='utf-8')); print('=====',f, list(j)[:30] if isinstance(j,dict) else type(j))
    def w(o,p):
        if isinstance(o,dict):
            for k,v in o.items():
                if re.search(r'type|score|ordinal|famil',str(k),re.I) and not isinstance(v,(dict,list)) or (isinstance(k,str) and k in('score','choice','noul')): print('  ',p+'.'+str(k),'=',str(v)[:200])
                elif re.search(r'type|qtype|by_type|famil',str(k),re.I): print('  ',p+'.'+str(k),'=',json.dumps(v)[:600])
                w(v,p+'.'+str(k))
        elif isinstance(o,list):
            for i,x in enumerate(o[:50]): w(x,p+f'[{i}]')
    w(j,'$')
