import json,sys,math
S=sys.argv[1]
f={'clef':S+'/clef-subset-score.json','stock':S+'/5cmat/s/out/index/score_stock.json','conv':S+'/5cmat/s/out/index/score_conv.json','rwkv':S+'/rwkv7-art2/x/phases/final/out/index/score.json'}
d={k:json.load(open(v))['slice_index'] for k,v in f.items()}
areas={'knowledge':(25.85,{25:1.2,30:1,31:1,32:1,33:1,43:1,44:1,45:1.2,57:1.2,58:1.2}),
'language':(25.85,{11:1,12:1.2,28:1.2,29:1.2,38:1,39:1,40:1,41:1,42:1,59:1}),
'retrieval':(20.02,{4:1.2,5:1.2,36:1.2,37:1,56:1,61:1}),
'tools':(18.28,{1:1.2,2:1,3:1.2,9:1,62:1}),
'arts':(10.0,{20:1,21:1,22:1,23:1,48:1.2,50:1,64:1})}
names={25:'GPQA Diamond*',30:'GSM8K',31:'ChessBench',32:'MuSR',33:'SATA-Bench',43:'CRUXEval',44:'CLadder',45:'HLE*',57:'MMLU-Pro*',58:'BBH*',11:'ContractNLI',12:'ANLI*',28:'WinoGrande*',29:'HellaSwag*',38:'ACOS',39:'FinEntity',40:'iSarcasmEval',41:'VAST',42:'NLI4CT',59:'RAGTruth',4:'BANKING77*',5:'CLINC150*',36:'BRIGHT*',37:'Amazon ESCI',56:'PhishNChips',61:'HoVer',1:'BFCL*',2:'ToolRet',3:'API-Bank*',9:'Home appliances',62:'When2Call',20:'BPoMP',21:'Humicroedit',22:'POP909',23:'cfcolor',48:'ForecastBench*',50:'Habermas',64:'New Yorker'}
rows=[];tot=0
for a,(w,bs) in areas.items():
  sw=sum(bs.values())
  for b,g in bs.items():
    p={k:d[k]['benchmarks'][str(b)]['point'] for k in d}
    se={k:d[k]['benchmarks'][str(b)]['se'] for k in d}
    iw=w*g/sw  # index points per 100 skill points... index contribution = iw*skill/100
    gap=p['clef']-p['stock']
    contrib=iw*gap/100
    tot+=contrib
    rows.append((a,b,names[b],iw,p['clef'],se['clef'],p['stock'],se['stock'],p['conv'],p['rwkv'],gap,contrib))
rows.sort(key=lambda r:-r[11])
print('| # | Benchmark (id) | Area | Index wt (pts per 100 skill) | Clef-flash | Q9 stock | Q9 1-GDN | RWKV7-13.3B | Gap CF-Q9 (skill) | Gap in Index pts | cum % of gap |')
print('|---|---|---|---|---|---|---|---|---|---|---|')
c=0
for i,r in enumerate(rows,1):
  c+=r[11]
  print(f'| {i} | {r[2]} ({r[1]}) | {r[0]} | {r[3]:.2f} | {r[4]:.1f} ±{r[5]:.1f} | {r[6]:.1f} ±{r[7]:.1f} | {r[8]:.1f} | {r[9]:.1f} | {r[10]:+.1f} | {r[11]:+.2f} | {100*c/tot:.0f}% |')
print('total gap index pts',round(tot,2), 'clef',round(d['clef']['index']['point'],2),'stock',round(d['stock']['index']['point'],2),'conv',round(d['conv']['index']['point'],2),'rwkv',round(d['rwkv']['index']['point'],2))
for a in areas: print(a,{k:round(d[k]['areas'][a]['point'],1) for k in d}, 'area gap idx pts', round(areas[a][0]*(d['clef']['areas'][a]['point']-d['stock']['areas'][a]['point'])/100,2))
full=json.load(open(S+'/clef-art1/score.json'))['slice_index']['benchmarks']
pos=sum(r[11] for r in rows if r[11]>0); neg=sum(r[11] for r in rows if r[11]<0)
print('pos',round(pos,2),'neg',round(neg,2))
for r in rows: print(r[1], round(full[str(r[1])]['point'],1))
