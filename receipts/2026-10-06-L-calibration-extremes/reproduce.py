import json
from decide.calibration import softmax,FeatureTemperature,CalibrationNotServable
rows=[]
for logits,t in [([1e308,0.0],.05),([-1e308,-1e308],.05),([1e308,-1e308],1e308)]:
 rows.append({'logits':logits,'temperature':t,'probabilities':softmax(logits,t)})
base=FeatureTemperature(t_min=.07)
manual=FeatureTemperature(t_min=.07,w=(base.w[0]+5e-13,0,0))
rejected=False
try:manual.assert_servable()
except CalibrationNotServable:rejected=True
print(json.dumps({'softmax_cases':rows,'canonical_temperature':base.temperature_for(2,{'evidence_tokens':10}),'manual_status':manual.identity()['status'],'manual_rejected':rejected},sort_keys=True,allow_nan=False))
