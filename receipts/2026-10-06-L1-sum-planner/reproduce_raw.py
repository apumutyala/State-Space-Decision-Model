from copy import deepcopy
import json
from language_growth.tests.test_sum_planner import SumPlannerTests,ScriptedProvider,SUM,run
class Recorded(ScriptedProvider):
 def __init__(self,selectors):
  super().__init__(selectors);self.responses=[]
 def decide(self,request):
  response=super().decide(request);self.responses.append(deepcopy(response));return response
for name,selectors in [('valid_sum',[SUM,'Fact atlas-v1 field price','Fact boreal-v1 field price']),('unit_mismatch',[SUM,'Fact atlas-v1 field price','Fact boreal-eur-v1 field price']),('legacy_read',['Assert one selected fact','Fact atlas-v1 field price'])]:
 world=SumPlannerTests();world.setUp();provider=Recorded(selectors)
 result=run(world.host,world.catalog,provider)
 print(json.dumps({'case':name,'selectors':selectors,'requests':provider.requests,'responses':provider.responses,'status':result.status,'reason':result.reason,'program':result.program,'text':result.text,'provider_calls':result.provider_calls,'forward_calls':result.forward_calls,'trace':result.trace},sort_keys=True))
