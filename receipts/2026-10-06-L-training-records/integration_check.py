import hashlib,json
from language_growth.development_data import generate_development_tasks,development_manifest
from language_growth.training_records import export_training_records
from train.data import items_from_records
from train.guard import SplitGuard
from heads import ByteEncoder
for split in ('development_train','development_validation'):
 examples=generate_development_tasks(17,6,choice_format='production',split=split)
 records=export_training_records(examples,split=split)
 items=items_from_records(records,ByteEncoder(),guard=SplitGuard.synthetic_only('new generated language development worlds'))
 for row,item in zip(records,items):
  qid,gold=next(iter(row['labels'].items()))
  assert item.row.option_ids[item.label]==gold
  assert item.id==row['id']+'/'+qid
 print(json.dumps({'split':split,'manifest':development_manifest(examples),'records':records,'encoded':[{'id':i.id,'label':i.label,'option_ids':i.row.option_ids,'token_ids_sha256':hashlib.sha256(json.dumps(i.row.ids).encode()).hexdigest()} for i in items]},sort_keys=True))
