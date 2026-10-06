import json
from language_growth.development_data import generate_development_tasks, development_manifest
from language_growth.tests.test_free_running_dev import ScriptedProvider,run
for split in ('development_train','development_validation'):
 examples=generate_development_tasks(17,30,choice_format='production',split=split)
 print(json.dumps({'manifest':development_manifest(examples),'report':run(examples,ScriptedProvider())},sort_keys=True))
