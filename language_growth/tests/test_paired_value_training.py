import copy
import unittest

from language_growth.development_data import comparison_swap_counterfactual, generate_development_tasks
from language_growth.paired_value_training import core, schedule_proof, select_batch


class PairedValueTests(unittest.TestCase):
    def test_same_question_pair_recomputes_relation_and_keeps_targets(self):
        example=generate_development_tasks(9101,3)[2]
        original=core.row(example,example["task_id"],"original")
        swapped=core.row(comparison_swap_counterfactual(example),example["task_id"],"swap")
        swapped["state"]["task_input"]=original["state"]["task_input"]
        self.assertEqual(original["state"]["task_input"],swapped["state"]["task_input"])
        a=core.gold_target(original["evaluation_only"]["program"])
        b=core.gold_target(swapped["evaluation_only"]["program"])
        self.assertEqual({k:v for k,v in a.items() if k!='relation'},{k:v for k,v in b.items() if k!='relation'})
        if a["relation"]=='equal': self.assertEqual(b["relation"],'equal')
        else: self.assertEqual({a["relation"],b["relation"]},{'greater_than','less_than'})
        visits=[0]
        first,members=select_batch([original],{original['id']:swapped},[0],visits)
        second,members2=select_batch([original],{original['id']:swapped},[0],visits)
        self.assertEqual(first,[original]);self.assertEqual(second,[swapped])
        self.assertEqual(members,['original']);self.assertEqual(members2,['swap'])

    def test_balanced_group_sampler_matches_baseline_draws(self):
        examples=generate_development_tasks(9101,600)
        rows=[core.row(e,e['task_id'],'original') for e in examples]
        swaps={r['id']:core.row(comparison_swap_counterfactual(e),e['task_id'],'swap')
               for r,e in zip(rows,examples) if core.gold_target(r['evaluation_only']['program'])['kind']=='compare'}
        before=copy.deepcopy(rows)
        proof=schedule_proof(rows,swaps,20261006)
        self.assertTrue(proof['base_world_id_draws_identical_to_baseline'])
        self.assertEqual(sum(proof['kind_counts_identical_to_baseline'].values()),9600)
        self.assertLessEqual(proof['max_per_world_member_count_difference'],1)
        self.assertEqual(rows,before)


if __name__=='__main__': unittest.main()
