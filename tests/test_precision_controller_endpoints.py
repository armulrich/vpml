"""Controller bookkeeping only; importing does not launch remote work."""
import copy
import unittest
from scripts.modal_precision_e5 import stopping_updates, evaluation_configuration, validate_continuation


class EndpointTests(unittest.TestCase):
    def setUp(self):
        self.p={'stop_epoch':5,'updates_per_epoch':592,
                'autonomous_gate_updates':[0,3,50,150,300,592,1184,1776,2368,2960]}

    def test_prefix_and_epochs_use_identical_gate_prefix(self):
        self.assertEqual(stopping_updates(self.p,5,50),(50,[0,3,50]))
        self.assertEqual(stopping_updates(self.p,1),(592,[0,3,50,150,300,592]))
        self.assertEqual(stopping_updates(self.p,5)[0],2960)
        for epoch,update in [(0,0),(6,0),(1,600),(5,51),(5,-1)]:
            with self.assertRaises(ValueError):stopping_updates(self.p,epoch,update)

    def test_only_learning_rates_are_ignored_for_e0_reuse(self):
        a={'learning_rate':.001,'final_learning_rate':.00001,'horizon':600,'seed':1729}
        self.assertEqual(evaluation_configuration(a),evaluation_configuration({**a,'learning_rate':.0001}))
        self.assertNotEqual(evaluation_configuration(a),evaluation_configuration({**a,'horizon':400}))
        self.assertNotEqual(evaluation_configuration(a),evaluation_configuration({**a,'seed':1}))

    def continuation(self):
        parent={**self.p,'config':{'planned_epochs':1000,'learning_rate':.0001},
            'source_sha256':'source','manifest_sha256':'data','normalization_sha256':'norm',
            'validation_configurations':{'full15':{'horizon':600}},
            'stop_on_nonfinite':True,'combined_cost_cap':350}
        child=copy.deepcopy(parent)
        child.update(stop_epoch=10,resume_from={'run':'parent_e5','accepted_updates':2960})
        child['autonomous_gate_updates'] += [592*e for e in range(6,11)]
        return parent,child

    def test_continuation_retains_schedule_and_epoch_gates(self):
        parent,child=self.continuation()
        self.assertEqual(validate_continuation(parent,child),2960)
        self.assertEqual(stopping_updates(child,10)[0],5920)
        self.assertEqual(parent['stop_epoch'],5)

    def test_continuation_rejects_scientific_or_budget_changes(self):
        for key,value in [('config',{'planned_epochs':1000,'learning_rate':.001}),
                ('source_sha256','other'),('manifest_sha256','other'),
                ('normalization_sha256','other'),('validation_configurations',{}),
                ('stop_on_nonfinite',False),('combined_cost_cap',1000)]:
            parent,child=self.continuation();child[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                validate_continuation(parent,child)

    def test_continuation_rejects_wrong_cursor_missing_gate_or_unsafe_parent(self):
        for mutation in ('cursor','gate','path','endpoint'):
            parent,child=self.continuation()
            if mutation=='cursor':child['resume_from']['accepted_updates']=2950
            if mutation=='gate':child['autonomous_gate_updates'].remove(3552)
            if mutation=='path':child['resume_from']['run']='../parent_e5'
            if mutation=='endpoint':child['stop_epoch']=5
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                validate_continuation(parent,child)


if __name__=='__main__':unittest.main()
