"""Controller bookkeeping only; importing does not launch remote work."""
import unittest
from scripts.modal_precision_e5 import stopping_updates, evaluation_configuration


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


if __name__=='__main__':unittest.main()
