"""Synthetic CPU checks of entry extraction, output safety and stop-before-step."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('diagnostic_entry', HERE/'diagnose.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class Guards(unittest.TestCase):
    def test_real_constructor_surface(self):
        dims,calls=entry.constructor_surface(HERE.parent/'code_xp04/train_cm.py')
        self.assertEqual(set(calls),{'model','model_mm','model_intra','model_inter'})
        self.assertEqual(calls['model'].func.id,'Model')
        self.assertEqual(calls['model_intra'].func.id,'Model_Intra')
        self.assertIn('feat2dim',ast.unparse(dims))
        self.assertNotIn('optimizer',ast.unparse(dims))

    def test_frozen_constructor_includes_scientific_arguments(self):
        _,calls=entry.constructor_surface(HERE.parent/'code_xp04/train_cm.py')
        for node in calls.values():
            names={k.arg for k in node.keywords}
            self.assertTrue({'dropout','num_L','num_K','norm','causal','att_type'}.issubset(names))

    def test_no_training_entry_called(self):
        tree=ast.parse((HERE/'diagnose.py').read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)
               and isinstance(n.func,ast.Attribute) and n.func.attr=='train_or_eval_graph_model']
        self.assertEqual(len(calls),1)
        kws={k.arg:k.value for k in calls[0].keywords}
        self.assertIs(ast.literal_eval(kws['train']),False)
        self.assertIs(ast.literal_eval(kws['optimizer']),None)

    def test_only_autograd_grad_not_step_or_backward_calls(self):
        tree=ast.parse((HERE/'diagnose.py').read_text())
        forbidden=[n for n in ast.walk(tree) if isinstance(n,ast.Call)
                   and isinstance(n.func,ast.Attribute) and n.func.attr in ('step','backward')]
        self.assertEqual(forbidden,[])
        grads=[n for n in ast.walk(tree) if isinstance(n,ast.Call)
               and isinstance(n.func,ast.Attribute) and n.func.attr=='grad']
        self.assertEqual(len(grads),1)

    def test_csv_refuses_schema_changes(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.csv';entry.append_csv(p,[{'x':1,'y':None}])
            entry.append_csv(p,[{'x':2,'y':3}])
            self.assertEqual(len(p.read_text().splitlines()),3)
            with self.assertRaises(AssertionError):entry.append_csv(p,[{'x':3,'z':1}])

    def test_nonfinite_is_missing_not_zero(self):
        self.assertIsNone(entry.scalar(float('nan')))
        self.assertIsNone(entry.scalar(float('inf')))
        self.assertEqual(entry.scalar(0),0.)

    def test_stop_before_optimizer_on_combiner_return(self):
        source='''def combine_training_losses(x):
    return x, {'loss':x}
def original_call(counter):
    total,components=combine_training_losses(7.)
    counter.append('backward')
    counter.append('optimizer.step')
'''
        scope={};exec(compile(source,'synthetic_original.py','exec'),scope)
        observed=[]
        def trace(frame,event,value):
            frame.f_trace_lines=False
            if frame.f_code.co_filename=='synthetic_original.py' and frame.f_code.co_name=='combine_training_losses' and event=='return':
                observed.append(value[0]);raise entry.CaseComplete()
            return trace
        counter=[];old=sys.gettrace()
        try:
            sys.settrace(trace)
            with self.assertRaises(entry.CaseComplete):scope['original_call'](counter)
        finally:sys.settrace(old)
        self.assertEqual(counter,[]);self.assertEqual(observed,[7.])

    def test_public_context_does_not_export_dialogue_keys_in_summary(self):
        text=(HERE/'diagnose.py').read_text()
        self.assertIn("'private' / item['checkpoint_id']",text)
        self.assertIn("'results/case_summary.csv'",text)
        tree=ast.parse(text)
        for n in ast.walk(tree):
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='append_csv':
                self.assertFalse(any(isinstance(x,ast.Constant) and x.value=='dialogue_ids' for x in ast.walk(n)))


if __name__=='__main__':
    unittest.main(verbosity=2)
