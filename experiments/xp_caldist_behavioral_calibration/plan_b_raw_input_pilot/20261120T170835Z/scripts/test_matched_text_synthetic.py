"""Bounded synthetic integrity checks; never loads real data or model weights."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch

SPEC = importlib.util.spec_from_file_location('matched_engine',Path(__file__).with_name('run_matched_text.py'))
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)

class Integrity(unittest.TestCase):
    def fixture(self):
        original = 'No!\u0085 Is that clear?\u2028Yes.'
        sample = dict(sample_id='S',dialogue_id='D',utterance_index='0')
        dialogue = {'D':{'texts':[original]}}
        rows = [dict(sample_id='S',dialogue_id='D',utterance_index=0,perturbation_type=kind,
                     variant_id=version,original_text=original,original_text_sha256=m.text_sha(original),
                     processed_text=original+' Page two.',review_status='UNCERTAIN')
                for kind,versions in m.FAMILIES.items() for version in versions]
        return sample,dialogue,rows

    def test_unicode_jsonl_physical_lines(self):
        _,_,rows = self.fixture()
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'unicode.jsonl'
            p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
            self.assertEqual(m.read_jsonl(p),rows)

    def test_complete_conditions_including_uncertain(self):
        sample,d,rows=self.fixture()
        self.assertEqual(set(m.validate_reviews([sample],rows,d)),{'S'})
        with self.assertRaises(ValueError):m.validate_reviews([sample],rows[:-1],d)
        with self.assertRaises(ValueError):m.validate_reviews([sample],rows+[rows[0]],d)

    def test_id_and_original_binding(self):
        sample,d,rows=self.fixture()
        rows[0]['original_text_sha256']='bad'
        with self.assertRaises(ValueError):m.validate_reviews([sample],rows,d)
        rows[0]['original_text_sha256']=m.text_sha(rows[0]['original_text'])
        rows[0]['utterance_index']=1
        with self.assertRaises(ValueError):m.validate_reviews([sample],rows,d)

    def test_model_input_file_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'state';p.write_text('original')
            hashes={str(p):m.shared.sha256(p)}
            m.check_bindings(hashes)
            p.write_text('mutation')
            with self.assertRaises(RuntimeError):m.check_bindings(hashes)

    def test_finite_fp32_feature_hash_and_tokens(self):
        feature=torch.randn(4,1024)
        record=m.feature_record({},feature,[0,5,2],torch.zeros_like(feature))
        m.check_cached_record(record)
        record['feature'][0,0]+=1
        with self.assertRaises(ValueError):m.check_cached_record(record)
        record=m.feature_record({},feature,[0,5,2],torch.zeros_like(feature))
        record['actual_token_ids']=[0,8,2]
        with self.assertRaises(ValueError):m.check_cached_record(record)
        with self.assertRaises(ValueError):m.feature_record({},feature.double(),[0,5,2],feature)
        with self.assertRaises(ValueError):m.feature_record({},torch.full_like(feature,float('nan')),[0,5,2],feature)

    def test_fail_qc_preserved_but_schema_strict(self):
        row=dict(status='FAILED_QC',actual_token_ids=[0,2],actual_token_ids_sha256=m.json_hash([0,2]))
        m.check_cached_record(row)
        row['status']='MAGIC'
        with self.assertRaises(ValueError):m.check_cached_record(row)

    def test_non_target_modality_and_rows_exact(self):
        text=torch.zeros(3,4,1024);audio=torch.zeros(3,768)
        changed=text.clone();changed[1]=1
        m.shared.assert_target_only(text,audio,changed,audio,1,'text')
        changed[2]=1
        with self.assertRaises(AssertionError):m.shared.assert_target_only(text,audio,changed,audio,1,'text')

    def test_bn_and_weight_mutation_fingerprint(self):
        model=torch.nn.Sequential(torch.nn.Linear(4,4),torch.nn.BatchNorm1d(4)).eval().requires_grad_(False)
        before=m.fingerprint(model)
        model(torch.randn(3,4))
        self.assertEqual(before,m.fingerprint(model))
        model[1].running_mean[0]+=1
        self.assertNotEqual(before,m.fingerprint(model))

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'test.json';m.save_json(p,{'valid':True})
            with self.assertRaises(FileExistsError):m.save_json(p,{'valid':False})
            t=Path(td)/'features.pt';m.save_torch(t,{'ok':True})
            with self.assertRaises(FileExistsError):m.save_torch(t,{'ok':False})

if __name__=='__main__':unittest.main(verbosity=2)
