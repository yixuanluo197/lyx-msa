"""Synthetic scientific-contract checks; never load a real encoder/checkpoint."""
import copy
import importlib.util
from pathlib import Path
import unittest
import json
import tempfile
import csv
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch

SCRIPT=Path(__file__).with_name('run_score_predictions.py')


def load_module():
    spec=importlib.util.spec_from_file_location('score_predictions',SCRIPT)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


class ScoringContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not SCRIPT.exists():
            raise AssertionError('Missing scoring implementation: scientific contracts not implemented')
        cls.s=load_module()

    def fixture(self):
        data={'dataset':'MELD','seed':67137,'class_names':list('ABCDEFG'),'splits':{}}
        roles=[]
        for role,did,texts,uids in [('score_fit','12',['first','second'],['0','4']),('score_eval','19',['third'],['2'])]:
            d={'dialogue_id':did,'utterance_ids':uids,'texts':texts,'labels':torch.tensor([0]*len(uids)),
               'visual':torch.zeros(len(uids),3),'speaker':torch.tensor([[1.,0.]]*len(uids))}
            data['splits'][role]=[d]
            roles.extend({'sample_id':f'MELD:train:{did}:{uid}','dialogue_id':did,'utterance_id':uid,
                          'utterance_index':str(i),'role':role} for i,uid in enumerate(uids))
        return data,roles

    def test_cohort_preserves_noncontiguous_ids_and_all_clean_records(self):
        data,roles=self.fixture();index=self.s.validate_cohort(data,roles,expected_count=3)
        self.assertEqual(list(index),['MELD:train:12:0','MELD:train:12:4','MELD:train:19:2'])
        self.assertEqual(index['MELD:train:12:4']['utterance_index'],1)
        self.assertEqual(self.s.validate_reviews([],index),{})

    def test_rejects_role_leakage_and_test_split(self):
        data,roles=self.fixture();roles[0]['role']='base_fit'
        with self.assertRaisesRegex(ValueError,'role'):self.s.validate_cohort(data,roles,expected_count=3)
        data,roles=self.fixture();data['splits']['test']=[]
        with self.assertRaisesRegex(ValueError,'split'):self.s.validate_cohort(data,roles,expected_count=3)

    def review(self,index):
        sid=next(iter(index));t=index[sid]
        return {'sample_id':sid,'role':t['role'],'dialogue_id':t['dialogue_id'],
                'utterance_index':t['utterance_index'],'utterance_id':t['utterance_id'],
                'variant_id':'paraphrase_a','original_text_sha256':self.s.text_sha(t['original_text']),
                'processed_text':'rewritten','review_status':'PASS','review_method':'AI_CONTEXT_SEMANTIC_REVIEW',
                'reviewer':'independent AI reviewer','review_notes':'preserves the same proposition'}

    def test_review_identity_hash_duplicate_and_role_fail_closed(self):
        data,roles=self.fixture();index=self.s.validate_cohort(data,roles,expected_count=3);r=self.review(index)
        self.assertEqual(len(self.s.validate_reviews([r],index)),1)
        for change in [{'utterance_id':'4'},{'original_text_sha256':'bad'},{'role':'score_eval'}]:
            with self.assertRaises(ValueError):self.s.validate_reviews([{**r,**change}],index)
        with self.assertRaisesRegex(ValueError,'duplicate'):self.s.validate_reviews([r,r],index)

    def test_d_pred_uses_clean_argmax_signed_probability_loss_not_flip_rate(self):
        p=np.array([.6,.3,.02,.02,.02,.02,.02])
        # Both perturbed argmax classes flip; the original class probabilities are .2 and .4.
        variants={'paraphrase_a':{'eligible':True,'probabilities':[.2,.7,.02,.02,.02,.02,.02]},
                  'paraphrase_b':{'eligible':True,'probabilities':[.4,.5,.02,.02,.02,.02,.02]}}
        self.assertAlmostEqual(self.s.d_pred(p,variants),.3)
        variants['paraphrase_a']['probabilities']=[.8,.1,.02,.02,.02,.02,.02]
        variants['paraphrase_b']['probabilities']=[.8,.1,.02,.02,.02,.02,.02]
        self.assertAlmostEqual(self.s.d_pred(p,variants),-.2)

    def test_missing_or_single_ineligible_variant_returns_missing_not_zero(self):
        p=[.6,.3,.02,.02,.02,.02,.02]
        v={'paraphrase_a':{'eligible':True,'probabilities':p}}
        self.assertIsNone(self.s.d_pred(p,v))
        v['paraphrase_b']={'eligible':False,'probabilities':p}
        self.assertIsNone(self.s.d_pred(p,v))
        self.assertIsNone(self.s.d_pred(p,{}))

    def test_full_dialogue_single_target_overlay_preserves_every_other_tensor(self):
        clean=torch.randn(3,4,1024);audio=torch.randn(3,768);source=clean.clone();audio_before=audio.clone()
        target=torch.zeros(4,1024)
        overlay=self.s.replace_target(clean,1,target)
        self.assertTrue(torch.equal(clean,source));self.assertTrue(torch.equal(audio,audio_before))
        self.assertTrue(torch.equal(overlay[[0,2]],clean[[0,2]]));self.assertTrue(torch.equal(overlay[1],target))
        self.assertNotEqual(overlay.data_ptr(),clean.data_ptr())
        with self.assertRaises(ValueError):self.s.replace_target(clean,1,target.double())

    def test_token_gating_rejects_ascii_identical_duplicates_and_truncation(self):
        clean=[0,7,2]
        self.assertFalse(self.s.token_qc(clean,[0,7,2],False,False)['eligible'])
        self.assertFalse(self.s.token_qc(clean,[0,8,2],True,False)['eligible'])
        self.assertFalse(self.s.token_qc(clean,[0,8,2],False,True)['eligible'])
        self.assertTrue(self.s.token_qc(clean,[0,8,2],False,False)['eligible'])
        v={k:{'eligible':True,'actual_token_ids':[0,8,2]} for k in ('paraphrase_a','paraphrase_b')}
        self.s.enforce_distinct_versions(v)
        self.assertTrue(all(not row['eligible'] for row in v.values()))

    def test_probabilities_are_not_renormalized_or_invalid_values_silently_fixed(self):
        for p in ([1.,0.,0.], [.2]*7, [float('nan')]+[0.]*6):
            with self.assertRaises(ValueError):self.s.validate_probability(p)
        p=np.array([.6,.3,.02,.02,.02,.02,.02]);summary=self.s.clean_summary(p,1)
        self.assertEqual(summary['original_correct'],0)
        self.assertAlmostEqual(summary['clean_margin'],.3)
        self.assertAlmostEqual(summary['clean_entropy'],-float(np.sum(p*np.log(p))))
        self.assertAlmostEqual(summary['original_confidence'],.6)


    def test_encode_keeps_every_clean_row_and_records_review_and_extraction_failures(self):
        data,roles=self.fixture();index=self.s.validate_cohort(data,roles,expected_count=3)
        first=self.review(index);second={**first,'variant_id':'paraphrase_b','processed_text':'changed second'}
        class ToyText:
            @staticmethod
            def tokenize(tokenizer,texts,dataset):
                values=[[0]+list(t.encode('ascii',errors='ignore'))+[2] for t in texts]
                return [v[:511]+[2] if len(v)>512 else v for v in values], [dict(truncated=len(v)>512,empty_after_processing=len(v)==2) for v in values]
            @staticmethod
            def extract_texts(model,tokenizer,texts,dataset,batch_size):
                if texts==['changed second']:raise RuntimeError('synthetic extraction failure')
                return torch.stack([torch.full((4,1024),float(len(t))) for t in texts])
        reviewed=self.s.validate_reviews([first,second],index)
        clean,records,tokens=self.s.make_encoded_records(index,reviewed,ToyText,None,None)
        self.assertEqual(sum(len(t) for role in clean.values() for t in role.values()),3)
        self.assertEqual(len(records),3)
        target=records[first['sample_id']]
        self.assertEqual(target['paraphrase_b']['encoding_status'],'FAILED')
        self.assertIsNone(self.s.d_pred([.6,.3,.02,.02,.02,.02,.02],target))
        self.assertEqual(records['MELD:train:19:2']['paraphrase_a']['encoding_status'],'NOT_ATTEMPTED')
        # MELD ASCII removal erases the only raw-string difference.
        first['processed_text']='first' + chr(0x85)
        reviewed=self.s.validate_reviews([first],index)
        _,records,_=self.s.make_encoded_records(index,reviewed,ToyText,None,None)
        self.assertFalse(records[first['sample_id']]['paraphrase_a']['eligible'])

    def test_forward_detects_context_mutation_and_never_uses_score_labels(self):
        data,roles=self.fixture();dialogue=data['splits']['score_fit'][0]
        text=torch.zeros(2,4,1024);audio=torch.zeros(2,768)
        class FakeAdapter:
            @staticmethod
            def infer_dialogue(model,d,t,a):
                return np.tile([.6,.3,.02,.02,.02,.02,.02],(2,1))
        before=copy.deepcopy(dialogue)
        out=self.s.locked_forward(FakeAdapter,None,dialogue,text,audio)
        self.assertEqual(out.shape,(2,7));self.assertTrue(torch.equal(before['labels'],dialogue['labels']))
        class MutatingAdapter:
            @staticmethod
            def infer_dialogue(model,d,t,a):
                d['speaker'][0,0]=0
                return np.tile([.6,.3,.02,.02,.02,.02,.02],(2,1))
        with self.assertRaisesRegex(RuntimeError,'mutated'):
            self.s.locked_forward(MutatingAdapter,None,dialogue,text,audio)

    def test_old_planb_checkpoint_cannot_masquerade_as_fresh_branch(self):
        data,roles=self.fixture();index=self.s.validate_cohort(data,roles,expected_count=3)
        config={'Dataset':'MELD','seed':67137,'text_dim':1024,'audio_dim':768,
                'text_layers':4,'visual_dim':3,'speaker_dim':2}
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);feature=root/'local_only/base_data/features'
            feature.mkdir(parents=True)
            (feature/'text_features.manifest.json').write_text(json.dumps({'model_sha256':'new'}))
            payload={'new_features_only':True,'stage':'cmerc','effective_config':config}
            with self.assertRaisesRegex(ValueError,'branch'):
                self.s.verify_cmerc_metadata(payload,data,index,{'model_sha256':'new'},root,root)

    def test_synthetic_two_stage_pipeline_keeps_uncovered_rows_and_full_probabilities(self):
        data,roles=self.fixture();index=self.s.validate_cohort(data,roles,expected_count=3)
        first=self.review(index)
        second={**first,'variant_id':'paraphrase_b','processed_text':'another rewrite'}
        reviewed=self.s.validate_reviews([first,second],index)
        audio={role:{str(d['dialogue_id']):torch.zeros(len(d['texts']),768) for d in data['splits'][role]} for role in data['splits']}
        class ToyText:
            @staticmethod
            def load_encoder(path,device):return torch.nn.Linear(1,1),None
            @staticmethod
            def tokenize(tokenizer,texts,dataset):
                return [[0]+list(t.encode())+[2] for t in texts],[dict(truncated=False,empty_after_processing=False) for t in texts]
            @staticmethod
            def extract_texts(model,tokenizer,texts,dataset,batch_size):
                return torch.stack([torch.full((4,1024),float(len(t))) for t in texts])
        class ToyAdapter:
            @staticmethod
            def load_model(root,path,device):return torch.nn.Linear(1,1)
            @staticmethod
            def infer_dialogue(model,d,text,audio):
                return np.array([[.5+float(t[0,0])*.001,.4-float(t[0,0])*.001,.02,.02,.02,.02,.02] for t in text],dtype=np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);checkpoint=root/'fake_checkpoint.pt';torch.save({},checkpoint)
            args=SimpleNamespace(run_dir=root,adapter_dir=root,features=root/'score_text.pt',device='cpu',threads=1)
            common=(data,index,reviewed,audio,{},dict(model_sha256='synthetic'),checkpoint)
            with patch.object(self.s,'require_runtime',return_value={'environment':'synthetic_cpu'}), \
                 patch.object(self.s,'common',return_value=common),patch.object(self.s,'seed_runtime'), \
                 patch.object(self.s,'import_parent',return_value=ToyText):
                self.s.encode(args)
            with patch.object(self.s,'require_runtime',return_value={'environment':'synthetic_cpu'}), \
                 patch.object(self.s,'common',return_value=common),patch.object(self.s,'seed_runtime'), \
                 patch.object(self.s,'verify_cmerc_metadata',return_value=({},{})), \
                 patch.object(self.s,'import_parent',return_value=ToyAdapter):
                self.s.infer(args)
            with (root/'results/score_inputs.csv').open() as stream:clean=list(csv.DictReader(stream))
            with (root/'results/score_transformations.csv').open() as stream:risks=list(csv.DictReader(stream))
            self.assertEqual(len(clean),3);self.assertEqual(len(risks),6)
            self.assertEqual(sum(r['D_pred']!='' for r in clean),1)
            self.assertEqual([r['utterance_id'] for r in clean],['0','4','2'])
            self.assertTrue(all(len(json.loads(r['original_probabilities']))==7 for r in clean))
            self.assertTrue(all(len(json.loads(r['perturbed_probabilities']))==7 for r in risks if r['perturbed_probabilities']))
            verification=json.loads((root/'results/score_prediction_verification.json').read_text())
            self.assertEqual(verification['model_state_before'],verification['model_state_after'])


if __name__=='__main__':unittest.main()
