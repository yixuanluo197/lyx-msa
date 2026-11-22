#!/usr/bin/env python3
"""CPU synthetic checks only; no real utterance, probability, feature or weights."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

PATH=Path(__file__).with_name('infer_corruption.py')
spec=importlib.util.spec_from_file_location('corruption_engine_test',PATH)
engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
OPTIONS=dict(seed=20261122,allowlist=['about','waiting','outside','could','there'])


class TextStub:
    @staticmethod
    def preprocessing(text,dataset):return text.encode('ascii',errors='ignore').decode('ascii')
    @staticmethod
    def tokenize(tokenizer,texts,dataset):
        ids=[(tokenizer or {}).get(text,[0]+[ord(c) for c in TextStub.preprocessing(text,dataset)]+[2]) for text in texts]
        return ids,[dict(truncated=False,empty_after_processing=not text) for text in texts]
    @staticmethod
    def extract_texts(model,tokenizer,texts,dataset,batch_size):
        return torch.stack([torch.full((4,1024),int(engine.text_sha(t)[:8],16)/2**32,dtype=torch.float32) for t in texts])


def fixture():
    index={};roster=[];cache={'clean':{'score_fit':{},'score_eval':{}},'clean_token_ids':{}}
    for role,did,sid,original in [('score_fit','synthetic_d1','synthetic_s1','We are waiting outside.'),
                                   ('score_eval','synthetic_d2','synthetic_s2','Hi.')]:
        index[sid]=dict(sample_id=sid,role=role,dialogue_id=did,utterance_id='0',utterance_index=0,original_text=original)
        roster.append(dict(blind_id='blind_'+sid,sample_id=sid,role=role,dialogue_id=did,utterance_id='0',utterance_index='0'))
        cache['clean'][role][did]=TextStub.extract_texts(None,None,[original,'Fixed synthetic context.'],'MELD',1)
        cache['clean_token_ids'][sid]=TextStub.tokenize(None,[original],'MELD')[0][0]
    roster=engine.validate_roster(roster,index,expected=None)
    rows=[]
    for sid,target in index.items():
        offered=engine.ranked_candidates(target['original_text'],roster[sid]['blind_id'],OPTIONS)
        for i,v in enumerate(engine.VARIANTS):
            text,edit=offered[i] if i<len(offered) else (target['original_text'],None)
            rows.append({k:target[k] for k in ('sample_id','role','dialogue_id','utterance_id','utterance_index')})
            rows[-1].update(variant_id=v,original_text_sha256=engine.text_sha(target['original_text']),
                processed_text=text,processed_text_sha256=engine.text_sha(text),edit=edit,
                generation_status='GENERATED' if edit else 'NO_ELIGIBLE_EDIT',
                review_status='PASS' if edit else 'FAIL',review_method='AI_CONTEXT_ORTHOGRAPHIC_REVIEW' if edit else 'NO_CANDIDATE',
                reviewer='Synthetic reviewer',review_notes='Synthetic fixture only.')
    reviewed=engine.validate_reviews(rows,index,roster,OPTIONS)
    qc=lambda a,b,t,e:dict(token_ids_changed=a!=b,no_truncation=not t,nonempty=not e,eligible=bool(a!=b and not t and not e))
    state=SimpleNamespace(index=index,roster=roster,cache=cache,reviews=reviewed,
        engine=SimpleNamespace(token_qc=qc),inputs={'synthetic':'hash'},
        data={'class_names':[str(i) for i in range(7)]},text_lock={'model_sha256':'synthetic_model'})
    return state,rows


def encoded_fixture(state):
    records=engine.encode_records(state,TextStub,None,None)
    runtime=dict(environment='TEXT_repro',torch='2.2.0+cu118',transformers='4.42.3')
    bundle=dict(format=engine.FORMAT,inputs=state.inputs,protocol_sha256='synthetic_protocol',
        class_names=state.data['class_names'],fp32=True,text_model_sha256='synthetic_model',
        encoder_state_before='unchanged',encoder_state_after='unchanged',runtime=runtime,variants=records)
    manifest=dict(format=engine.FORMAT,source_bindings=state.inputs,protocol_sha256='synthetic_protocol',
        encoder_state_before='unchanged',encoder_state_after='unchanged',runtime=runtime)
    return bundle,manifest


class CorruptionSynthetic(unittest.TestCase):
    def test_fixed_rank_single_swap_codepoints_boundaries(self):
        original='é about';selected=engine.ranked_candidates(original,'blind_fixture',OPTIONS)
        self.assertEqual(len(selected),2)
        for changed,edit in selected:
            pos=edit['position'];self.assertEqual(edit['before'],original[pos:pos+2]);self.assertEqual(edit['after'],edit['before'][::-1])
            self.assertEqual(changed,original[:pos]+edit['after']+original[pos+2:])
            self.assertEqual(edit['original_word'],'about');self.assertEqual(edit['word_start'],2)
            self.assertEqual(edit['edited_word'][0],'a');self.assertEqual(edit['edited_word'][-1],'t')
        self.assertEqual(selected,engine.ranked_candidates(original,'blind_fixture',OPTIONS))
        for text in ["couldn't",'could\u0092nt','there\u2019s','about-more','about—more','About','ABOUT']:
            self.assertEqual(engine.ranked_candidates(text,'blind_fixture',OPTIONS),[],msg=text)

    def test_exact_review_rank_hash_identity_guard(self):
        state,rows=fixture()
        for field,value in [('processed_text','arbitrary replacement'),('edit',None),('role','score_eval'),('review_status','PENDING')]:
            changed=copy.deepcopy(rows);changed[0][field]=value
            with self.assertRaises(ValueError):engine.validate_reviews(changed,state.index,state.roster,OPTIONS)
        changed=copy.deepcopy(rows);changed[0]['edit']['position']+=1
        with self.assertRaises(ValueError):engine.validate_reviews(changed,state.index,state.roster,OPTIONS)
        with self.assertRaises(ValueError):engine.validate_reviews(rows[:-1],state.index,state.roster,OPTIONS)

    def test_no_candidate_rows_remain_exact_failed(self):
        state,rows=fixture();r=next(x for x in rows if x['generation_status']=='NO_ELIGIBLE_EDIT')
        self.assertEqual(r['processed_text'],state.index[r['sample_id']]['original_text'])
        altered=copy.deepcopy(rows)
        for row in altered:
            if row['edit'] is None:row['review_status']='PASS'
        with self.assertRaises(ValueError):engine.validate_reviews(altered,state.index,state.roster,OPTIONS)

    def test_roster_never_changes_targets_or_role(self):
        state,_=fixture();rows=list(state.roster.values())
        with self.assertRaises(ValueError):engine.validate_roster(rows+rows[:1],state.index,expected=None)
        rows=copy.deepcopy(rows);rows[0]['role']='base_fit'
        with self.assertRaises(ValueError):engine.validate_roster(rows,state.index,expected=None)

    def test_physical_jsonl_preserves_unicode_separators(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'synthetic.jsonl';rows=[{'text':'a\u0085b\u2028c\u2029d'},{'text':'next'}]
            path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
            self.assertEqual(engine.read_jsonl(path),rows)

    def test_token_difference_and_layerwise_features(self):
        diff=engine.token_difference([0,10,11,2],[0,10,22,23,2])
        self.assertTrue(diff['actual_ids_changed']);self.assertEqual(diff['changes'][0]['clean_token_ids'],[11])
        result=engine.feature_difference(torch.zeros(4,1024),torch.full((4,1024),2.))
        self.assertEqual(len(result['layers']),4)
        self.assertEqual(result['overall']['max_abs'],2.)
        self.assertEqual(result['overall']['mean_abs'],2.)
        self.assertEqual(result['overall']['rmse'],2.)
        self.assertIsNone(result['overall']['cosine'])

    def test_encode_retains_all_and_only_pass_can_extract(self):
        state,_=fixture();sid='synthetic_s1';state.reviews[sid]['corruption_b']['review_status']='UNCERTAIN'
        before={r:engine.tensor_sha(x) for r,values in state.cache['clean'].items() for x in values.values()}
        encoded=engine.encode_records(state,TextStub,None,None)
        self.assertEqual(sum(map(len,encoded.values())),4)
        self.assertTrue(encoded[sid]['corruption_a']['eligible'])
        self.assertFalse(encoded[sid]['corruption_b']['eligible'])
        self.assertNotIn('feature',encoded[sid]['corruption_b'])
        for record in encoded['synthetic_s2'].values():
            self.assertFalse(record['eligible']);self.assertEqual(record['reason'],'NO_ELIGIBLE_EDIT')
        after={r:engine.tensor_sha(x) for r,values in state.cache['clean'].items() for x in values.values()}
        self.assertEqual(before,after)

    def test_duplicate_actual_tokens_disable_both_not_reselect(self):
        state,_=fixture();reviews=state.reviews['synthetic_s1']
        tokenizer={r['processed_text']:[0,99,2] for r in reviews.values()}
        result=engine.encode_records(state,TextStub,tokenizer,None)['synthetic_s1']
        for record in result.values():
            self.assertFalse(record['eligible']);self.assertEqual(record['reason'],'DUPLICATE_K2_ACTUAL_TOKEN_IDS')

    def test_encoded_consumer_rejects_runtime_state_feature_tamper(self):
        state,_=fixture();bundle,manifest=encoded_fixture(state)
        engine.validate_encoded_records(bundle,manifest,state,'synthetic_protocol')
        tampered=copy.deepcopy(bundle);tampered['runtime']['transformers']='different'
        with self.assertRaises(ValueError):engine.validate_encoded_records(tampered,manifest,state,'synthetic_protocol')
        tampered=copy.deepcopy(bundle);tampered['encoder_state_after']='changed'
        with self.assertRaises(ValueError):engine.validate_encoded_records(tampered,manifest,state,'synthetic_protocol')
        tampered=copy.deepcopy(bundle);tampered['variants']['synthetic_s1']['corruption_a']['feature'][0,0]+=1
        with self.assertRaises(ValueError):engine.validate_encoded_records(tampered,manifest,state,'synthetic_protocol')

    def test_all_clean_rows_and_semantic_cells_exact_and_missing_blank(self):
        source=[dict(sample_id='a',D_pred='0.123456789',original_probabilities='[0.4, 0.1]'),dict(sample_id='b',D_pred=''),dict(sample_id='c',D_pred='0')]
        result=engine.merge_score_inputs(source,{'a':dict(corruption_d_pred=.2,corruption_d_abs=.3,corruption_cancellation=.1),'b':None},{'b':'NO_EDIT'})
        for old,new in zip(source,result):self.assertEqual({k:new[k] for k in old},old)
        self.assertEqual(result[1]['corruption_d_pred'],'');self.assertEqual(result[1]['corruption_k2_available'],0)
        self.assertEqual(result[2]['corruption_D_pred_missing_reason'],'NOT_IN_FROZEN_192_TARGET_ROSTER')

    def test_k2_signed_response_no_k1_and_cancellation(self):
        prior=engine.parent_engine(engine.PREVIOUS/'scripts/run_score_predictions.py')
        p=np.asarray([.4]+[.1]*6)
        versions={v:dict(eligible=True,probabilities=[x]+[(1-x)/6]*6) for v,x in zip(engine.VARIANTS,[.3,.5])}
        result=engine.k2_response(p,versions,prior.validate_probability)
        self.assertAlmostEqual(result['corruption_d_pred'],0.)
        self.assertAlmostEqual(result['corruption_d_abs'],.1)
        self.assertAlmostEqual(result['corruption_cancellation'],.1)
        versions['corruption_b']['eligible']=False
        self.assertIsNone(engine.k2_response(p,versions,prior.validate_probability))

    def test_reused_target_only_and_context_guard_cpu(self):
        prior=engine.parent_engine(engine.PREVIOUS/'scripts/run_score_predictions.py')
        clean=torch.zeros(3,4,1024);changed=prior.replace_target(clean,1,torch.ones(4,1024))
        self.assertTrue(torch.equal(clean,torch.zeros_like(clean)))
        self.assertTrue(torch.equal(clean[[0,2]],changed[[0,2]]))
        dialogue=dict(dialogue_id='synthetic',utterance_ids=['0','1','2'],texts=['a','b','c'],
            visual=torch.zeros(3,2),speaker=torch.zeros(3,2),labels=torch.zeros(3,dtype=torch.long))
        audio=torch.zeros(3,768)
        class Good:
            @staticmethod
            def infer_dialogue(model,dialogue,text,audio):return np.full((3,7),1/7)
        self.assertEqual(prior.locked_forward(Good,None,dialogue,clean,audio).shape,(3,7))
        class Bad:
            @staticmethod
            def infer_dialogue(model,dialogue,text,audio):
                audio[0,0]=1;return np.full((3,7),1/7)
        with self.assertRaises(prior.FrozenInputMutation):prior.locked_forward(Bad,None,dialogue,clean,audio)


if __name__=='__main__':unittest.main(verbosity=2)
