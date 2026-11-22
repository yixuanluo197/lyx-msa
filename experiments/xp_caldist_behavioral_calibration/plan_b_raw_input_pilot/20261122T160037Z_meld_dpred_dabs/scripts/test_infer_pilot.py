#!/usr/bin/env python3
"""Synthetic CPU-only Pilot adapter tests; no real exports or GPU/model loads."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

PATH=Path(__file__).with_name('infer_pilot.py')
spec=importlib.util.spec_from_file_location('synthetic_pilot_adapter',PATH)
pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)


def source_fixture():
    data=dict(dataset='MELD',seed=67137,class_names=[str(i) for i in range(7)],splits={})
    audio={};cache=dict(clean={},clean_token_ids={});cohort=[];source=[];targets=[]
    p=np.asarray([.4]+[.1]*6)
    for role,did in [('pilot_fit','syntheticD1'),('pilot_eval','syntheticD2')]:
        texts=['We are waiting outside.','A fixed synthetic context.'];uids=['0','1']
        d=dict(dialogue_id=did,utterance_ids=uids,texts=texts,speaker=torch.zeros(2,3),visual=torch.zeros(2,342),labels=torch.zeros(2,dtype=torch.long))
        data['splits'][role]=[d];audio[role]={did:torch.zeros(2,768)};cache['clean'][role]={did:torch.zeros(2,4,1024)}
        for i,uid in enumerate(uids):
            identity=dict(sample_id=f'MELD:train:{did}:{uid}',role=role,dialogue_id=did,utterance_id=uid,utterance_index=str(i))
            cohort.append(dict(identity));source.append(dict(identity,original_role='score_fit',original_correct=str(i%2),
                original_prediction='0',original_confidence=str(p.max()),clean_margin=str(.3),
                clean_entropy=str(float(-np.sum(p*np.log(p)))),original_probabilities=json.dumps(p.tolist())))
            if i==0:
                targets.append(dict(identity,blind_id='blind_'+did));cache['clean_token_ids'][identity['sample_id']]=[0]+list(texts[i].encode())+[2]
    return data,audio,cache,cohort,source,targets


def validator(*values):return pilot.validate_exports(*values,expected_roles={'pilot_fit':1,'pilot_eval':1})


def rank_helper():return pilot.load_module('synthetic_candidate_helper',Path(__file__).with_name('freeze_pilot.py')).ranked_candidates


def reviewed_fixture(index,roster,options):
    rows=[];ranked=rank_helper();base=pilot.parent_adapter()
    for sid,t in index.items():
        if sid not in roster:continue
        offered=ranked(t['original_text'],roster[sid]['blind_id'],options)
        for i,v in enumerate(pilot.VARIANTS):
            text,edit=offered[i] if i<len(offered) else (t['original_text'],None)
            identity={k:t[k] for k in ('sample_id','role','dialogue_id','utterance_index','utterance_id')}
            rows.append(dict(identity,variant_id=v,processed_text=text,edit=edit,
                processed_text_sha256=base.text_sha(text),original_text_sha256=base.text_sha(t['original_text']),
                generation_status='GENERATED' if edit else 'NO_ELIGIBLE_EDIT',review_status='PASS' if edit else 'FAIL',
                review_method='AI_CONTEXT_ORTHOGRAPHIC_REVIEW' if edit else 'NO_CANDIDATE',reviewer='Synthetic judge',review_notes='Synthetic fixture.'))
    return rows


class TextStub:
    @staticmethod
    def preprocessing(text,dataset):return text.encode('ascii',errors='ignore').decode()
    @staticmethod
    def tokenize(tokenizer,texts,dataset):
        return [[0]+list(TextStub.preprocessing(t,dataset).encode())+[2] for t in texts],[dict(truncated=False,empty_after_processing=not t) for t in texts]
    @staticmethod
    def extract_texts(model,tokenizer,texts,dataset,batch_size):
        return torch.stack([torch.full((4,1024),int(pilot.parent_adapter().text_sha(t)[:8],16)/2**32) for t in texts])


class PilotSynthetic(unittest.TestCase):
    def setUp(self):self.values=source_fixture()

    def test_exact_physical_score_fit_only_export_contract(self):
        index,roster,lookup=validator(*self.values)
        self.assertEqual(len(index),4);self.assertEqual(len(roster),2);self.assertEqual(len(lookup),4)
        bad=copy.deepcopy(self.values);bad[0]['splits']['score_eval']=[]
        with self.assertRaisesRegex(ValueError,'no old score_eval'):validator(*bad)
        bad=copy.deepcopy(self.values);bad[4][0]['original_role']='score_eval'
        with self.assertRaisesRegex(ValueError,'wrong original role'):validator(*bad)

    def test_manifest_all_clean_rows_and_fixed_target_identity(self):
        bad=copy.deepcopy(self.values);bad[3].pop()
        with self.assertRaisesRegex(ValueError,'exact clean cohort'):validator(*bad)
        bad=copy.deepcopy(self.values);bad[4].pop()
        with self.assertRaisesRegex(ValueError,'Missing/duplicate'):validator(*bad)
        bad=copy.deepcopy(self.values);bad[5][1]['sample_id']=bad[5][0]['sample_id']
        with self.assertRaises(ValueError):validator(*bad)

    def test_legacy_behavior_columns_and_bad_scalar_are_rejected(self):
        bad=copy.deepcopy(self.values);bad[4][0]['D_pred']='.1'
        with self.assertRaisesRegex(ValueError,'Previous behavioral'):validator(*bad)
        bad=copy.deepcopy(self.values);bad[4][0]['original_confidence']='0.9'
        with self.assertRaisesRegex(ValueError,'scalar mismatch'):validator(*bad)

    def test_shape_precision_and_target_tokens_exact(self):
        bad=copy.deepcopy(self.values);bad[1]['pilot_fit']['syntheticD1']=torch.zeros(2,767)
        with self.assertRaisesRegex(ValueError,'shape/precision'):validator(*bad)
        bad=copy.deepcopy(self.values);bad[2]['clean_token_ids']['UNPLANNED']=[0,2]
        with self.assertRaisesRegex(ValueError,'one unchanged target'):validator(*bad)

    def test_new_four_letter_capitalized_rule_and_collision_guard(self):
        ranked=rank_helper();options=dict(seed=1,allowlist=['have','about'],collision_words=[])
        candidates=ranked('Have about?','synthetic_blind',options)
        self.assertEqual(len(candidates),2)
        self.assertTrue(any(e['original_word']=='Have' for _,e in candidates) or all(e['original_word']=='about' for _,e in candidates))
        one=ranked('Have','synthetic_blind',options)
        self.assertEqual(len(one),1);self.assertEqual(one[0][1]['edited_word'],'Hvae')
        self.assertEqual(ranked('Have','synthetic_blind',dict(options,collision_words=['hvae'])),[])
        self.assertEqual(ranked('x Have','synthetic_blind',options),[])

    def test_review_exact_generation_and_no_adaptive_replacement(self):
        index,roster,_=validator(*self.values);options=dict(seed=1,allowlist=['waiting','outside'],collision_words=[])
        rows=reviewed_fixture(index,roster,options);base=pilot.parent_adapter()
        reviewed=pilot.validate_reviews(rows,index,roster,rank_helper(),options,base.text_sha)
        self.assertEqual(sum(map(len,reviewed.values())),4)
        bad=copy.deepcopy(rows);bad[0]['edit']['position']+=1
        with self.assertRaises(ValueError):pilot.validate_reviews(bad,index,roster,rank_helper(),options,base.text_sha)
        bad=copy.deepcopy(rows);bad[0]['original_correct']=1
        with self.assertRaisesRegex(ValueError,'leakage'):pilot.validate_reviews(bad,index,roster,rank_helper(),options,base.text_sha)
        with self.assertRaises(ValueError):pilot.validate_reviews(rows[:-1],index,roster,rank_helper(),options,base.text_sha)

    def test_reused_encoder_and_consumer_accept_only_new_role_targets(self):
        data,audio,cache,cohort,source,targets=self.values;index,roster,_=validator(*self.values)
        base=pilot.parent_adapter();old=base.parent_engine(pilot.PREVIOUS/'scripts/run_score_predictions.py')
        options=dict(seed=1,allowlist=['waiting','outside'],collision_words=[])
        reviews=pilot.validate_reviews(reviewed_fixture(index,roster,options),index,roster,rank_helper(),options,base.text_sha)
        state=SimpleNamespace(index=index,roster=roster,cache=cache,engine=old,reviews=reviews,
            inputs={'synthetic':'hash'},data=data,text_lock={'model_sha256':'synthetic_weights'})
        records=base.encode_records(state,TextStub,None,None)
        runtime=dict(environment='TEXT_repro',torch='2.2.0+cu118',transformers='4.42.3')
        bundle=dict(format=base.FORMAT,inputs=state.inputs,protocol_sha256='synthetic',class_names=data['class_names'],fp32=True,
            text_model_sha256='synthetic_weights',encoder_state_before='same',encoder_state_after='same',runtime=runtime,variants=records)
        manifest=dict(format=base.FORMAT,source_bindings=state.inputs,protocol_sha256='synthetic',runtime=runtime,
            encoder_state_before='same',encoder_state_after='same')
        base.validate_encoded_records(bundle,manifest,state,'synthetic')
        bad=copy.deepcopy(bundle);first=next(iter(bad['variants']));bad['variants'][first]['corruption_a']['role']='score_eval'
        with self.assertRaises(ValueError):base.validate_encoded_records(bad,manifest,state,'synthetic')

    def test_same_k2_dpred_dabs_joint_missing_and_clean_cells_exact(self):
        source=self.values[4];sid=source[0]['sample_id']
        result=pilot.make_inputs(source,{sid:dict(corruption_d_pred=-.1,corruption_d_abs=.2)}, {})
        self.assertEqual(result[0]['D_pred'],-.1);self.assertEqual(result[0]['D_abs'],.2)
        for old,new in zip(source,result):self.assertEqual({k:new[k] for k in old},old)
        for row in result[1:]:self.assertEqual((row['D_pred'],row['D_abs'],row['k2_available']),('','',0))
        with self.assertRaises(RuntimeError):pilot.make_inputs(source,{sid:dict(corruption_d_pred=.2,corruption_d_abs=.1)}, {})

    def test_same_successful_forwards_and_no_k1_metric(self):
        base=pilot.parent_adapter();old=base.parent_engine(pilot.PREVIOUS/'scripts/run_score_predictions.py');p=[.4]+[.1]*6
        versions={v:dict(eligible=True,probabilities=[x]+[(1-x)/6]*6) for v,x in zip(pilot.VARIANTS,[.3,.5])}
        values=base.k2_response(p,versions,old.validate_probability)
        self.assertAlmostEqual(values['corruption_d_pred'],0);self.assertAlmostEqual(values['corruption_d_abs'],.1)
        versions['corruption_a']['eligible']=False
        self.assertIsNone(base.k2_response(p,versions,old.validate_probability))

    def test_physical_jsonl_unicode_and_no_old_common_calls(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'synthetic.jsonl';rows=[{'text':'x\u0085y\u2028z\u2029w'},{'text':'next'}]
            path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
            self.assertEqual(pilot.read_jsonl(path),rows)
        tree=ast.parse(PATH.read_text())
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='common' for n in ast.walk(tree)))

    def test_help_does_not_load_real_assets(self):
        result=subprocess.run([sys.executable,str(PATH),'--help'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,msg=result.stderr);self.assertIn('--stage {encode,infer}',result.stdout)


if __name__=='__main__':unittest.main(verbosity=2)
