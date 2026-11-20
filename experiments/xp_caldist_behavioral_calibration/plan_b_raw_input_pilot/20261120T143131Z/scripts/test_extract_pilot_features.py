#!/usr/bin/env python3
"""Synthetic producer contract checks; no dataset/model/media loading or GPU use."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

PATH=Path(__file__).with_name('extract_pilot_features.py')
spec=importlib.util.spec_from_file_location('producer_under_test',PATH)
producer=importlib.util.module_from_spec(spec);spec.loader.exec_module(producer)


class ProducerContract(unittest.TestCase):
    def test_runtime_boundary(self):
        actual=producer.shared.require_runtime('encoder')
        self.assertEqual(actual['environment'],'TEXT_repro')

    def test_token_ids_hashes_actual_truncated_ids(self):
        class FakeTextModule:
            @staticmethod
            def tokenize(tokenizer,texts,dataset):
                return [[0,13,2]],[{'tokens_before':3,'tokens_after':3,'truncated':False}]
            @staticmethod
            def preprocessing(text,dataset):return text.upper()
        record=producer.actual_text_input(FakeTextModule(),object(),'test','MELD')
        self.assertEqual(record['actual_token_ids'],[0,13,2])
        self.assertEqual(record['actual_token_ids_sha256'],producer.shared.array_hash(np.array([0,13,2],dtype=np.int64)))
        self.assertEqual(record['extractor_processed_text_sha256'],producer.text_sha('TEST'))

    def test_finite_shape_dtype_hash_contract(self):
        clean=torch.ones(4,1024,dtype=torch.float32)
        changed=clean*2
        record=producer.extracted_record({'modality':'text'},changed,clean)
        self.assertEqual(record['status'],'EXTRACTED')
        self.assertEqual(record['target_feature_sha256'],producer.shared.array_hash(changed))
        self.assertEqual(record['original_feature_sha256'],producer.shared.array_hash(clean))
        self.assertEqual(record['feature'].device.type,'cpu')
        with self.assertRaises(ValueError):producer.extracted_record({'modality':'text'},changed.double(),clean)
        with self.assertRaises(ValueError):producer.extracted_record({'modality':'audio'},changed,clean)
        changed[0,0]=float('nan')
        with self.assertRaises(ValueError):producer.extracted_record({'modality':'text'},changed,clean)

    def test_failures_never_main_eligible(self):
        result=producer.failed_record({'label_preserving_main_eligible':True},'fixture failure')
        self.assertFalse(result['label_preserving_main_eligible'])
        self.assertEqual(result['error'],'fixture failure')

    def test_fixed_variants_include_failures_and_reject_omission(self):
        rows=[{'sample_id':'s','perturbation_type':family,'variant_id':variant,'status':'FAILED'}
              for family in ('neutral_append','semantic_paraphrase') for variant in producer.shared.TYPES[family]]
        producer.validate_planned(rows,{'s'},'text')
        with self.assertRaises(ValueError):producer.validate_planned(rows[:-1],{'s'},'text')
        with self.assertRaises(ValueError):producer.validate_planned(rows+[rows[0]],{'s'},'text')
        wrong=[dict(row) for row in rows];wrong[0]['perturbation_type']='pitch_shift'
        with self.assertRaises(ValueError):producer.validate_planned(wrong,{'s'},'text')
        wrong[0]['perturbation_type']='unknown_family'
        with self.assertRaises(ValueError):producer.validate_planned(wrong,{'s'},'text')
        wrong=[dict(row) for row in rows];wrong[0]['modality']='audio'
        with self.assertRaises(ValueError):producer.validate_planned(wrong,{'s'},'text')

    def test_target_identity_cannot_move_to_another_utterance(self):
        sample=dict(sample_id='s',dialogue_id='dialogue_a',utterance_index=3)
        producer.validate_target_identity(dict(sample),sample)
        wrong=dict(sample,utterance_index=4)
        with self.assertRaises(ValueError):producer.validate_target_identity(wrong,sample)
        with self.assertRaises(ValueError):producer.extracted_record({'modality':'visual'},torch.zeros(768),torch.zeros(768))

    def test_file_identity_and_state_mutation_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'fixture.txt';path.write_text('initial')
            inputs={};digest=producer.bind_file(inputs,path)
            path.write_text('mutated')
            with self.assertRaises(ValueError):producer.bind_file(inputs,path,digest)
        model=torch.nn.Sequential(torch.nn.Linear(3,2),torch.nn.BatchNorm1d(2)).eval()
        before=producer.state_fingerprint(model)
        with torch.no_grad():model[1].running_mean[0]=1
        after=producer.state_fingerprint(model)
        self.assertNotEqual(before['state_sha256'],after['state_sha256'])
        self.assertNotEqual(before['bn_sha256'],after['bn_sha256'])

    def test_actual_consumer_accepts_synthetic_bundle_and_rejects_token_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            args=SimpleNamespace(dataset='IEMOCAP',dataset_dir=root,
                text_features=root/'text.pt',audio_features=root/'audio.pt',checkpoint=root/'model.pt')
            inputs={}
            for path in (root/'dataset.pt',args.text_features,args.audio_features,args.checkpoint):
                path.write_text('synthetic fixture only');producer.bind_file(inputs,path)
            tokens=[0,13,2]
            identity=dict(sample_id='train_smoke',dialogue_id='synthetic_dialogue',utterance_index=0)
            text=producer.extracted_record(dict(identity,modality='text',actual_token_ids=tokens,
                actual_token_ids_sha256=producer.shared.array_hash(np.asarray(tokens,dtype=np.int64))),
                torch.zeros(4,1024),torch.zeros(4,1024))
            audio=producer.extracted_record(dict(identity,modality='audio'),torch.zeros(768),torch.zeros(768))
            states={'text':producer.state_fingerprint(torch.nn.Linear(2,2)),
                    'audio':producer.state_fingerprint(torch.nn.Linear(2,2))}
            metadata=dict(format=producer.FORMAT,mode='smoke',dataset='IEMOCAP',status='COMPLETE',
                runtime=producer.shared.require_runtime('encoder'),input_sha256=inputs,
                scientific_protocol_sha256=producer.shared.sha256(producer.ROOT/'configs/PLAN_B_PROTOCOL.json'),
                encoder_states_unchanged=True,input_files_unchanged=True,
                encoder_states_before=states,encoder_states_after=states)
            bundle=dict(metadata=metadata,clean_controls={'train_smoke':{'text':text,'audio':audio}},
                        variants=[dict(text,perturbation_type='neutral_append',variant_id='neutral_append'),
                                  dict(audio,perturbation_type='pitch_shift',variant_id='pitch_plus_0.5')])
            path=root/'cache.pt'
            def write():
                producer.atomic_bundle(path,bundle)
                producer.shared.save_json(path.with_suffix('.manifest.json'),dict(cache_sha256=producer.shared.sha256(path),
                    format=producer.FORMAT,mode='smoke',dataset='IEMOCAP',status='COMPLETE',metadata=metadata))
            write()
            restored=producer.shared.load_precomputed_features(path,args,'smoke')
            self.assertEqual(len(restored['variants']),2)
            bundle['variants'][0]['actual_token_ids']=[0,14,2]
            write()
            with self.assertRaises(ValueError):producer.shared.load_precomputed_features(path,args,'smoke')


if __name__=='__main__':unittest.main()
