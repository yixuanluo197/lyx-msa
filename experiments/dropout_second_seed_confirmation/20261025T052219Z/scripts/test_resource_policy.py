"""Pure CPU checks for the shared-GPU engineering gate; never invoke CUDA or nvidia-smi."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_single as runtime


UUID = 'GPU-e53c5e2b-cabd-d4e7-ab11-b12f94557456'
POLICY = {'preexisting_gpu3_pids': [2527571, 2560936], 'estimated_own_peak_mib': 2200,
          'admission_min_free_mib': 3000, 'admission_max_utilization_percent': 95,
          'samples_required': 3, 'sample_seconds': 10, 'new_foreign_pids_pause_admission': False,
          'no_foreign_process_signals': True, 'co_resident_processes_disclosed': True}


def sample(own_pids=(), stage='preflight_admission', own_visible=True):
    preregistered = [{'pid': pid, 'name': 'python', 'used_memory': '430'} for pid in POLICY['preexisting_gpu3_pids']]
    other = [{'pid': 999, 'name': 'python', 'used_memory': '500'}]
    foreign = preregistered + other
    processes = ([{'pid': pid, 'name': 'python', 'used_memory': '1600'} for pid in own_pids]
                 if own_visible else []) + foreign
    return {'time': runtime.now(), 'stage': stage, 'gpu_index': 3, 'gpu_uuid': UUID,
            'gpu_name': 'TITAN Xp', 'total_mib': 12196, 'free_mib': 3000, 'used_mib': 9196,
            'utilization_percent': 95, 'processes': processes, 'own_pids': list(own_pids),
            'all_foreign_processes': foreign, 'allowed_foreign_processes': preregistered,
            'unexpected_processes': other}


def initial_state():
    return {'resource_query_errors': 0, 'resource_samples': 0, 'preexisting_pid_samples': 0,
            'unexpected_process_samples': 0, 'unexpected_pids': [], 'observed_foreign_pids': [],
            'admission_paused': False, 'admission_conditions_currently_met': True}


class ResourcePolicyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for name in ('logs', 'state', 'configs', 'preflight', 'results', 'runs', 'provenance'):
            (self.root / name).mkdir()
        self.root_patch = patch.object(runtime, 'ROOT', self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.addCleanup(self.temporary.cleanup)

    def test_capacity_boundaries_and_new_pid_disclosure_only(self):
        runtime.validate_resource_policy(POLICY)
        runtime.validate_resource_policy({**POLICY, 'preexisting_gpu3_pids': []})
        self.assertTrue(runtime.admissible(sample(), POLICY))
        self.assertFalse(runtime.admissible({**sample(), 'free_mib': 2999}, POLICY))
        self.assertFalse(runtime.admissible({**sample(), 'utilization_percent': 96}, POLICY))
        self.assertFalse(runtime.admissible(sample([42]), POLICY))
        self.assertFalse(runtime.admissible(None, POLICY))

    def test_new_pid_is_counted_without_pausing_capacity_admission(self):
        state = initial_state()
        with patch.object(runtime, 'resource_snapshot', return_value=sample()):
            result = runtime.observe({'resource_policy': POLICY}, state, 'preflight_admission')
        self.assertTrue(runtime.admissible(result, POLICY))
        self.assertEqual(state['unexpected_process_samples'], 1)
        self.assertEqual(state['unexpected_pids'], [999])
        self.assertEqual(state['observed_foreign_pids'], [999, 2527571, 2560936])
        self.assertFalse(state['admission_paused'])
        self.assertTrue(state['admission_conditions_currently_met'])

    def test_failed_query_closes_gate_and_records_error(self):
        state = initial_state()
        with patch.object(runtime, 'resource_snapshot', side_effect=RuntimeError('CPU injected query failure')):
            result = runtime.observe({'resource_policy': POLICY}, state, 'formal_admission')
        self.assertIsNone(result)
        self.assertFalse(runtime.admissible(result, POLICY))
        self.assertTrue(state['admission_paused'])
        self.assertFalse(state['admission_conditions_currently_met'])
        self.assertEqual(state['resource_query_errors'], 1)
        self.assertTrue((self.root / 'logs/resource_errors.jsonl').exists())

    def test_wrong_gpu_identity_cannot_authorize_admission(self):
        class Response:
            def __init__(self, stdout):
                self.stdout = stdout
        registry = {'gpu_uuid': UUID, 'gpu_index': 3, 'resource_policy': POLICY}
        replies = [Response('3,GPU-wrong,TITAN Xp,12196,8000,4196,10\n'), Response('')]
        with patch.object(runtime.subprocess, 'run', side_effect=replies):
            result = runtime.observe(registry, initial_state(), 'preflight_admission')
        self.assertIsNone(result)

    def test_gpu0_and_gpu1_identity_snapshots_pass(self):
        class Response:
            def __init__(self, stdout):
                self.stdout = stdout
        for index in (0, 1):
            with self.subTest(gpu_index=index):
                uuid = runtime.HEALTHY_XP_GPUS[index]
                registry = {'gpu_uuid': uuid, 'gpu_index': index, 'resource_policy': POLICY}
                replies = [Response(f'{index},{uuid},TITAN Xp,12196,8000,4196,80\n'), Response('')]
                with patch.object(runtime.subprocess, 'run', side_effect=replies) as query:
                    result = runtime.resource_snapshot(registry, stage='preflight_admission')
                self.assertEqual(query.call_count, 2)
                self.assertEqual(result['gpu_index'], index)
                self.assertEqual(result['gpu_uuid'], uuid)
                self.assertTrue(runtime.admissible(result, POLICY))

    def test_cross_card_uuid_and_unregistered_device_are_rejected(self):
        with self.assertRaisesRegex(AssertionError, 'healthy XP mapping'):
            runtime.validate_device(0, runtime.HEALTHY_XP_GPUS[1])
        with self.assertRaisesRegex(AssertionError, 'healthy XP mapping'):
            runtime.validate_device(2, 'GPU-unregistered')

    def test_three_consecutive_samples_reset_after_error(self):
        observations = [sample(), None, sample(), sample(), sample()]
        registry = {'runs': [{'run_id': 'registered'}], 'resource_policy': POLICY}
        with patch.object(runtime, 'observe', side_effect=observations), patch.object(runtime.time, 'sleep') as sleep:
            admitted = runtime.wait_admission(registry, {}, 'preflight_admission')
        self.assertEqual(len(admitted), 3)
        self.assertEqual(sleep.call_count, 4)
        self.assertTrue(all(call.args == (10,) for call in sleep.call_args_list))

    def test_cpu_gate_requires_resource_guard_evidence(self):
        evidence = {'status': 'passed', 'cuda_initialized': False, 'tests': 8, 'failures': 0, 'errors': 0}
        for name in ('guard_tests.json', 'resource_guard_tests.json'):
            (self.root / 'preflight' / name).write_text(json.dumps(evidence))
        (self.root / 'preflight/capacity_cpu.json').write_text(json.dumps(
            {'status': 'passed', 'cuda_initialized': False, 'epochs_checked': 40, 'records': [{}]*40}))
        self.assertEqual(len(runtime.cpu_evidence()), 3)
        (self.root / 'preflight/resource_guard_tests.json').unlink()
        with self.assertRaises(FileNotFoundError):
            runtime.cpu_evidence()

    def test_probe_budget_and_positive_free_samples(self):
        run = {'run_id': 'registered'}
        folder = self.root / 'preflight/registered/observer'
        folder.mkdir(parents=True)
        status = {'phase': 'preflight_passed', 'first_step_completed': True,
                  'counts': {'train': 0, 'validation': 0, 'test': 0},
                  'test_forward_passes': 0, 'target_line_events': 0}
        first = {'gradients_finite': True, 'optimizer_updated_parameters': True, 'gradient_l2': 1.,
                 'components': {'classification_loss': 1., 'cc_loss': 2., 'mscl_loss': 3.,
                                'cscl_loss': 4., 'total_loss': 2.054}}
        (folder / 'status.json').write_text(json.dumps(status))
        (folder / 'first.json').write_text(json.dumps(first))
        audit = {'cli_dropout': .5, 'cli_rec_dropout_metadata': .1,
                 'models': {name: {'language_gru': {'dropout': .5, 'num_layers': 2, 'bidirectional': True},
                                  'hidden_dropout_p': .5, 'registered_fixed_modality_dropouts': {}}
                            for name in ('model', 'model_mm', 'model_intra', 'model_inter')},
                 'fixed_modality_dropout_definition': {'class': 'MMGatedAttention',
                    'values': {'dropouta': .5, 'dropoutv': .5, 'dropoutl': .5},
                    'active_instances': {name: False for name in ('model', 'model_mm', 'model_intra', 'model_inter')}},
                 'rng_before': 'unchanged', 'rng_after': 'unchanged'}
        (folder / 'module_audit.json').write_text(json.dumps(audit))
        (self.root / 'configs/registry.json').write_text(json.dumps({'resource_policy': POLICY}))
        memory_path = self.root / 'logs/preflight_registered_memory.jsonl'
        memory_path.write_text(json.dumps({'peak_allocated_mib': 939., 'peak_reserved_mib': 1630.}) + '\n')
        (self.root / 'logs/resources.jsonl').write_text(json.dumps(sample(stage='preflight_running')) + '\n')
        result = runtime.verify_probe(run)
        self.assertEqual(result['torch_peak_reserved_mib'], 1630.)
        self.assertEqual(result['registered_memory_budget_mib'], 2200)
        memory_path.write_text(json.dumps({'peak_allocated_mib': 939., 'peak_reserved_mib': 2201.}) + '\n')
        with self.assertRaisesRegex(AssertionError, 'registered Torch reserved-memory budget'):
            runtime.verify_probe(run)
        memory_path.write_text(json.dumps({'peak_allocated_mib': 939., 'peak_reserved_mib': 1630.}) + '\n')
        (self.root / 'logs/resources.jsonl').write_text(json.dumps({**sample(stage='preflight_running'), 'free_mib': 0}) + '\n')
        with self.assertRaises(AssertionError):
            runtime.verify_probe(run)

    def registry(self):
        run = {'run_id': 'MELD_cmerc_dropout0p5_lr1e-4_seed67137', 'dataset': 'MELD', 'seed': 67137,
               'lr': .0001, 'epochs': 40, 'batch_size': 16, 'validation_batch_size': 16, 'dropout': .5, 'rec_dropout': .1,
               'argv': ['--lr', '0.0001', '--batch-size', '16', '--dropout', '0.5', '--rec-dropout', '0.1'],
               'output_root': str(self.root / 'runs/formal')}
        return {'runs': [run], 'formal_run_count': 1, 'python': '/fake/python',
                'gpu_index': 3, 'gpu_uuid': UUID, 'environment': {'CUDA_VISIBLE_DEVICES': UUID},
                'resource_policy': copy.deepcopy(POLICY)}

    def simulate_controller(self, query_failure_after_launch=False):
        registry = self.registry()
        (self.root / 'configs/registry.json').write_text(json.dumps(registry))
        children = []
        class Child:
            def __init__(self, argv, **kwargs):
                self.pid, self.argv = 100 + len(children), argv
                self.returncode, self.ticks = None, 0
                children.append(self)
            def poll(self):
                self.ticks += 1
                if self.ticks >= 3:
                    self.returncode = 0
                return self.returncode
        def snapshot(registry, own_pids=(), stage=None):
            if query_failure_after_launch and children:
                raise RuntimeError('CPU injected persistent query failure')
            live = bool(children and children[-1].returncode is None)
            return sample(own_pids, stage, own_visible=live)
        patches = [patch.object(runtime, 'validate_capsule', return_value='frozen'),
                   patch.object(runtime, 'sha', return_value='frozen'),
                   patch.object(runtime, 'cpu_evidence', return_value=[]),
                   patch.object(runtime, 'validate_gate'),
                   patch.object(runtime, 'verify_probe', return_value={'status': 'passed', 'test_forward_passes': 0}),
                   patch.object(runtime, 'verify_formal'), patch.object(runtime.fcntl, 'flock'),
                   patch.object(runtime.subprocess, 'Popen', side_effect=Child),
                   patch.object(runtime.time, 'sleep'), patch.object(runtime, 'resource_snapshot', side_effect=snapshot)]
        for context in patches:
            context.start()
        try:
            if query_failure_after_launch:
                with self.assertRaisesRegex(AssertionError, 'resource queries failed'):
                    runtime.main()
            else:
                runtime.main()
                with self.assertRaisesRegex(AssertionError, 'Existing supervision state'):
                    runtime.main()
            return runtime.read(self.root / 'state/status.json'), children
        finally:
            for context in reversed(patches):
                context.stop()

    def test_unique_probe_then_formal_and_no_duplicate_or_test(self):
        state, children = self.simulate_controller()
        self.assertEqual(state['phase'], 'completed')
        self.assertEqual(state['preflight_started'], 1)
        self.assertEqual(state['formal_started'], 1)
        self.assertEqual(state['formal_completed'], 1)
        self.assertEqual(len(children), 2)
        self.assertIn('--preflight', children[0].argv)
        self.assertNotIn('--preflight', children[1].argv)
        self.assertTrue(all(child.returncode == 0 for child in children))
        self.assertGreater(state['unexpected_process_samples'], 0)
        self.assertFalse(state['new_foreign_pids_pause_admission'])
        self.assertEqual(state['observed_foreign_pids'], [999, 2527571, 2560936])
        gate = runtime.read(self.root / 'preflight/gate.json')
        self.assertEqual(gate['resource_policy'], POLICY)
        self.assertEqual(gate['test_forward_passes'], 0)

    def test_persistent_query_failure_waits_for_own_probe_and_prevents_formal(self):
        state, children = self.simulate_controller(query_failure_after_launch=True)
        self.assertEqual(state['phase'], 'failed')
        self.assertEqual(len(children), 1)
        self.assertEqual(children[0].returncode, 0)
        self.assertEqual(state['preflight']['returncode'], 0)
        self.assertEqual(state['formal_started'], 0)
        self.assertEqual(state['active_pids'], [])
        self.assertFalse((self.root / 'preflight/gate.json').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
