"""CPU-only device selection and duplicate-launch safety checks."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock
import select_xp_and_launch as selected

class SelectionTests(unittest.TestCase):
    def test_gpu1_selected_once_and_frozen_before_controllers(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for n in ['state','provenance','configs','logs','results','preflight']:(root/n).mkdir()
            registry={'formal_run_count':1,'runs':[{'run_id':'only','output_root':str(root/'runs/only'),'argv':['--lr','0.0001','--batch-size','8']}],
                      'gpu_index':3,'gpu_uuid':selected.DEVICES[3], 'environment':{'CUDA_VISIBLE_DEVICES':selected.DEVICES[3],'OMP_NUM_THREADS':'1'},
                      'resource_policy':{'preexisting_gpu3_pids':[5],'preexisting_memory_mib':100}}
            selected.write(root/'provenance/selection_registry_template.json',registry)
            selected.write(root/'provenance/frozen_selection_base_v3.json',{'files':{}})
            output='\n'.join(f'{i}, {u}, NVIDIA TITAN Xp, {6000 if i==1 else 1023}, 50' for i,u in selected.DEVICES.items())
            starts=[]
            def launch(argv,**kwargs):
                # No controller can start before selected config and hashes exist.
                config=selected.read(root/'configs/registry.json');capsule=selected.read(root/'provenance/frozen_execution.json')
                self.assertEqual(config['gpu_index'],1);self.assertEqual(config['environment']['CUDA_VISIBLE_DEVICES'],selected.DEVICES[1])
                self.assertEqual(config['runs'],registry['runs'])
                for n,d in capsule['files'].items():self.assertEqual(selected.sha(root/n),d)
                starts.append(argv);return Mock(pid=100+len(starts))
            with patch.object(selected,'ROOT',root),patch.object(selected.subprocess,'check_output',side_effect=[output,output,output,'']),patch.object(selected.subprocess,'Popen',side_effect=launch),patch.object(selected.time,'sleep') as sleep:
                selected.main()
            self.assertEqual(len(starts),3);self.assertEqual(sleep.call_count,2)
            result=selected.read(root/'state/selection.json');self.assertFalse(result['active'])
            self.assertEqual(result['selected']['selected_gpu_index'],1)
            self.assertEqual(len(result['selected']['selection_samples']),3)
            self.assertEqual(selected.read(root/'configs/registry.json')['resource_policy']['preexisting_gpu3_pids'],[])
    def test_existing_controller_rejects_duplicate_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'state').mkdir();(root/'state/status.json').write_text('{}')
            with patch.object(selected,'ROOT',root),patch.object(selected.subprocess,'check_output') as query,patch.object(selected.subprocess,'Popen') as launch:
                with self.assertRaisesRegex(AssertionError,'duplicate controller'):selected.main()
                query.assert_not_called();launch.assert_not_called()

if __name__=='__main__':unittest.main()
