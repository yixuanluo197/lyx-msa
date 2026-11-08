"""Equivalent structural-index reuse; features and full conversation forward recomputed."""
import json
from pathlib import Path
import sys
import time
import numpy as np
import run_pilot as runner
from graph_index_cache import install_graph_cache


def main():
    root=Path(sys.argv[sys.argv.index('--root')+1]).resolve()
    dataset=sys.argv[sys.argv.index('--dataset')+1];modality=sys.argv[sys.argv.index('--modality')+1]
    assert '--stage' in sys.argv and sys.argv[sys.argv.index('--stage')+1]=='pilot'
    evidence={'protocol_change':False,'optimization':'reuse constant graph indices only; recompute all feature tensors and model operations',
              'wrapper_sha256':runner.sha(__file__),'cache_sha256':runner.sha(Path(__file__).with_name('graph_index_cache.py')),
              'runner_sha256':runner.sha(Path(runner.__file__)),'parity_checks':[], 'started_at':time.time()}
    original_load=runner.load_model;original_forward=runner.forward;models=[];batch_counts={}

    def load(*args,**kwargs):
        model=original_load(*args,**kwargs)
        model._pilot_graph_cache=install_graph_cache(model.graph_model)
        models.append(model)
        return model

    def forward(model,tensors,lengths):
        cache=model._pilot_graph_cache
        key=(id(model),tuple(lengths),tuple(tensors[0].shape))
        count=batch_counts.get(key,0)+1;batch_counts[key]=count
        cached=original_forward(model,tensors,lengths)
        # Baseline plus first target's Gaussian, Dropout and Attenuation in each original batch.
        if count in (1,2,7,12):
            cache.restore()
            try:reference=original_forward(model,tensors,lengths)
            finally:cache.enable()
            a=cached.detach().cpu().numpy();b=reference.detach().cpu().numpy()
            probabilities_passed=bool(np.allclose(np.exp(a),np.exp(b),atol=1e-6,rtol=1e-5))
            passed=bool(np.allclose(a,b,atol=1e-6,rtol=1e-5) and probabilities_passed and np.array_equal(a.argmax(-1),b.argmax(-1)))
            check={'batch_lengths':lengths,'call_in_batch':count,'passed':passed,'probabilities_passed':probabilities_passed,'max_log_probability_error':float(np.max(np.abs(a-b))),
                   'max_probability_error':float(np.max(np.abs(np.exp(a)-np.exp(b))))}
            evidence['parity_checks'].append(check)
            runner.write_json(root/f'provenance/cache_parity_{dataset}_{modality}.json',evidence)
            assert passed,'Cached graph forward differs from original full forward'
        return cached

    runner.load_model=load;runner.forward=forward
    try:
        runner.main()
        evidence.update(passed=True,finished_at=time.time(),cache_statistics=[m._pilot_graph_cache.stats for m in models])
    except BaseException as error:
        evidence.update(passed=False,error=repr(error),finished_at=time.time());raise
    finally:
        runner.write_json(root/f'provenance/cache_parity_{dataset}_{modality}.json',evidence)

if __name__=='__main__':main()
