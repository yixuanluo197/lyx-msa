"""Summarize completed local GPU evaluation jobs; refuses incomplete evidence."""

from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main():
    jobs=json.loads((ROOT/'provenance/job_registry.json').read_text())
    assert len(jobs)==8
    rows=[]
    for job in jobs:
        path=ROOT/f"runs/{job['dataset']}/seed{job['seed']}/complete.json"
        e=json.loads(path.read_text())
        assert e['status']=='completed' and e['weights_unchanged'] and e['batch_inputs_restored']
        gpu_lines=e['resource']['gpu'].splitlines()
        line=[line for line in gpu_lines if line.split(',')[0].strip()==str(job['host_gpu'])]
        assert len(line)==1
        uuid=line[0].split(',')[1].strip()
        rows.append({'dataset':job['dataset'],'seed':job['seed'],'host':e['resource']['host'],
                     'host_gpu':job['host_gpu'],'gpu_uuid':uuid,'pid':job['pid'],'utterances':e['utterances'],
                     'target_forwards':e['perturbed_rows'],'elapsed_seconds':e['seconds'],
                     'forwards_per_second':e['perturbed_rows']/e['seconds'],
                     'peak_allocated_mib':e['peak_allocated_mib'],'peak_reserved_mib':e['peak_reserved_mib'],
                     'probability_parity_max_abs_error':e['parity']['checks']['probabilities']['max_abs_error'],
                     'log_probability_parity_max_abs_error':e['parity']['checks']['log_probabilities']['max_abs_error'],
                     'cached_uncached_checks':e['cached_uncached_parity_checks'],'target_mask_checks':e['target_mask_checks'],
                     'graph_hook_checks':e['graph_hook_checks'],'weights_unchanged':e['weights_unchanged']})
    df=pd.DataFrame(rows)
    assert df.target_forwards.sum()==130880
    df.to_csv(ROOT/'gpu_resource_summary.csv',index=False)
    records=['# 五Seed GPU推理与完整性证据','',
             'Seed1475的joint结果复用已核验Pilot；其余4个Seed在同一台本地服务器补齐。没有CMERC训练或权重更新。', '',
             '| Dataset | Seed | GPU | UUID | Seconds | Peak allocated MiB | Probability parity max error |',
             '|---|---:|---:|---|---:|---:|---:|']
    for r in df.itertuples():
        records.append(f'| {r.dataset} | {r.seed} | {r.host_gpu} | {r.gpu_uuid} | {r.elapsed_seconds:.2f} | {r.peak_allocated_mib:.1f} | {r.probability_parity_max_abs_error:.3g} |')
    records+=['','总计130880次新增目标前向；包含复用Seed1475的32720条joint记录后，完整五Seed行为诊断共有163600条扰动记录。', '',
              '每batch保持原mixed-dialogue batch16，核对uncached/cached基线及扰动前向；每Dialogue核对目标话语之外的全部特征和mask不变，每batch结束恢复原张量，最后完整state_dict精确相同。', '',
              '正式数据加载器会反序列化含多划分条目的原特征容器，但只使用真正Validation keys索引话语；Test标签不用于推理评价、校准拟合或配置选择。', '',
              '两任务/卡的容量试运行无OOM，显存留有余量。运行期同卡并发增大总吞吐，但每任务速度受共享资源影响，不能作为模型性能速度比较。详见job_registry和resource_monitor。']
    (ROOT/'reports/GPU_EXECUTION_REPORT.md').write_text('\n'.join(records)+'\n')
    print(df[['dataset','seed','host_gpu','target_forwards','elapsed_seconds','peak_allocated_mib']].to_string(index=False))


if __name__=='__main__':
    main()
