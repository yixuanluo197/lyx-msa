"""Update navigation only after CPU final verification/analysis/plotting succeed."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import time

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def save(value):
    target = ROOT/'state/delivery.json'
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(target)


def main():
    state = {'active': True, 'phase': 'waiting_for_finalization', 'pid': os.getpid()}
    save(state)
    while True:
        if not (ROOT/'state/finalization.json').exists():
            time.sleep(20)
            continue
        finalization = read(ROOT/'state/finalization.json')
        if not finalization['active']:
            if finalization['phase'] != 'completed':
                state.update(active=False,phase='blocked',reason=finalization)
                save(state)
                return
            break
        time.sleep(20)
    verification = read(ROOT/'provenance/final_verification.json')
    summary = read(ROOT/'results/metrics_summary.json')
    plots = read(ROOT/'results/figures/manifest.json')
    assert verification['status'] == 'passed' and verification['formal_runs'] == 1
    assert verification['test_forwards'] == summary['test_forward_passes'] == 0
    assert not verification['cuda_initialized'] and summary['new_runs_completed'] == 1
    for artifact in plots['artifacts']:
        assert sha(ROOT/artifact['path']) == artifact['sha256']
    comp = summary['comparison'][0]
    assert comp['seed'] == 2024 and comp['lr'] == .0001
    category = comp['candidate_category']
    conclusions = {
        'second_seed_candidate': '本Seed满足第二Seed确认候选资格；尚未证实跨Seed稳定收益。',
        'mixed_evidence': '最佳值提高但末段下降，属于混合线索；不据此启动五Seed。',
        'no_best_improvement': '本Seed最佳验证W-F1未改善，不支持主指标提升；不自动扩大搜索。',
    }
    lines = ['# 第09组：MELD Batch Size诊断', '',
        '唯一登记正式训练已完成并通过独立CPU验收：完整非因果CMERC，MELD seed2024/split_seed1475、LR=0.0001、Batch8、40轮，验证W-F1选模、零测试前向。', '',
        '| 指标 | Batch16（XP04） | Batch8（09） |', '|---|---:|---:|',
        f"| 最佳验证W-F1 | {comp['control_best_validation_wf1']:.2f}% | {comp['new_best_validation_wf1']:.2f}% |",
        f"| 最佳Epoch | {comp['control_best_epoch']} | {comp['new_best_epoch']} |",
        f"| 末10轮验证均值 | {comp['control_last10_validation_wf1']:.3f}% | {comp['new_last10_validation_wf1']:.3f}% |", '',
        conclusions[category], '',
        '原--batch-size同时改变训练与验证分组；BN/padding及未pack双向GRU可能影响验证预测，另有更新次数、HSCL配对、CC尺度和RNG消耗耦合。单Seed不能证明某内部机制或论文差距根因。', '',
        '- [完整中文报告](../../reports/CMERC_BATCH_SIZE_DIAGNOSIS.md)',
        '- [独立批次](../../experiments/batch_size_diagnosis/20261024T131956Z/README.md)',
        '- [训练前登记](../../experiments/batch_size_diagnosis/20261024T131956Z/reports/PREREGISTRATION.md)',
        '- [完整argv](../../experiments/batch_size_diagnosis/20261024T131956Z/configs/registry.json)',
        '- [结果对照CSV](../../experiments/batch_size_diagnosis/20261024T131956Z/results/comparison.csv)',
        '- [独立验收](../../experiments/batch_size_diagnosis/20261024T131956Z/provenance/final_verification.json)',
        '- [图及SHA清单](../../experiments/batch_size_diagnosis/20261024T131956Z/results/figures/manifest.json)', '',
        '未追加任何训练或测试评估；本次未自动推送GitHub。数据、权重和完整原始日志在服务器独立批次保留。', '']
    (REPO/'experiment_records/09_batch_size_diagnosis/README.md').write_text('\n'.join(lines))
    (ROOT/'README.md').write_text('\n'.join([
        '# 第09组：MELD Batch Size诊断', '', '唯一正式40轮训练及独立CPU验收已完成，零测试前向。', '',
        '[完整报告](reports/CMERC_BATCH_SIZE_DIAGNOSIS.md)、[预登记](reports/PREREGISTRATION.md)、[配置](configs/registry.json)、[CSV](results/comparison.csv)、[验收](provenance/final_verification.json)、[曲线](results/figures/manifest.json)。', '',
        conclusions[category], '不自动追加训练或测试。', '']))
    state.update(active=False,phase='completed',completed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        result=comp,report_sha256=sha(ROOT/'reports/CMERC_BATCH_SIZE_DIAGNOSIS.md'),
        root_report_sha256=sha(REPO/'reports/CMERC_BATCH_SIZE_DIAGNOSIS.md'),plot_artifacts=len(plots['artifacts']))
    save(state)


if __name__ == '__main__':
    main()
