"""Generate reports from completed extension measurements; never load a model."""
from pathlib import Path
import json
import math

D=Path(__file__).resolve().parents[1]


def main():
    summary=json.loads((D/'results/extension_summary.json').read_text())
    execution=json.loads((D/'results/execution.json').read_text())
    assert execution['status']=='COMPLETED'
    report=['# XP04 RoBERTa 小样本蒸馏延长实验', '',
        '本轮完成两个独立的固定128条Train Fit实验。原500步学生权重、选样顺序、Token ID及四层教师目标均保持不变；仅在新目录继续训练。结果属于训练样本拟合诊断，不能外推为独立泛化或原XP04概率兼容。', '',
        '## 初始化与固定协议', '',
        '- 来源版本：5957f296ea7a0c27da1e647670717efd1f19fa9a；原诊断20261118T171911Z；服务器user23，IEMOCAP使用GPU0、MELD使用GPU2。命令、设备UUID、PID、退出码见results/execution.json。',
        '- 旧500步模型没有优化器/RNG/采样器状态。本轮加载其实际final_student权重，重新初始化AdamW与采样器，属于权重延续，不是原训练状态的精确断点恢复；500步初始化损失与旧记录逐层逐值相等。',
        '- 保持FP32、dropout=0、weight decay=0、AdamW lr=1e-5、betas=(0.9,0.999)、eps=1e-8、无调度器、梯度裁剪1、microbatch4×累积4、nonreentrant gradient checkpointing。所有编码器参数可训练，无额外头/映射/归一化。',
        '- 直接复用旧batch/features/evaluate：最后四层[-1,-2,-3,-4]取BOS；逐层MSE按128×1024平均，再平均四层。总体RMSE=sqrt(总体MSE)，不是四个RMSE的均值。',
        '- 先固定到总步数2000；严格使用(MSE1500−MSE2000)/MSE1500>5%决定是否继续到5000。额外750/1250/1750评估只记录轨迹，不影响决策。没有基于中途损失改参、延长预算或选择最佳点替代终点。',
        '- 逐层RMSE≤1e-4仍是原兼容性的必要条件；不因损失下降而放宽。即使此条件满足，也不等于完整特征Parity或冻结XP04概率Parity通过。',
        '- 正式训练前的两步smoke只作工程检查；正式任务重新加载原500步权重。smoke与正式均保留结果，包括新优化器启动后非单调变化。', '',
        '## 完整评估轨迹', '',
        '| 数据集 | 总步数 | 四层总体MSE | 总体RMSE | 相邻评估点MSE降幅 |',
        '|---|---:|---:|---:|---:|']
    decision_rows=[]
    layer_rows=[]
    detail=[]
    for ds,s in summary.items():
        run=D/'local_only/formal'/ds
        evaluations=json.loads((run/'evaluations.json').read_text())
        complete=json.loads((run/'completion.json').read_text())
        for item in evaluations:
            relative=item['relative_decrease_from_previous_eval']
            change='—' if relative is None else f'{100*relative:.3f}%'
            report.append(f"| {ds} | {item['global_step']} | {item['mse']:.10g} | {item['overall_rmse']:.8g} | {change} |")
        for layer,(initial,final) in enumerate(zip(evaluations[0]['layer_mse'],s['final_layer_mse']),1):
            rmse=math.sqrt(final)
            layer_rows.append(f'| {ds} | -{layer} | {initial:.10g} | {final:.10g} | {rmse:.8g} | {rmse/1e-4:.2f}× | {"PASS" if rmse<=1e-4 else "FAIL"} |')
        rule=s['decision']
        decision_rows.append(f"| {ds} | {100*rule['relative_decrease']:.3f}% | {'继续到5000' if rule['continue_to_5000'] else '停止于2000'} | {s['final_global_step']} |")
        plateau='未超过5%，按预定窗口规则表现为改善不足/平台或回退' if s['final_window_plateau_by_5percent_rule'] else '仍超过5%，该终点窗口未显示平台'
        finaltrend='下降' if s['relative_reduction_from_step500']>=0 else '上升'
        detail.extend([f'### {ds}', '',
            f"相对于500步，总体MSE{finaltrend}{100*abs(s['relative_reduction_from_step500']):.3f}%，最终MSE={s['final_mse']:.10g}。最终{s['final_window_start']}→{s['final_global_step']}步降幅为{100*s['final_window_relative_decrease']:.3f}%，{plateau}。",
            '', f"最终四层RMSE范围{min(s['final_layer_rmse']):.8g}–{max(s['final_layer_rmse']):.8g}，为1e-4上限的{min(s['final_layer_rmse'])/1e-4:.2f}–{max(s['final_layer_rmse'])/1e-4:.2f}倍；必要条件{'四层均通过' if s['rmse_necessary_condition_pass'] else '未全部通过'}。", '',
            f"重点两层：最后一层RMSE从{math.sqrt(evaluations[0]['layer_mse'][0]):.8g}变为{s['final_layer_rmse'][0]:.8g}，倒数第二层从{math.sqrt(evaluations[0]['layer_mse'][1]):.8g}变为{s['final_layer_rmse'][1]:.8g}。这两层的残差与下降情况均保留，不能由四层平均值代替。", '',
            f"新增优化器更新{complete['new_optimizer_steps']}次；总步数计入旧500步，累计话语呈现{s['final_global_step']*16}次，相当于每条文本平均{s['final_global_step']*16/128:g}次。最终权重SHA-256：`{complete['final_model_sha256']}`。完整训练状态：`local_only/formal/{ds}/{complete['final_training_checkpoint']}`。", ''])
    report.extend(['', '## 1500→2000决策', '', '| 数据集 | 总体MSE降幅 | 严格>5%规则 | 实际终点 |', '|---|---:|---|---:|', *decision_rows, '',
        '## 最终四层对比', '', '| 数据集 | 层 | 500步MSE | 最终MSE | 最终RMSE | RMSE/1e-4 | 必要条件 |', '|---|---:|---:|---:|---:|---:|---|', *layer_rows, '', *detail,
        '## 曲线与解释限制', '', '![训练延长曲线](results/training_extension.png)', '',
        '曲线连接的是固定完整128条评估点，逐步优化损失另外完整保存在optimizer_steps.csv；两者口径不同，不能用某个微批次损失替代完整评估。图中500步竖线标示新AdamW状态初始化。负降幅及波动全部保留，未挑最好checkpoint替代预定终点。', '',
        '5%规则是本轮指定的有界资源决策，不是损失已经数学收敛的证明。5000步终点的4500→5000窗口只描述趋势，不再触发更多训练。重置AdamW/采样器与追加训练共同发生，不能与保留原优化器状态的反事实轨迹直接比较，也不能单独归因于某个因素。', '',
        '仅拟合原128条训练文本，不估计全训练集/未见文本泛化，不访问官方Validation/Test，不进行XP04概率推理或文本干扰。原IEMOCAP教师历史划分重叠风险仍沿用既有说明，本轮未进一步核查或宣称消除。', '',
        '## 产物与核验', '',
        '- configs/EXTENSION_PROTOCOL.json及provenance/launch_code_lock.json：运行前固定协议和代码哈希。',
        '- results/per_layer_comparison.csv、stage_comparison.csv、extension_summary.json：四层、每阶段及终点统计；training_extension.png/pdf：曲线。',
        '- local_only/formal/{dataset}/final_student/：最终学生；stage_2000_checkpoint.pt及可适用的final_training_checkpoint.pt包含实际模型、AdamW、随机状态和采样器进度。',
        '- local_only/formal/{dataset}/：完整逐步CSV、所有评估、选样及目标哈希、决策、梯度检查、500步复现和checkpoint清单；logs/：原始日志。',
        '- provenance/original_input_hashes_before.json记录467项原诊断文件及Train缓存；最终独立核验结果见verification.json。核验会检查旧文件不变、新权重/完整状态及逐步记录，而不执行新模型推理。',
        '- 所有模型/完整特征/逐话语文件保持本地；未自动Git commit或push。完成本轮后停止。', ''])
    (D/'TRAINING_EXTENSION_REPORT.md').write_text('\n'.join(report))
    all_pass=all(s['rmse_necessary_condition_pass'] for s in summary.values())
    recommendation=('两套均满足小样本RMSE必要条件，可以讨论下一轮Train内部泛化验证，但尚不构成直接完整蒸馏或文本干扰的成功依据。' if all_pass else '不建议直接按相同配置启动第二轮完整数据集蒸馏。小样本延长仍未全部满足严格RMSE必要条件，尚不能用扩大样本量或继续堆步数来保证兼容。若后续另行研究优化策略，应先在Train内部提出明确改进目标；本轮不执行额外训练。')
    final=['# 最终延长实验判断', '', '两个固定128条实验已按预定条件结束。诊断执行完成与原特征兼容成功分开判断。', '',
        '## 1. 延长后MSE是否继续下降？', '']
    for ds,s in summary.items():
        final.append(f"- {ds}：500步{s['initial_mse']:.10g} → 总{s['final_global_step']}步{s['final_mse']:.10g}；相对降幅{100*s['relative_reduction_from_step500']:.3f}%，过程存在波动，完整结果未筛选。")
    final.extend(['', '## 2. 是否出现平台期？', ''])
    for ds,s in summary.items():
        final.append(f"- {ds}：1500→2000降幅{100*s['decision']['relative_decrease']:.3f}%，故{'按规则延至5000' if s['decision']['continue_to_5000'] else '按规则停于2000'}；最终{s['final_window_start']}→{s['final_global_step']}降幅{100*s['final_window_relative_decrease']:.3f}%，{'未超过5%，该窗口改善不足/平台或回退' if s['final_window_plateau_by_5percent_rule'] else '超过5%，该窗口仍在改善'}。这不证明全局收敛。")
    final.extend(['', '## 3. 是否接近原兼容精度？', ''])
    for ds,s in summary.items():
        final.append(f"- {ds}：最终逐层RMSE {min(s['final_layer_rmse']):.8g}–{max(s['final_layer_rmse']):.8g}，要求≤1e-4；{'必要条件通过，但未做完整Parity' if s['rmse_necessary_condition_pass'] else '必要条件未通过'}。")
    final.extend(['', '## 4. 是否值得第二轮完整蒸馏？', '',recommendation,'',
        '本轮只从旧500步学生权重继续，新AdamW状态与采样器重置，未声称精确恢复旧优化器。未访问官方Validation/Test、未更改教师或Token ID、未重训CMERC、未进行文本干扰。模型与完整训练状态已隔离保存，原500步模型及诊断产物保持不变；停止并等待研究者安排。', '',
        '详细数字与运行记录见[TRAINING_EXTENSION_REPORT.md](TRAINING_EXTENSION_REPORT.md)，最终交付核验见verification.json。', ''])
    (D/'FINAL_EXTENSION_DECISION.md').write_text('\n'.join(final))


if __name__=='__main__':
    main()
