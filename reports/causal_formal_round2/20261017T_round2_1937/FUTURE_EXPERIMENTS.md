# 后续正式实验矩阵（全部 planned，未启动）

先完成 MELD 单种子 1937 闭环，再决定配对训练种子 1937、2025、3407 的扩展。
训练 seed 与 split seed 分开；IEMOCAP 固定 split seed=1475，MELD 使用官方 train/dev ID，1475 仅记录。
已有 IEMOCAP/1937 成果只能校验哈希后复用；新运行使用新的 FUTURE_OUTPUT_ROOT，禁止覆盖历史。

每次启动前重新核对服务器归属、GPU UUID、健康状态、空闲显存和其他进程，显式设置 GPU_UUID；禁止沿用旧物理编号。
在已有可用环境指定 CMERC_PYTHON。每个数据集/seed 先冻结代码、真实特征 SHA256、训练/验证 ID 哈希和 DATA_ENVIRONMENT_*。
TEACHER_BEST_*、BASELINE_RUN_*、CMERC_BEST_*、DIFFICULTY_* 均必须绑定对应真实产物；无默认旧难度文件。

执行顺序：Baseline → 完整预算/验证选模教师严格验收 → 真实 difficulty → artifact 严格验收 → Full → 完整训练与配对验收 → validation 评估/比较 → 冻结分析及 best/数据/协议哈希 → 每个 best 一次 test。
FUTURE_MATRIX.json 的 full_admission_required 是测试前强制验收条件。round2_verify_run.py 已提供 --dataset、--seed、--split-seed 与 --epochs 参数，下面给出各组真实接口的完整验收命令。
B 阶段 strict_difficulty_admission 输出的 difficulty_sha256 必须固定到 DIFFICULTY_SHA_*；训练前检查该哈希，Full 验收也显式绑定该 B 阶段哈希。不得从 C 的 checkpoint 反推 B 的预期值。
validation_comparison 完成后执行 validation_freeze_create，VALIDATION_FREEZE_* 为新冻结文件路径，输出的 validation_freeze_sha256 固定到 VALIDATION_FREEZE_SHA_*。每次 test 前检查冻结 JSON、比较内容、验证输入文件、两 best 的 SHA256 及 dataset/train seed/split。已有 test 输出目录安全拒绝重跑。禁止根据 test 结果改变 epoch、seed 或参数。

所有训练命令直接调用 train_cm.py，并显式延后测试、启用逐批非有限值保护和首批梯度诊断。difficulty/evaluator 使用各自实际 CLI，不传不支持的训练参数。
w/o CL、w/o HSCL、w/o CC 分别只关闭指定组件；其余系数与预算固定。same_step_same_mode 属于独立辅助策略实验、独立 run 身份，可复用同 seed、同 backbone、validation 来源的 baseline 与 prefix_mean_v1 difficulty。

统一报告 W-F1、ECE、五路及总 CEL；prefix_mean_v1 是可逆前缀平滑视图，不是完全模态移除。单种子只报告观察差值，不宣称统计显著性。
多种子按相同训练 seed 配对，固定 split seed、数据和 replacement policy，报告逐 seed 差值、配对均值/标准差和预定配对统计；禁止挑最好 seed。
按预定相对位置 (utterance_index+0.5)/dialogue_length 的三个等宽区间报告前/中/后段分类与校准。

## IEMOCAP / training seed 1937

### baseline_training

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u train_cm.py --Dataset IEMOCAP --base-model GRU --graph_type hyper --graph_construct direct --multi_modal --mm_fusion_mthd concat_DHT --modals avl --norm LN --num_L 3 --num_K 4 --batch-size 16 --lr 0.0001 --l2 0.00003 --dropout 0.5 --rec-dropout 0.1 --seed 1937 --split-seed 1475 --causal --replacement-policy prefix_mean_v1 --epochs 80 --selection-protocol validation --defer-final-test --fail-on-nonfinite --diagnostic-batches 1 --num-workers 0 --auxiliary-policy legacy_epoch_snapshot --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed1937/baseline
```

### strict_teacher_admission

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u scripts/round2_verify_run.py --dataset IEMOCAP --seed 1937 --split-seed 1475 --epochs 80 --expected-provenance "${DATA_ENVIRONMENT_IEMOCAP_1937:?bind_verified_DATA_ENVIRONMENT_IEMOCAP_1937}" --variant baseline --run-dir "${BASELINE_RUN_IEMOCAP_1937:?bind_verified_BASELINE_RUN_IEMOCAP_1937}" --output "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed1937/verification/baseline.json
```

### difficulty_generation

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u caldiff/cal_diff.py --Dataset IEMOCAP --base-model GRU --graph_type hyper --graph_construct direct --multi_modal --mm_fusion_mthd concat_DHT --modals avl --norm LN --num_L 3 --num_K 4 --batch-size 16 --lr 0.0001 --l2 0.00003 --dropout 0.5 --rec-dropout 0.1 --seed 1937 --split-seed 1475 --causal --replacement-policy prefix_mean_v1 --ckpt_path "${TEACHER_BEST_IEMOCAP_1937:?bind_verified_TEACHER_BEST_IEMOCAP_1937}" --difficulty-output "${DIFFICULTY_IEMOCAP_1937:?bind_verified_DIFFICULTY_IEMOCAP_1937}"
```

### strict_difficulty_admission

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import json,sys,torch
from pathlib import Path
from causal_provenance import architecture_from_args,resolve_causal_difficulty,sha256_file
dataset,seed,path,teacher_path,inventory_path,baseline_run=sys.argv[1:]
teacher=torch.load(teacher_path,map_location='"'"'cpu'"'"')
if sha256_file(teacher_path)!=sha256_file(Path(baseline_run)/'"'"'best.pt'"'"'): raise ValueError('"'"'Teacher is not the admitted paired baseline best'"'"')
expected=json.loads(Path(inventory_path).read_text())['"'"'data_provenance'"'"']
values,artifact=resolve_causal_difficulty(path,dataset,expected,architecture_from_args(teacher['"'"'args'"'"']),expected_teacher_seed=int(seed))
source=artifact['"'"'manifest'"'"']['"'"'source_checkpoint'"'"']
if source['"'"'sha256'"'"']!=sha256_file(teacher_path) or source['"'"'epoch'"'"']!=source['"'"'best_epoch'"'"']: raise ValueError('"'"'Difficulty teacher is not the admitted baseline best'"'"')
print(json.dumps({'"'"'status'"'"':'"'"'passed'"'"','"'"'difficulty_sha256'"'"':artifact['"'"'sha256'"'"'],'"'"'teacher_sha256'"'"':source['"'"'sha256'"'"'],'"'"'train_count'"'"':len(values)}))' IEMOCAP 1937 "${DIFFICULTY_IEMOCAP_1937:?bind_verified_DIFFICULTY_IEMOCAP_1937}" "${TEACHER_BEST_IEMOCAP_1937:?bind_verified_TEACHER_BEST_IEMOCAP_1937}" "${DATA_ENVIRONMENT_IEMOCAP_1937:?bind_verified_DATA_ENVIRONMENT_IEMOCAP_1937}" "${BASELINE_RUN_IEMOCAP_1937:?bind_verified_BASELINE_RUN_IEMOCAP_1937}"
```

### full_training

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import sys;from causal_provenance import sha256_file;actual=sha256_file(sys.argv[1]);
if actual!=sys.argv[2]: raise ValueError("Stage-B difficulty SHA256 changed before training")' "${DIFFICULTY_IEMOCAP_1937:?bind_verified_DIFFICULTY_IEMOCAP_1937}" "${DIFFICULTY_SHA_IEMOCAP_1937:?bind_verified_DIFFICULTY_SHA_IEMOCAP_1937}" && CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u train_cm.py --Dataset IEMOCAP --base-model GRU --graph_type hyper --graph_construct direct --multi_modal --mm_fusion_mthd concat_DHT --modals avl --norm LN --num_L 3 --num_K 4 --batch-size 16 --lr 0.0001 --l2 0.00003 --dropout 0.5 --rec-dropout 0.1 --seed 1937 --split-seed 1475 --causal --replacement-policy prefix_mean_v1 --epochs 80 --selection-protocol validation --defer-final-test --fail-on-nonfinite --diagnostic-batches 1 --num-workers 0 --auxiliary-policy legacy_epoch_snapshot --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed1937/full --calibrate --rank_coff 0.005 --contrastlearning --mscl_coff 0.05 --cscl_coff 0.05 --courselearning --scheduler_steps 1 --epoch_ratio 0.15 --difficulty-path "${DIFFICULTY_IEMOCAP_1937:?bind_verified_DIFFICULTY_IEMOCAP_1937}"
```

### completed_full_and_pair_admission

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u scripts/round2_verify_run.py --dataset IEMOCAP --seed 1937 --split-seed 1475 --epochs 80 --expected-provenance "${DATA_ENVIRONMENT_IEMOCAP_1937:?bind_verified_DATA_ENVIRONMENT_IEMOCAP_1937}" --variant cmerc --run-dir "${CMERC_RUN_IEMOCAP_1937:?bind_verified_CMERC_RUN_IEMOCAP_1937}" --paired-run "${BASELINE_RUN_IEMOCAP_1937:?bind_verified_BASELINE_RUN_IEMOCAP_1937}" --difficulty "${DIFFICULTY_IEMOCAP_1937:?bind_verified_DIFFICULTY_IEMOCAP_1937}" --expected-difficulty-sha256 "${DIFFICULTY_SHA_IEMOCAP_1937:?bind_verified_DIFFICULTY_SHA_IEMOCAP_1937}" --output "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed1937/verification/full.json
```

### validation_evaluation_baseline

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u scripts/evaluate_checkpoint.py --dataset IEMOCAP --causal --checkpoint "${TEACHER_BEST_IEMOCAP_1937:?bind_verified_TEACHER_BEST_IEMOCAP_1937}" --device cuda:0 --split validation --split-seed 1475 --batch-size 16 --num-workers 0 --n-bins 15 --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed1937/evaluation/validation/baseline
```

### validation_evaluation_full

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u scripts/evaluate_checkpoint.py --dataset IEMOCAP --causal --checkpoint "${CMERC_BEST_IEMOCAP_1937:?bind_verified_CMERC_BEST_IEMOCAP_1937}" --device cuda:0 --split validation --split-seed 1475 --batch-size 16 --num-workers 0 --n-bins 15 --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed1937/evaluation/validation/full
```

### validation_comparison

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u scripts/round2_analysis.py --baseline-eval "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed1937/evaluation/validation/baseline --cmerc-eval "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed1937/evaluation/validation/full --baseline-training "${BASELINE_RUN_IEMOCAP_1937:?bind_verified_BASELINE_RUN_IEMOCAP_1937}"/metrics.jsonl --cmerc-training "${CMERC_RUN_IEMOCAP_1937:?bind_verified_CMERC_RUN_IEMOCAP_1937}"/metrics.jsonl --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed1937/comparison_validation
```

### validation_freeze_create

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import datetime,json,sys,torch
from pathlib import Path
from causal_provenance import sha256_file
mode,dataset,seed,comparison_path,baseline_path,full_path,baseline_eval,full_eval,freeze_path,*extra=sys.argv[1:]
seed=int(seed)
def check(condition,message):
    if not condition: raise ValueError(message)
comparison=json.loads(Path(comparison_path).read_text())
identity=dict(dataset=dataset,split='"'"'validation'"'"',split_seed=1475,replacement_policy='"'"'prefix_mean_v1'"'"')
check(all(comparison.get(key)==value for key,value in identity.items()),'"'"'Validation comparison identity mismatch'"'"')
best_hashes={}
for name,path,directory in [('"'"'baseline'"'"',baseline_path,baseline_eval),('"'"'cmerc'"'"',full_path,full_eval)]:
    checkpoint=torch.load(path,map_location='"'"'cpu'"'"'); args=checkpoint['"'"'args'"'"']
    check(args.get('"'"'Dataset'"'"')==dataset and args.get('"'"'seed'"'"')==seed and args.get('"'"'split_seed'"'"')==1475,'"'"'Frozen dataset/training seed mismatch'"'"')
    check(args.get('"'"'causal'"'"') is True and args.get('"'"'selection_protocol'"'"')=='"'"'validation'"'"' and checkpoint['"'"'epoch'"'"']==checkpoint['"'"'best_epoch'"'"'],'"'"'Frozen checkpoint is not validation-selected causal best'"'"')
    metadata=json.loads((Path(directory)/'"'"'metrics.json'"'"').read_text())
    check(all(metadata.get(key)==value for key,value in identity.items()),'"'"'Validation evaluator identity mismatch'"'"')
    check(Path(metadata['"'"'checkpoint'"'"']).resolve()==Path(path).resolve(),'"'"'Validation evaluated a different checkpoint'"'"')
    for filename in ('"'"'metrics.json'"'"','"'"'predictions.npz'"'"'):
        check(comparison['"'"'input_sha256'"'"'][name][filename]==sha256_file(Path(directory)/filename),'"'"'Validation comparison input SHA256 changed'"'"')
    best_hashes[name]=sha256_file(path)
comparison_sha=sha256_file(comparison_path)
if mode=='"'"'create'"'"':
    frozen={**comparison,'"'"'training_seed'"'"':seed,'"'"'validation_comparison_sha256'"'"':comparison_sha,'"'"'best_sha256'"'"':best_hashes,'"'"'frozen_utc'"'"':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with Path(freeze_path).open('"'"'x'"'"') as handle: json.dump(frozen,handle,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps({'"'"'status'"'"':'"'"'frozen'"'"','"'"'validation_freeze_sha256'"'"':sha256_file(freeze_path),'"'"'training_seed'"'"':seed,'"'"'best_sha256'"'"':best_hashes}))
elif mode=='"'"'check'"'"':
    check(len(extra)==1 and sha256_file(freeze_path)==extra[0],'"'"'Frozen validation JSON SHA256 changed'"'"')
    frozen=json.loads(Path(freeze_path).read_text())
    check(all(frozen.get(key)==value for key,value in identity.items()) and frozen.get('"'"'training_seed'"'"')==seed,'"'"'Frozen validation identity mismatch'"'"')
    check(frozen.get('"'"'validation_comparison_sha256'"'"')==comparison_sha and frozen.get('"'"'best_sha256'"'"')==best_hashes,'"'"'Frozen comparison or selected best SHA256 changed'"'"')
    check(frozen.get('"'"'input_sha256'"'"')==comparison.get('"'"'input_sha256'"'"'),'"'"'Frozen validation input identities changed'"'"')
    print(json.dumps({'"'"'status'"'"':'"'"'passed'"'"','"'"'validation_freeze_sha256'"'"':extra[0]}))
else: raise ValueError('"'"'Unknown freeze operation'"'"')' create IEMOCAP 1937 "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed1937/comparison_validation/comparison.json "${TEACHER_BEST_IEMOCAP_1937:?bind_verified_TEACHER_BEST_IEMOCAP_1937}" "${CMERC_BEST_IEMOCAP_1937:?bind_verified_CMERC_BEST_IEMOCAP_1937}" "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed1937/evaluation/validation/baseline "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed1937/evaluation/validation/full "${VALIDATION_FREEZE_IEMOCAP_1937:?bind_verified_VALIDATION_FREEZE_IEMOCAP_1937}"
```

### validation_freeze_check

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import datetime,json,sys,torch
from pathlib import Path
from causal_provenance import sha256_file
mode,dataset,seed,comparison_path,baseline_path,full_path,baseline_eval,full_eval,freeze_path,*extra=sys.argv[1:]
seed=int(seed)
def check(condition,message):
    if not condition: raise ValueError(message)
comparison=json.loads(Path(comparison_path).read_text())
identity=dict(dataset=dataset,split='"'"'validation'"'"',split_seed=1475,replacement_policy='"'"'prefix_mean_v1'"'"')
check(all(comparison.get(key)==value for key,value in identity.items()),'"'"'Validation comparison identity mismatch'"'"')
best_hashes={}
for name,path,directory in [('"'"'baseline'"'"',baseline_path,baseline_eval),('"'"'cmerc'"'"',full_path,full_eval)]:
    checkpoint=torch.load(path,map_location='"'"'cpu'"'"'); args=checkpoint['"'"'args'"'"']
    check(args.get('"'"'Dataset'"'"')==dataset and args.get('"'"'seed'"'"')==seed and args.get('"'"'split_seed'"'"')==1475,'"'"'Frozen dataset/training seed mismatch'"'"')
    check(args.get('"'"'causal'"'"') is True and args.get('"'"'selection_protocol'"'"')=='"'"'validation'"'"' and checkpoint['"'"'epoch'"'"']==checkpoint['"'"'best_epoch'"'"'],'"'"'Frozen checkpoint is not validation-selected causal best'"'"')
    metadata=json.loads((Path(directory)/'"'"'metrics.json'"'"').read_text())
    check(all(metadata.get(key)==value for key,value in identity.items()),'"'"'Validation evaluator identity mismatch'"'"')
    check(Path(metadata['"'"'checkpoint'"'"']).resolve()==Path(path).resolve(),'"'"'Validation evaluated a different checkpoint'"'"')
    for filename in ('"'"'metrics.json'"'"','"'"'predictions.npz'"'"'):
        check(comparison['"'"'input_sha256'"'"'][name][filename]==sha256_file(Path(directory)/filename),'"'"'Validation comparison input SHA256 changed'"'"')
    best_hashes[name]=sha256_file(path)
comparison_sha=sha256_file(comparison_path)
if mode=='"'"'create'"'"':
    frozen={**comparison,'"'"'training_seed'"'"':seed,'"'"'validation_comparison_sha256'"'"':comparison_sha,'"'"'best_sha256'"'"':best_hashes,'"'"'frozen_utc'"'"':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with Path(freeze_path).open('"'"'x'"'"') as handle: json.dump(frozen,handle,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps({'"'"'status'"'"':'"'"'frozen'"'"','"'"'validation_freeze_sha256'"'"':sha256_file(freeze_path),'"'"'training_seed'"'"':seed,'"'"'best_sha256'"'"':best_hashes}))
elif mode=='"'"'check'"'"':
    check(len(extra)==1 and sha256_file(freeze_path)==extra[0],'"'"'Frozen validation JSON SHA256 changed'"'"')
    frozen=json.loads(Path(freeze_path).read_text())
    check(all(frozen.get(key)==value for key,value in identity.items()) and frozen.get('"'"'training_seed'"'"')==seed,'"'"'Frozen validation identity mismatch'"'"')
    check(frozen.get('"'"'validation_comparison_sha256'"'"')==comparison_sha and frozen.get('"'"'best_sha256'"'"')==best_hashes,'"'"'Frozen comparison or selected best SHA256 changed'"'"')
    check(frozen.get('"'"'input_sha256'"'"')==comparison.get('"'"'input_sha256'"'"'),'"'"'Frozen validation input identities changed'"'"')
    print(json.dumps({'"'"'status'"'"':'"'"'passed'"'"','"'"'validation_freeze_sha256'"'"':extra[0]}))
else: raise ValueError('"'"'Unknown freeze operation'"'"')' check IEMOCAP 1937 "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed1937/comparison_validation/comparison.json "${TEACHER_BEST_IEMOCAP_1937:?bind_verified_TEACHER_BEST_IEMOCAP_1937}" "${CMERC_BEST_IEMOCAP_1937:?bind_verified_CMERC_BEST_IEMOCAP_1937}" "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed1937/evaluation/validation/baseline "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed1937/evaluation/validation/full "${VALIDATION_FREEZE_IEMOCAP_1937:?bind_verified_VALIDATION_FREEZE_IEMOCAP_1937}" "${VALIDATION_FREEZE_SHA_IEMOCAP_1937:?bind_verified_VALIDATION_FREEZE_SHA_IEMOCAP_1937}"
```

### test_evaluation_baseline

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import datetime,json,sys,torch
from pathlib import Path
from causal_provenance import sha256_file
mode,dataset,seed,comparison_path,baseline_path,full_path,baseline_eval,full_eval,freeze_path,*extra=sys.argv[1:]
seed=int(seed)
def check(condition,message):
    if not condition: raise ValueError(message)
comparison=json.loads(Path(comparison_path).read_text())
identity=dict(dataset=dataset,split='"'"'validation'"'"',split_seed=1475,replacement_policy='"'"'prefix_mean_v1'"'"')
check(all(comparison.get(key)==value for key,value in identity.items()),'"'"'Validation comparison identity mismatch'"'"')
best_hashes={}
for name,path,directory in [('"'"'baseline'"'"',baseline_path,baseline_eval),('"'"'cmerc'"'"',full_path,full_eval)]:
    checkpoint=torch.load(path,map_location='"'"'cpu'"'"'); args=checkpoint['"'"'args'"'"']
    check(args.get('"'"'Dataset'"'"')==dataset and args.get('"'"'seed'"'"')==seed and args.get('"'"'split_seed'"'"')==1475,'"'"'Frozen dataset/training seed mismatch'"'"')
    check(args.get('"'"'causal'"'"') is True and args.get('"'"'selection_protocol'"'"')=='"'"'validation'"'"' and checkpoint['"'"'epoch'"'"']==checkpoint['"'"'best_epoch'"'"'],'"'"'Frozen checkpoint is not validation-selected causal best'"'"')
    metadata=json.loads((Path(directory)/'"'"'metrics.json'"'"').read_text())
    check(all(metadata.get(key)==value for key,value in identity.items()),'"'"'Validation evaluator identity mismatch'"'"')
    check(Path(metadata['"'"'checkpoint'"'"']).resolve()==Path(path).resolve(),'"'"'Validation evaluated a different checkpoint'"'"')
    for filename in ('"'"'metrics.json'"'"','"'"'predictions.npz'"'"'):
        check(comparison['"'"'input_sha256'"'"'][name][filename]==sha256_file(Path(directory)/filename),'"'"'Validation comparison input SHA256 changed'"'"')
    best_hashes[name]=sha256_file(path)
comparison_sha=sha256_file(comparison_path)
if mode=='"'"'create'"'"':
    frozen={**comparison,'"'"'training_seed'"'"':seed,'"'"'validation_comparison_sha256'"'"':comparison_sha,'"'"'best_sha256'"'"':best_hashes,'"'"'frozen_utc'"'"':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with Path(freeze_path).open('"'"'x'"'"') as handle: json.dump(frozen,handle,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps({'"'"'status'"'"':'"'"'frozen'"'"','"'"'validation_freeze_sha256'"'"':sha256_file(freeze_path),'"'"'training_seed'"'"':seed,'"'"'best_sha256'"'"':best_hashes}))
elif mode=='"'"'check'"'"':
    check(len(extra)==1 and sha256_file(freeze_path)==extra[0],'"'"'Frozen validation JSON SHA256 changed'"'"')
    frozen=json.loads(Path(freeze_path).read_text())
    check(all(frozen.get(key)==value for key,value in identity.items()) and frozen.get('"'"'training_seed'"'"')==seed,'"'"'Frozen validation identity mismatch'"'"')
    check(frozen.get('"'"'validation_comparison_sha256'"'"')==comparison_sha and frozen.get('"'"'best_sha256'"'"')==best_hashes,'"'"'Frozen comparison or selected best SHA256 changed'"'"')
    check(frozen.get('"'"'input_sha256'"'"')==comparison.get('"'"'input_sha256'"'"'),'"'"'Frozen validation input identities changed'"'"')
    print(json.dumps({'"'"'status'"'"':'"'"'passed'"'"','"'"'validation_freeze_sha256'"'"':extra[0]}))
else: raise ValueError('"'"'Unknown freeze operation'"'"')' check IEMOCAP 1937 "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed1937/comparison_validation/comparison.json "${TEACHER_BEST_IEMOCAP_1937:?bind_verified_TEACHER_BEST_IEMOCAP_1937}" "${CMERC_BEST_IEMOCAP_1937:?bind_verified_CMERC_BEST_IEMOCAP_1937}" "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed1937/evaluation/validation/baseline "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed1937/evaluation/validation/full "${VALIDATION_FREEZE_IEMOCAP_1937:?bind_verified_VALIDATION_FREEZE_IEMOCAP_1937}" "${VALIDATION_FREEZE_SHA_IEMOCAP_1937:?bind_verified_VALIDATION_FREEZE_SHA_IEMOCAP_1937}" && test ! -e "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed1937/evaluation/test/baseline && CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u scripts/evaluate_checkpoint.py --dataset IEMOCAP --causal --checkpoint "${TEACHER_BEST_IEMOCAP_1937:?bind_verified_TEACHER_BEST_IEMOCAP_1937}" --device cuda:0 --split test --split-seed 1475 --batch-size 16 --num-workers 0 --n-bins 15 --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed1937/evaluation/test/baseline
```

### test_evaluation_full

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import datetime,json,sys,torch
from pathlib import Path
from causal_provenance import sha256_file
mode,dataset,seed,comparison_path,baseline_path,full_path,baseline_eval,full_eval,freeze_path,*extra=sys.argv[1:]
seed=int(seed)
def check(condition,message):
    if not condition: raise ValueError(message)
comparison=json.loads(Path(comparison_path).read_text())
identity=dict(dataset=dataset,split='"'"'validation'"'"',split_seed=1475,replacement_policy='"'"'prefix_mean_v1'"'"')
check(all(comparison.get(key)==value for key,value in identity.items()),'"'"'Validation comparison identity mismatch'"'"')
best_hashes={}
for name,path,directory in [('"'"'baseline'"'"',baseline_path,baseline_eval),('"'"'cmerc'"'"',full_path,full_eval)]:
    checkpoint=torch.load(path,map_location='"'"'cpu'"'"'); args=checkpoint['"'"'args'"'"']
    check(args.get('"'"'Dataset'"'"')==dataset and args.get('"'"'seed'"'"')==seed and args.get('"'"'split_seed'"'"')==1475,'"'"'Frozen dataset/training seed mismatch'"'"')
    check(args.get('"'"'causal'"'"') is True and args.get('"'"'selection_protocol'"'"')=='"'"'validation'"'"' and checkpoint['"'"'epoch'"'"']==checkpoint['"'"'best_epoch'"'"'],'"'"'Frozen checkpoint is not validation-selected causal best'"'"')
    metadata=json.loads((Path(directory)/'"'"'metrics.json'"'"').read_text())
    check(all(metadata.get(key)==value for key,value in identity.items()),'"'"'Validation evaluator identity mismatch'"'"')
    check(Path(metadata['"'"'checkpoint'"'"']).resolve()==Path(path).resolve(),'"'"'Validation evaluated a different checkpoint'"'"')
    for filename in ('"'"'metrics.json'"'"','"'"'predictions.npz'"'"'):
        check(comparison['"'"'input_sha256'"'"'][name][filename]==sha256_file(Path(directory)/filename),'"'"'Validation comparison input SHA256 changed'"'"')
    best_hashes[name]=sha256_file(path)
comparison_sha=sha256_file(comparison_path)
if mode=='"'"'create'"'"':
    frozen={**comparison,'"'"'training_seed'"'"':seed,'"'"'validation_comparison_sha256'"'"':comparison_sha,'"'"'best_sha256'"'"':best_hashes,'"'"'frozen_utc'"'"':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with Path(freeze_path).open('"'"'x'"'"') as handle: json.dump(frozen,handle,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps({'"'"'status'"'"':'"'"'frozen'"'"','"'"'validation_freeze_sha256'"'"':sha256_file(freeze_path),'"'"'training_seed'"'"':seed,'"'"'best_sha256'"'"':best_hashes}))
elif mode=='"'"'check'"'"':
    check(len(extra)==1 and sha256_file(freeze_path)==extra[0],'"'"'Frozen validation JSON SHA256 changed'"'"')
    frozen=json.loads(Path(freeze_path).read_text())
    check(all(frozen.get(key)==value for key,value in identity.items()) and frozen.get('"'"'training_seed'"'"')==seed,'"'"'Frozen validation identity mismatch'"'"')
    check(frozen.get('"'"'validation_comparison_sha256'"'"')==comparison_sha and frozen.get('"'"'best_sha256'"'"')==best_hashes,'"'"'Frozen comparison or selected best SHA256 changed'"'"')
    check(frozen.get('"'"'input_sha256'"'"')==comparison.get('"'"'input_sha256'"'"'),'"'"'Frozen validation input identities changed'"'"')
    print(json.dumps({'"'"'status'"'"':'"'"'passed'"'"','"'"'validation_freeze_sha256'"'"':extra[0]}))
else: raise ValueError('"'"'Unknown freeze operation'"'"')' check IEMOCAP 1937 "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed1937/comparison_validation/comparison.json "${TEACHER_BEST_IEMOCAP_1937:?bind_verified_TEACHER_BEST_IEMOCAP_1937}" "${CMERC_BEST_IEMOCAP_1937:?bind_verified_CMERC_BEST_IEMOCAP_1937}" "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed1937/evaluation/validation/baseline "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed1937/evaluation/validation/full "${VALIDATION_FREEZE_IEMOCAP_1937:?bind_verified_VALIDATION_FREEZE_IEMOCAP_1937}" "${VALIDATION_FREEZE_SHA_IEMOCAP_1937:?bind_verified_VALIDATION_FREEZE_SHA_IEMOCAP_1937}" && test ! -e "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed1937/evaluation/test/full && CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u scripts/evaluate_checkpoint.py --dataset IEMOCAP --causal --checkpoint "${CMERC_BEST_IEMOCAP_1937:?bind_verified_CMERC_BEST_IEMOCAP_1937}" --device cuda:0 --split test --split-seed 1475 --batch-size 16 --num-workers 0 --n-bins 15 --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed1937/evaluation/test/full
```

### test_comparison

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import datetime,json,sys,torch
from pathlib import Path
from causal_provenance import sha256_file
mode,dataset,seed,comparison_path,baseline_path,full_path,baseline_eval,full_eval,freeze_path,*extra=sys.argv[1:]
seed=int(seed)
def check(condition,message):
    if not condition: raise ValueError(message)
comparison=json.loads(Path(comparison_path).read_text())
identity=dict(dataset=dataset,split='"'"'validation'"'"',split_seed=1475,replacement_policy='"'"'prefix_mean_v1'"'"')
check(all(comparison.get(key)==value for key,value in identity.items()),'"'"'Validation comparison identity mismatch'"'"')
best_hashes={}
for name,path,directory in [('"'"'baseline'"'"',baseline_path,baseline_eval),('"'"'cmerc'"'"',full_path,full_eval)]:
    checkpoint=torch.load(path,map_location='"'"'cpu'"'"'); args=checkpoint['"'"'args'"'"']
    check(args.get('"'"'Dataset'"'"')==dataset and args.get('"'"'seed'"'"')==seed and args.get('"'"'split_seed'"'"')==1475,'"'"'Frozen dataset/training seed mismatch'"'"')
    check(args.get('"'"'causal'"'"') is True and args.get('"'"'selection_protocol'"'"')=='"'"'validation'"'"' and checkpoint['"'"'epoch'"'"']==checkpoint['"'"'best_epoch'"'"'],'"'"'Frozen checkpoint is not validation-selected causal best'"'"')
    metadata=json.loads((Path(directory)/'"'"'metrics.json'"'"').read_text())
    check(all(metadata.get(key)==value for key,value in identity.items()),'"'"'Validation evaluator identity mismatch'"'"')
    check(Path(metadata['"'"'checkpoint'"'"']).resolve()==Path(path).resolve(),'"'"'Validation evaluated a different checkpoint'"'"')
    for filename in ('"'"'metrics.json'"'"','"'"'predictions.npz'"'"'):
        check(comparison['"'"'input_sha256'"'"'][name][filename]==sha256_file(Path(directory)/filename),'"'"'Validation comparison input SHA256 changed'"'"')
    best_hashes[name]=sha256_file(path)
comparison_sha=sha256_file(comparison_path)
if mode=='"'"'create'"'"':
    frozen={**comparison,'"'"'training_seed'"'"':seed,'"'"'validation_comparison_sha256'"'"':comparison_sha,'"'"'best_sha256'"'"':best_hashes,'"'"'frozen_utc'"'"':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with Path(freeze_path).open('"'"'x'"'"') as handle: json.dump(frozen,handle,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps({'"'"'status'"'"':'"'"'frozen'"'"','"'"'validation_freeze_sha256'"'"':sha256_file(freeze_path),'"'"'training_seed'"'"':seed,'"'"'best_sha256'"'"':best_hashes}))
elif mode=='"'"'check'"'"':
    check(len(extra)==1 and sha256_file(freeze_path)==extra[0],'"'"'Frozen validation JSON SHA256 changed'"'"')
    frozen=json.loads(Path(freeze_path).read_text())
    check(all(frozen.get(key)==value for key,value in identity.items()) and frozen.get('"'"'training_seed'"'"')==seed,'"'"'Frozen validation identity mismatch'"'"')
    check(frozen.get('"'"'validation_comparison_sha256'"'"')==comparison_sha and frozen.get('"'"'best_sha256'"'"')==best_hashes,'"'"'Frozen comparison or selected best SHA256 changed'"'"')
    check(frozen.get('"'"'input_sha256'"'"')==comparison.get('"'"'input_sha256'"'"'),'"'"'Frozen validation input identities changed'"'"')
    print(json.dumps({'"'"'status'"'"':'"'"'passed'"'"','"'"'validation_freeze_sha256'"'"':extra[0]}))
else: raise ValueError('"'"'Unknown freeze operation'"'"')' check IEMOCAP 1937 "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed1937/comparison_validation/comparison.json "${TEACHER_BEST_IEMOCAP_1937:?bind_verified_TEACHER_BEST_IEMOCAP_1937}" "${CMERC_BEST_IEMOCAP_1937:?bind_verified_CMERC_BEST_IEMOCAP_1937}" "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed1937/evaluation/validation/baseline "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed1937/evaluation/validation/full "${VALIDATION_FREEZE_IEMOCAP_1937:?bind_verified_VALIDATION_FREEZE_IEMOCAP_1937}" "${VALIDATION_FREEZE_SHA_IEMOCAP_1937:?bind_verified_VALIDATION_FREEZE_SHA_IEMOCAP_1937}" && "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u scripts/round2_analysis.py --baseline-eval "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed1937/evaluation/test/baseline --cmerc-eval "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed1937/evaluation/test/full --baseline-training "${BASELINE_RUN_IEMOCAP_1937:?bind_verified_BASELINE_RUN_IEMOCAP_1937}"/metrics.jsonl --cmerc-training "${CMERC_RUN_IEMOCAP_1937:?bind_verified_CMERC_RUN_IEMOCAP_1937}"/metrics.jsonl --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed1937/comparison_test --validation-freeze "${VALIDATION_FREEZE_IEMOCAP_1937:?bind_verified_VALIDATION_FREEZE_IEMOCAP_1937}"
```

### extension_without_cl

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u train_cm.py --Dataset IEMOCAP --base-model GRU --graph_type hyper --graph_construct direct --multi_modal --mm_fusion_mthd concat_DHT --modals avl --norm LN --num_L 3 --num_K 4 --batch-size 16 --lr 0.0001 --l2 0.00003 --dropout 0.5 --rec-dropout 0.1 --seed 1937 --split-seed 1475 --causal --replacement-policy prefix_mean_v1 --epochs 80 --selection-protocol validation --defer-final-test --fail-on-nonfinite --diagnostic-batches 1 --num-workers 0 --auxiliary-policy legacy_epoch_snapshot --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed1937/without_cl --calibrate --rank_coff 0.005 --contrastlearning --mscl_coff 0.05 --cscl_coff 0.05
```

### extension_without_hscl

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import sys;from causal_provenance import sha256_file;actual=sha256_file(sys.argv[1]);
if actual!=sys.argv[2]: raise ValueError("Stage-B difficulty SHA256 changed before training")' "${DIFFICULTY_IEMOCAP_1937:?bind_verified_DIFFICULTY_IEMOCAP_1937}" "${DIFFICULTY_SHA_IEMOCAP_1937:?bind_verified_DIFFICULTY_SHA_IEMOCAP_1937}" && CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u train_cm.py --Dataset IEMOCAP --base-model GRU --graph_type hyper --graph_construct direct --multi_modal --mm_fusion_mthd concat_DHT --modals avl --norm LN --num_L 3 --num_K 4 --batch-size 16 --lr 0.0001 --l2 0.00003 --dropout 0.5 --rec-dropout 0.1 --seed 1937 --split-seed 1475 --causal --replacement-policy prefix_mean_v1 --epochs 80 --selection-protocol validation --defer-final-test --fail-on-nonfinite --diagnostic-batches 1 --num-workers 0 --auxiliary-policy legacy_epoch_snapshot --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed1937/without_hscl --calibrate --rank_coff 0.005 --courselearning --scheduler_steps 1 --epoch_ratio 0.15 --difficulty-path "${DIFFICULTY_IEMOCAP_1937:?bind_verified_DIFFICULTY_IEMOCAP_1937}"
```

### extension_without_cc

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import sys;from causal_provenance import sha256_file;actual=sha256_file(sys.argv[1]);
if actual!=sys.argv[2]: raise ValueError("Stage-B difficulty SHA256 changed before training")' "${DIFFICULTY_IEMOCAP_1937:?bind_verified_DIFFICULTY_IEMOCAP_1937}" "${DIFFICULTY_SHA_IEMOCAP_1937:?bind_verified_DIFFICULTY_SHA_IEMOCAP_1937}" && CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u train_cm.py --Dataset IEMOCAP --base-model GRU --graph_type hyper --graph_construct direct --multi_modal --mm_fusion_mthd concat_DHT --modals avl --norm LN --num_L 3 --num_K 4 --batch-size 16 --lr 0.0001 --l2 0.00003 --dropout 0.5 --rec-dropout 0.1 --seed 1937 --split-seed 1475 --causal --replacement-policy prefix_mean_v1 --epochs 80 --selection-protocol validation --defer-final-test --fail-on-nonfinite --diagnostic-batches 1 --num-workers 0 --auxiliary-policy legacy_epoch_snapshot --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed1937/without_cc --contrastlearning --mscl_coff 0.05 --cscl_coff 0.05 --courselearning --scheduler_steps 1 --epoch_ratio 0.15 --difficulty-path "${DIFFICULTY_IEMOCAP_1937:?bind_verified_DIFFICULTY_IEMOCAP_1937}"
```

### extension_same_step_same_mode

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import sys;from causal_provenance import sha256_file;actual=sha256_file(sys.argv[1]);
if actual!=sys.argv[2]: raise ValueError("Stage-B difficulty SHA256 changed before training")' "${DIFFICULTY_IEMOCAP_1937:?bind_verified_DIFFICULTY_IEMOCAP_1937}" "${DIFFICULTY_SHA_IEMOCAP_1937:?bind_verified_DIFFICULTY_SHA_IEMOCAP_1937}" && CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u train_cm.py --Dataset IEMOCAP --base-model GRU --graph_type hyper --graph_construct direct --multi_modal --mm_fusion_mthd concat_DHT --modals avl --norm LN --num_L 3 --num_K 4 --batch-size 16 --lr 0.0001 --l2 0.00003 --dropout 0.5 --rec-dropout 0.1 --seed 1937 --split-seed 1475 --causal --replacement-policy prefix_mean_v1 --epochs 80 --selection-protocol validation --defer-final-test --fail-on-nonfinite --diagnostic-batches 1 --num-workers 0 --auxiliary-policy same_step_same_mode --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed1937/same_step_same_mode --calibrate --rank_coff 0.005 --contrastlearning --mscl_coff 0.05 --cscl_coff 0.05 --courselearning --scheduler_steps 1 --epoch_ratio 0.15 --difficulty-path "${DIFFICULTY_IEMOCAP_1937:?bind_verified_DIFFICULTY_IEMOCAP_1937}"
```

按 ordered_stages 执行；comparison 与冻结成功后才执行 test，全部原始产物保留。

## IEMOCAP / training seed 2025

### baseline_training

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u train_cm.py --Dataset IEMOCAP --base-model GRU --graph_type hyper --graph_construct direct --multi_modal --mm_fusion_mthd concat_DHT --modals avl --norm LN --num_L 3 --num_K 4 --batch-size 16 --lr 0.0001 --l2 0.00003 --dropout 0.5 --rec-dropout 0.1 --seed 2025 --split-seed 1475 --causal --replacement-policy prefix_mean_v1 --epochs 80 --selection-protocol validation --defer-final-test --fail-on-nonfinite --diagnostic-batches 1 --num-workers 0 --auxiliary-policy legacy_epoch_snapshot --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed2025/baseline
```

### strict_teacher_admission

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u scripts/round2_verify_run.py --dataset IEMOCAP --seed 2025 --split-seed 1475 --epochs 80 --expected-provenance "${DATA_ENVIRONMENT_IEMOCAP_2025:?bind_verified_DATA_ENVIRONMENT_IEMOCAP_2025}" --variant baseline --run-dir "${BASELINE_RUN_IEMOCAP_2025:?bind_verified_BASELINE_RUN_IEMOCAP_2025}" --output "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed2025/verification/baseline.json
```

### difficulty_generation

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u caldiff/cal_diff.py --Dataset IEMOCAP --base-model GRU --graph_type hyper --graph_construct direct --multi_modal --mm_fusion_mthd concat_DHT --modals avl --norm LN --num_L 3 --num_K 4 --batch-size 16 --lr 0.0001 --l2 0.00003 --dropout 0.5 --rec-dropout 0.1 --seed 2025 --split-seed 1475 --causal --replacement-policy prefix_mean_v1 --ckpt_path "${TEACHER_BEST_IEMOCAP_2025:?bind_verified_TEACHER_BEST_IEMOCAP_2025}" --difficulty-output "${DIFFICULTY_IEMOCAP_2025:?bind_verified_DIFFICULTY_IEMOCAP_2025}"
```

### strict_difficulty_admission

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import json,sys,torch
from pathlib import Path
from causal_provenance import architecture_from_args,resolve_causal_difficulty,sha256_file
dataset,seed,path,teacher_path,inventory_path,baseline_run=sys.argv[1:]
teacher=torch.load(teacher_path,map_location='"'"'cpu'"'"')
if sha256_file(teacher_path)!=sha256_file(Path(baseline_run)/'"'"'best.pt'"'"'): raise ValueError('"'"'Teacher is not the admitted paired baseline best'"'"')
expected=json.loads(Path(inventory_path).read_text())['"'"'data_provenance'"'"']
values,artifact=resolve_causal_difficulty(path,dataset,expected,architecture_from_args(teacher['"'"'args'"'"']),expected_teacher_seed=int(seed))
source=artifact['"'"'manifest'"'"']['"'"'source_checkpoint'"'"']
if source['"'"'sha256'"'"']!=sha256_file(teacher_path) or source['"'"'epoch'"'"']!=source['"'"'best_epoch'"'"']: raise ValueError('"'"'Difficulty teacher is not the admitted baseline best'"'"')
print(json.dumps({'"'"'status'"'"':'"'"'passed'"'"','"'"'difficulty_sha256'"'"':artifact['"'"'sha256'"'"'],'"'"'teacher_sha256'"'"':source['"'"'sha256'"'"'],'"'"'train_count'"'"':len(values)}))' IEMOCAP 2025 "${DIFFICULTY_IEMOCAP_2025:?bind_verified_DIFFICULTY_IEMOCAP_2025}" "${TEACHER_BEST_IEMOCAP_2025:?bind_verified_TEACHER_BEST_IEMOCAP_2025}" "${DATA_ENVIRONMENT_IEMOCAP_2025:?bind_verified_DATA_ENVIRONMENT_IEMOCAP_2025}" "${BASELINE_RUN_IEMOCAP_2025:?bind_verified_BASELINE_RUN_IEMOCAP_2025}"
```

### full_training

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import sys;from causal_provenance import sha256_file;actual=sha256_file(sys.argv[1]);
if actual!=sys.argv[2]: raise ValueError("Stage-B difficulty SHA256 changed before training")' "${DIFFICULTY_IEMOCAP_2025:?bind_verified_DIFFICULTY_IEMOCAP_2025}" "${DIFFICULTY_SHA_IEMOCAP_2025:?bind_verified_DIFFICULTY_SHA_IEMOCAP_2025}" && CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u train_cm.py --Dataset IEMOCAP --base-model GRU --graph_type hyper --graph_construct direct --multi_modal --mm_fusion_mthd concat_DHT --modals avl --norm LN --num_L 3 --num_K 4 --batch-size 16 --lr 0.0001 --l2 0.00003 --dropout 0.5 --rec-dropout 0.1 --seed 2025 --split-seed 1475 --causal --replacement-policy prefix_mean_v1 --epochs 80 --selection-protocol validation --defer-final-test --fail-on-nonfinite --diagnostic-batches 1 --num-workers 0 --auxiliary-policy legacy_epoch_snapshot --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed2025/full --calibrate --rank_coff 0.005 --contrastlearning --mscl_coff 0.05 --cscl_coff 0.05 --courselearning --scheduler_steps 1 --epoch_ratio 0.15 --difficulty-path "${DIFFICULTY_IEMOCAP_2025:?bind_verified_DIFFICULTY_IEMOCAP_2025}"
```

### completed_full_and_pair_admission

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u scripts/round2_verify_run.py --dataset IEMOCAP --seed 2025 --split-seed 1475 --epochs 80 --expected-provenance "${DATA_ENVIRONMENT_IEMOCAP_2025:?bind_verified_DATA_ENVIRONMENT_IEMOCAP_2025}" --variant cmerc --run-dir "${CMERC_RUN_IEMOCAP_2025:?bind_verified_CMERC_RUN_IEMOCAP_2025}" --paired-run "${BASELINE_RUN_IEMOCAP_2025:?bind_verified_BASELINE_RUN_IEMOCAP_2025}" --difficulty "${DIFFICULTY_IEMOCAP_2025:?bind_verified_DIFFICULTY_IEMOCAP_2025}" --expected-difficulty-sha256 "${DIFFICULTY_SHA_IEMOCAP_2025:?bind_verified_DIFFICULTY_SHA_IEMOCAP_2025}" --output "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed2025/verification/full.json
```

### validation_evaluation_baseline

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u scripts/evaluate_checkpoint.py --dataset IEMOCAP --causal --checkpoint "${TEACHER_BEST_IEMOCAP_2025:?bind_verified_TEACHER_BEST_IEMOCAP_2025}" --device cuda:0 --split validation --split-seed 1475 --batch-size 16 --num-workers 0 --n-bins 15 --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed2025/evaluation/validation/baseline
```

### validation_evaluation_full

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u scripts/evaluate_checkpoint.py --dataset IEMOCAP --causal --checkpoint "${CMERC_BEST_IEMOCAP_2025:?bind_verified_CMERC_BEST_IEMOCAP_2025}" --device cuda:0 --split validation --split-seed 1475 --batch-size 16 --num-workers 0 --n-bins 15 --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed2025/evaluation/validation/full
```

### validation_comparison

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u scripts/round2_analysis.py --baseline-eval "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed2025/evaluation/validation/baseline --cmerc-eval "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed2025/evaluation/validation/full --baseline-training "${BASELINE_RUN_IEMOCAP_2025:?bind_verified_BASELINE_RUN_IEMOCAP_2025}"/metrics.jsonl --cmerc-training "${CMERC_RUN_IEMOCAP_2025:?bind_verified_CMERC_RUN_IEMOCAP_2025}"/metrics.jsonl --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed2025/comparison_validation
```

### validation_freeze_create

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import datetime,json,sys,torch
from pathlib import Path
from causal_provenance import sha256_file
mode,dataset,seed,comparison_path,baseline_path,full_path,baseline_eval,full_eval,freeze_path,*extra=sys.argv[1:]
seed=int(seed)
def check(condition,message):
    if not condition: raise ValueError(message)
comparison=json.loads(Path(comparison_path).read_text())
identity=dict(dataset=dataset,split='"'"'validation'"'"',split_seed=1475,replacement_policy='"'"'prefix_mean_v1'"'"')
check(all(comparison.get(key)==value for key,value in identity.items()),'"'"'Validation comparison identity mismatch'"'"')
best_hashes={}
for name,path,directory in [('"'"'baseline'"'"',baseline_path,baseline_eval),('"'"'cmerc'"'"',full_path,full_eval)]:
    checkpoint=torch.load(path,map_location='"'"'cpu'"'"'); args=checkpoint['"'"'args'"'"']
    check(args.get('"'"'Dataset'"'"')==dataset and args.get('"'"'seed'"'"')==seed and args.get('"'"'split_seed'"'"')==1475,'"'"'Frozen dataset/training seed mismatch'"'"')
    check(args.get('"'"'causal'"'"') is True and args.get('"'"'selection_protocol'"'"')=='"'"'validation'"'"' and checkpoint['"'"'epoch'"'"']==checkpoint['"'"'best_epoch'"'"'],'"'"'Frozen checkpoint is not validation-selected causal best'"'"')
    metadata=json.loads((Path(directory)/'"'"'metrics.json'"'"').read_text())
    check(all(metadata.get(key)==value for key,value in identity.items()),'"'"'Validation evaluator identity mismatch'"'"')
    check(Path(metadata['"'"'checkpoint'"'"']).resolve()==Path(path).resolve(),'"'"'Validation evaluated a different checkpoint'"'"')
    for filename in ('"'"'metrics.json'"'"','"'"'predictions.npz'"'"'):
        check(comparison['"'"'input_sha256'"'"'][name][filename]==sha256_file(Path(directory)/filename),'"'"'Validation comparison input SHA256 changed'"'"')
    best_hashes[name]=sha256_file(path)
comparison_sha=sha256_file(comparison_path)
if mode=='"'"'create'"'"':
    frozen={**comparison,'"'"'training_seed'"'"':seed,'"'"'validation_comparison_sha256'"'"':comparison_sha,'"'"'best_sha256'"'"':best_hashes,'"'"'frozen_utc'"'"':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with Path(freeze_path).open('"'"'x'"'"') as handle: json.dump(frozen,handle,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps({'"'"'status'"'"':'"'"'frozen'"'"','"'"'validation_freeze_sha256'"'"':sha256_file(freeze_path),'"'"'training_seed'"'"':seed,'"'"'best_sha256'"'"':best_hashes}))
elif mode=='"'"'check'"'"':
    check(len(extra)==1 and sha256_file(freeze_path)==extra[0],'"'"'Frozen validation JSON SHA256 changed'"'"')
    frozen=json.loads(Path(freeze_path).read_text())
    check(all(frozen.get(key)==value for key,value in identity.items()) and frozen.get('"'"'training_seed'"'"')==seed,'"'"'Frozen validation identity mismatch'"'"')
    check(frozen.get('"'"'validation_comparison_sha256'"'"')==comparison_sha and frozen.get('"'"'best_sha256'"'"')==best_hashes,'"'"'Frozen comparison or selected best SHA256 changed'"'"')
    check(frozen.get('"'"'input_sha256'"'"')==comparison.get('"'"'input_sha256'"'"'),'"'"'Frozen validation input identities changed'"'"')
    print(json.dumps({'"'"'status'"'"':'"'"'passed'"'"','"'"'validation_freeze_sha256'"'"':extra[0]}))
else: raise ValueError('"'"'Unknown freeze operation'"'"')' create IEMOCAP 2025 "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed2025/comparison_validation/comparison.json "${TEACHER_BEST_IEMOCAP_2025:?bind_verified_TEACHER_BEST_IEMOCAP_2025}" "${CMERC_BEST_IEMOCAP_2025:?bind_verified_CMERC_BEST_IEMOCAP_2025}" "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed2025/evaluation/validation/baseline "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed2025/evaluation/validation/full "${VALIDATION_FREEZE_IEMOCAP_2025:?bind_verified_VALIDATION_FREEZE_IEMOCAP_2025}"
```

### validation_freeze_check

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import datetime,json,sys,torch
from pathlib import Path
from causal_provenance import sha256_file
mode,dataset,seed,comparison_path,baseline_path,full_path,baseline_eval,full_eval,freeze_path,*extra=sys.argv[1:]
seed=int(seed)
def check(condition,message):
    if not condition: raise ValueError(message)
comparison=json.loads(Path(comparison_path).read_text())
identity=dict(dataset=dataset,split='"'"'validation'"'"',split_seed=1475,replacement_policy='"'"'prefix_mean_v1'"'"')
check(all(comparison.get(key)==value for key,value in identity.items()),'"'"'Validation comparison identity mismatch'"'"')
best_hashes={}
for name,path,directory in [('"'"'baseline'"'"',baseline_path,baseline_eval),('"'"'cmerc'"'"',full_path,full_eval)]:
    checkpoint=torch.load(path,map_location='"'"'cpu'"'"'); args=checkpoint['"'"'args'"'"']
    check(args.get('"'"'Dataset'"'"')==dataset and args.get('"'"'seed'"'"')==seed and args.get('"'"'split_seed'"'"')==1475,'"'"'Frozen dataset/training seed mismatch'"'"')
    check(args.get('"'"'causal'"'"') is True and args.get('"'"'selection_protocol'"'"')=='"'"'validation'"'"' and checkpoint['"'"'epoch'"'"']==checkpoint['"'"'best_epoch'"'"'],'"'"'Frozen checkpoint is not validation-selected causal best'"'"')
    metadata=json.loads((Path(directory)/'"'"'metrics.json'"'"').read_text())
    check(all(metadata.get(key)==value for key,value in identity.items()),'"'"'Validation evaluator identity mismatch'"'"')
    check(Path(metadata['"'"'checkpoint'"'"']).resolve()==Path(path).resolve(),'"'"'Validation evaluated a different checkpoint'"'"')
    for filename in ('"'"'metrics.json'"'"','"'"'predictions.npz'"'"'):
        check(comparison['"'"'input_sha256'"'"'][name][filename]==sha256_file(Path(directory)/filename),'"'"'Validation comparison input SHA256 changed'"'"')
    best_hashes[name]=sha256_file(path)
comparison_sha=sha256_file(comparison_path)
if mode=='"'"'create'"'"':
    frozen={**comparison,'"'"'training_seed'"'"':seed,'"'"'validation_comparison_sha256'"'"':comparison_sha,'"'"'best_sha256'"'"':best_hashes,'"'"'frozen_utc'"'"':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with Path(freeze_path).open('"'"'x'"'"') as handle: json.dump(frozen,handle,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps({'"'"'status'"'"':'"'"'frozen'"'"','"'"'validation_freeze_sha256'"'"':sha256_file(freeze_path),'"'"'training_seed'"'"':seed,'"'"'best_sha256'"'"':best_hashes}))
elif mode=='"'"'check'"'"':
    check(len(extra)==1 and sha256_file(freeze_path)==extra[0],'"'"'Frozen validation JSON SHA256 changed'"'"')
    frozen=json.loads(Path(freeze_path).read_text())
    check(all(frozen.get(key)==value for key,value in identity.items()) and frozen.get('"'"'training_seed'"'"')==seed,'"'"'Frozen validation identity mismatch'"'"')
    check(frozen.get('"'"'validation_comparison_sha256'"'"')==comparison_sha and frozen.get('"'"'best_sha256'"'"')==best_hashes,'"'"'Frozen comparison or selected best SHA256 changed'"'"')
    check(frozen.get('"'"'input_sha256'"'"')==comparison.get('"'"'input_sha256'"'"'),'"'"'Frozen validation input identities changed'"'"')
    print(json.dumps({'"'"'status'"'"':'"'"'passed'"'"','"'"'validation_freeze_sha256'"'"':extra[0]}))
else: raise ValueError('"'"'Unknown freeze operation'"'"')' check IEMOCAP 2025 "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed2025/comparison_validation/comparison.json "${TEACHER_BEST_IEMOCAP_2025:?bind_verified_TEACHER_BEST_IEMOCAP_2025}" "${CMERC_BEST_IEMOCAP_2025:?bind_verified_CMERC_BEST_IEMOCAP_2025}" "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed2025/evaluation/validation/baseline "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed2025/evaluation/validation/full "${VALIDATION_FREEZE_IEMOCAP_2025:?bind_verified_VALIDATION_FREEZE_IEMOCAP_2025}" "${VALIDATION_FREEZE_SHA_IEMOCAP_2025:?bind_verified_VALIDATION_FREEZE_SHA_IEMOCAP_2025}"
```

### test_evaluation_baseline

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import datetime,json,sys,torch
from pathlib import Path
from causal_provenance import sha256_file
mode,dataset,seed,comparison_path,baseline_path,full_path,baseline_eval,full_eval,freeze_path,*extra=sys.argv[1:]
seed=int(seed)
def check(condition,message):
    if not condition: raise ValueError(message)
comparison=json.loads(Path(comparison_path).read_text())
identity=dict(dataset=dataset,split='"'"'validation'"'"',split_seed=1475,replacement_policy='"'"'prefix_mean_v1'"'"')
check(all(comparison.get(key)==value for key,value in identity.items()),'"'"'Validation comparison identity mismatch'"'"')
best_hashes={}
for name,path,directory in [('"'"'baseline'"'"',baseline_path,baseline_eval),('"'"'cmerc'"'"',full_path,full_eval)]:
    checkpoint=torch.load(path,map_location='"'"'cpu'"'"'); args=checkpoint['"'"'args'"'"']
    check(args.get('"'"'Dataset'"'"')==dataset and args.get('"'"'seed'"'"')==seed and args.get('"'"'split_seed'"'"')==1475,'"'"'Frozen dataset/training seed mismatch'"'"')
    check(args.get('"'"'causal'"'"') is True and args.get('"'"'selection_protocol'"'"')=='"'"'validation'"'"' and checkpoint['"'"'epoch'"'"']==checkpoint['"'"'best_epoch'"'"'],'"'"'Frozen checkpoint is not validation-selected causal best'"'"')
    metadata=json.loads((Path(directory)/'"'"'metrics.json'"'"').read_text())
    check(all(metadata.get(key)==value for key,value in identity.items()),'"'"'Validation evaluator identity mismatch'"'"')
    check(Path(metadata['"'"'checkpoint'"'"']).resolve()==Path(path).resolve(),'"'"'Validation evaluated a different checkpoint'"'"')
    for filename in ('"'"'metrics.json'"'"','"'"'predictions.npz'"'"'):
        check(comparison['"'"'input_sha256'"'"'][name][filename]==sha256_file(Path(directory)/filename),'"'"'Validation comparison input SHA256 changed'"'"')
    best_hashes[name]=sha256_file(path)
comparison_sha=sha256_file(comparison_path)
if mode=='"'"'create'"'"':
    frozen={**comparison,'"'"'training_seed'"'"':seed,'"'"'validation_comparison_sha256'"'"':comparison_sha,'"'"'best_sha256'"'"':best_hashes,'"'"'frozen_utc'"'"':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with Path(freeze_path).open('"'"'x'"'"') as handle: json.dump(frozen,handle,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps({'"'"'status'"'"':'"'"'frozen'"'"','"'"'validation_freeze_sha256'"'"':sha256_file(freeze_path),'"'"'training_seed'"'"':seed,'"'"'best_sha256'"'"':best_hashes}))
elif mode=='"'"'check'"'"':
    check(len(extra)==1 and sha256_file(freeze_path)==extra[0],'"'"'Frozen validation JSON SHA256 changed'"'"')
    frozen=json.loads(Path(freeze_path).read_text())
    check(all(frozen.get(key)==value for key,value in identity.items()) and frozen.get('"'"'training_seed'"'"')==seed,'"'"'Frozen validation identity mismatch'"'"')
    check(frozen.get('"'"'validation_comparison_sha256'"'"')==comparison_sha and frozen.get('"'"'best_sha256'"'"')==best_hashes,'"'"'Frozen comparison or selected best SHA256 changed'"'"')
    check(frozen.get('"'"'input_sha256'"'"')==comparison.get('"'"'input_sha256'"'"'),'"'"'Frozen validation input identities changed'"'"')
    print(json.dumps({'"'"'status'"'"':'"'"'passed'"'"','"'"'validation_freeze_sha256'"'"':extra[0]}))
else: raise ValueError('"'"'Unknown freeze operation'"'"')' check IEMOCAP 2025 "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed2025/comparison_validation/comparison.json "${TEACHER_BEST_IEMOCAP_2025:?bind_verified_TEACHER_BEST_IEMOCAP_2025}" "${CMERC_BEST_IEMOCAP_2025:?bind_verified_CMERC_BEST_IEMOCAP_2025}" "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed2025/evaluation/validation/baseline "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed2025/evaluation/validation/full "${VALIDATION_FREEZE_IEMOCAP_2025:?bind_verified_VALIDATION_FREEZE_IEMOCAP_2025}" "${VALIDATION_FREEZE_SHA_IEMOCAP_2025:?bind_verified_VALIDATION_FREEZE_SHA_IEMOCAP_2025}" && test ! -e "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed2025/evaluation/test/baseline && CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u scripts/evaluate_checkpoint.py --dataset IEMOCAP --causal --checkpoint "${TEACHER_BEST_IEMOCAP_2025:?bind_verified_TEACHER_BEST_IEMOCAP_2025}" --device cuda:0 --split test --split-seed 1475 --batch-size 16 --num-workers 0 --n-bins 15 --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed2025/evaluation/test/baseline
```

### test_evaluation_full

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import datetime,json,sys,torch
from pathlib import Path
from causal_provenance import sha256_file
mode,dataset,seed,comparison_path,baseline_path,full_path,baseline_eval,full_eval,freeze_path,*extra=sys.argv[1:]
seed=int(seed)
def check(condition,message):
    if not condition: raise ValueError(message)
comparison=json.loads(Path(comparison_path).read_text())
identity=dict(dataset=dataset,split='"'"'validation'"'"',split_seed=1475,replacement_policy='"'"'prefix_mean_v1'"'"')
check(all(comparison.get(key)==value for key,value in identity.items()),'"'"'Validation comparison identity mismatch'"'"')
best_hashes={}
for name,path,directory in [('"'"'baseline'"'"',baseline_path,baseline_eval),('"'"'cmerc'"'"',full_path,full_eval)]:
    checkpoint=torch.load(path,map_location='"'"'cpu'"'"'); args=checkpoint['"'"'args'"'"']
    check(args.get('"'"'Dataset'"'"')==dataset and args.get('"'"'seed'"'"')==seed and args.get('"'"'split_seed'"'"')==1475,'"'"'Frozen dataset/training seed mismatch'"'"')
    check(args.get('"'"'causal'"'"') is True and args.get('"'"'selection_protocol'"'"')=='"'"'validation'"'"' and checkpoint['"'"'epoch'"'"']==checkpoint['"'"'best_epoch'"'"'],'"'"'Frozen checkpoint is not validation-selected causal best'"'"')
    metadata=json.loads((Path(directory)/'"'"'metrics.json'"'"').read_text())
    check(all(metadata.get(key)==value for key,value in identity.items()),'"'"'Validation evaluator identity mismatch'"'"')
    check(Path(metadata['"'"'checkpoint'"'"']).resolve()==Path(path).resolve(),'"'"'Validation evaluated a different checkpoint'"'"')
    for filename in ('"'"'metrics.json'"'"','"'"'predictions.npz'"'"'):
        check(comparison['"'"'input_sha256'"'"'][name][filename]==sha256_file(Path(directory)/filename),'"'"'Validation comparison input SHA256 changed'"'"')
    best_hashes[name]=sha256_file(path)
comparison_sha=sha256_file(comparison_path)
if mode=='"'"'create'"'"':
    frozen={**comparison,'"'"'training_seed'"'"':seed,'"'"'validation_comparison_sha256'"'"':comparison_sha,'"'"'best_sha256'"'"':best_hashes,'"'"'frozen_utc'"'"':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with Path(freeze_path).open('"'"'x'"'"') as handle: json.dump(frozen,handle,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps({'"'"'status'"'"':'"'"'frozen'"'"','"'"'validation_freeze_sha256'"'"':sha256_file(freeze_path),'"'"'training_seed'"'"':seed,'"'"'best_sha256'"'"':best_hashes}))
elif mode=='"'"'check'"'"':
    check(len(extra)==1 and sha256_file(freeze_path)==extra[0],'"'"'Frozen validation JSON SHA256 changed'"'"')
    frozen=json.loads(Path(freeze_path).read_text())
    check(all(frozen.get(key)==value for key,value in identity.items()) and frozen.get('"'"'training_seed'"'"')==seed,'"'"'Frozen validation identity mismatch'"'"')
    check(frozen.get('"'"'validation_comparison_sha256'"'"')==comparison_sha and frozen.get('"'"'best_sha256'"'"')==best_hashes,'"'"'Frozen comparison or selected best SHA256 changed'"'"')
    check(frozen.get('"'"'input_sha256'"'"')==comparison.get('"'"'input_sha256'"'"'),'"'"'Frozen validation input identities changed'"'"')
    print(json.dumps({'"'"'status'"'"':'"'"'passed'"'"','"'"'validation_freeze_sha256'"'"':extra[0]}))
else: raise ValueError('"'"'Unknown freeze operation'"'"')' check IEMOCAP 2025 "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed2025/comparison_validation/comparison.json "${TEACHER_BEST_IEMOCAP_2025:?bind_verified_TEACHER_BEST_IEMOCAP_2025}" "${CMERC_BEST_IEMOCAP_2025:?bind_verified_CMERC_BEST_IEMOCAP_2025}" "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed2025/evaluation/validation/baseline "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed2025/evaluation/validation/full "${VALIDATION_FREEZE_IEMOCAP_2025:?bind_verified_VALIDATION_FREEZE_IEMOCAP_2025}" "${VALIDATION_FREEZE_SHA_IEMOCAP_2025:?bind_verified_VALIDATION_FREEZE_SHA_IEMOCAP_2025}" && test ! -e "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed2025/evaluation/test/full && CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u scripts/evaluate_checkpoint.py --dataset IEMOCAP --causal --checkpoint "${CMERC_BEST_IEMOCAP_2025:?bind_verified_CMERC_BEST_IEMOCAP_2025}" --device cuda:0 --split test --split-seed 1475 --batch-size 16 --num-workers 0 --n-bins 15 --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed2025/evaluation/test/full
```

### test_comparison

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import datetime,json,sys,torch
from pathlib import Path
from causal_provenance import sha256_file
mode,dataset,seed,comparison_path,baseline_path,full_path,baseline_eval,full_eval,freeze_path,*extra=sys.argv[1:]
seed=int(seed)
def check(condition,message):
    if not condition: raise ValueError(message)
comparison=json.loads(Path(comparison_path).read_text())
identity=dict(dataset=dataset,split='"'"'validation'"'"',split_seed=1475,replacement_policy='"'"'prefix_mean_v1'"'"')
check(all(comparison.get(key)==value for key,value in identity.items()),'"'"'Validation comparison identity mismatch'"'"')
best_hashes={}
for name,path,directory in [('"'"'baseline'"'"',baseline_path,baseline_eval),('"'"'cmerc'"'"',full_path,full_eval)]:
    checkpoint=torch.load(path,map_location='"'"'cpu'"'"'); args=checkpoint['"'"'args'"'"']
    check(args.get('"'"'Dataset'"'"')==dataset and args.get('"'"'seed'"'"')==seed and args.get('"'"'split_seed'"'"')==1475,'"'"'Frozen dataset/training seed mismatch'"'"')
    check(args.get('"'"'causal'"'"') is True and args.get('"'"'selection_protocol'"'"')=='"'"'validation'"'"' and checkpoint['"'"'epoch'"'"']==checkpoint['"'"'best_epoch'"'"'],'"'"'Frozen checkpoint is not validation-selected causal best'"'"')
    metadata=json.loads((Path(directory)/'"'"'metrics.json'"'"').read_text())
    check(all(metadata.get(key)==value for key,value in identity.items()),'"'"'Validation evaluator identity mismatch'"'"')
    check(Path(metadata['"'"'checkpoint'"'"']).resolve()==Path(path).resolve(),'"'"'Validation evaluated a different checkpoint'"'"')
    for filename in ('"'"'metrics.json'"'"','"'"'predictions.npz'"'"'):
        check(comparison['"'"'input_sha256'"'"'][name][filename]==sha256_file(Path(directory)/filename),'"'"'Validation comparison input SHA256 changed'"'"')
    best_hashes[name]=sha256_file(path)
comparison_sha=sha256_file(comparison_path)
if mode=='"'"'create'"'"':
    frozen={**comparison,'"'"'training_seed'"'"':seed,'"'"'validation_comparison_sha256'"'"':comparison_sha,'"'"'best_sha256'"'"':best_hashes,'"'"'frozen_utc'"'"':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with Path(freeze_path).open('"'"'x'"'"') as handle: json.dump(frozen,handle,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps({'"'"'status'"'"':'"'"'frozen'"'"','"'"'validation_freeze_sha256'"'"':sha256_file(freeze_path),'"'"'training_seed'"'"':seed,'"'"'best_sha256'"'"':best_hashes}))
elif mode=='"'"'check'"'"':
    check(len(extra)==1 and sha256_file(freeze_path)==extra[0],'"'"'Frozen validation JSON SHA256 changed'"'"')
    frozen=json.loads(Path(freeze_path).read_text())
    check(all(frozen.get(key)==value for key,value in identity.items()) and frozen.get('"'"'training_seed'"'"')==seed,'"'"'Frozen validation identity mismatch'"'"')
    check(frozen.get('"'"'validation_comparison_sha256'"'"')==comparison_sha and frozen.get('"'"'best_sha256'"'"')==best_hashes,'"'"'Frozen comparison or selected best SHA256 changed'"'"')
    check(frozen.get('"'"'input_sha256'"'"')==comparison.get('"'"'input_sha256'"'"'),'"'"'Frozen validation input identities changed'"'"')
    print(json.dumps({'"'"'status'"'"':'"'"'passed'"'"','"'"'validation_freeze_sha256'"'"':extra[0]}))
else: raise ValueError('"'"'Unknown freeze operation'"'"')' check IEMOCAP 2025 "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed2025/comparison_validation/comparison.json "${TEACHER_BEST_IEMOCAP_2025:?bind_verified_TEACHER_BEST_IEMOCAP_2025}" "${CMERC_BEST_IEMOCAP_2025:?bind_verified_CMERC_BEST_IEMOCAP_2025}" "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed2025/evaluation/validation/baseline "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed2025/evaluation/validation/full "${VALIDATION_FREEZE_IEMOCAP_2025:?bind_verified_VALIDATION_FREEZE_IEMOCAP_2025}" "${VALIDATION_FREEZE_SHA_IEMOCAP_2025:?bind_verified_VALIDATION_FREEZE_SHA_IEMOCAP_2025}" && "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u scripts/round2_analysis.py --baseline-eval "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed2025/evaluation/test/baseline --cmerc-eval "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed2025/evaluation/test/full --baseline-training "${BASELINE_RUN_IEMOCAP_2025:?bind_verified_BASELINE_RUN_IEMOCAP_2025}"/metrics.jsonl --cmerc-training "${CMERC_RUN_IEMOCAP_2025:?bind_verified_CMERC_RUN_IEMOCAP_2025}"/metrics.jsonl --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed2025/comparison_test --validation-freeze "${VALIDATION_FREEZE_IEMOCAP_2025:?bind_verified_VALIDATION_FREEZE_IEMOCAP_2025}"
```

### extension_without_cl

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u train_cm.py --Dataset IEMOCAP --base-model GRU --graph_type hyper --graph_construct direct --multi_modal --mm_fusion_mthd concat_DHT --modals avl --norm LN --num_L 3 --num_K 4 --batch-size 16 --lr 0.0001 --l2 0.00003 --dropout 0.5 --rec-dropout 0.1 --seed 2025 --split-seed 1475 --causal --replacement-policy prefix_mean_v1 --epochs 80 --selection-protocol validation --defer-final-test --fail-on-nonfinite --diagnostic-batches 1 --num-workers 0 --auxiliary-policy legacy_epoch_snapshot --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed2025/without_cl --calibrate --rank_coff 0.005 --contrastlearning --mscl_coff 0.05 --cscl_coff 0.05
```

### extension_without_hscl

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import sys;from causal_provenance import sha256_file;actual=sha256_file(sys.argv[1]);
if actual!=sys.argv[2]: raise ValueError("Stage-B difficulty SHA256 changed before training")' "${DIFFICULTY_IEMOCAP_2025:?bind_verified_DIFFICULTY_IEMOCAP_2025}" "${DIFFICULTY_SHA_IEMOCAP_2025:?bind_verified_DIFFICULTY_SHA_IEMOCAP_2025}" && CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u train_cm.py --Dataset IEMOCAP --base-model GRU --graph_type hyper --graph_construct direct --multi_modal --mm_fusion_mthd concat_DHT --modals avl --norm LN --num_L 3 --num_K 4 --batch-size 16 --lr 0.0001 --l2 0.00003 --dropout 0.5 --rec-dropout 0.1 --seed 2025 --split-seed 1475 --causal --replacement-policy prefix_mean_v1 --epochs 80 --selection-protocol validation --defer-final-test --fail-on-nonfinite --diagnostic-batches 1 --num-workers 0 --auxiliary-policy legacy_epoch_snapshot --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed2025/without_hscl --calibrate --rank_coff 0.005 --courselearning --scheduler_steps 1 --epoch_ratio 0.15 --difficulty-path "${DIFFICULTY_IEMOCAP_2025:?bind_verified_DIFFICULTY_IEMOCAP_2025}"
```

### extension_without_cc

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import sys;from causal_provenance import sha256_file;actual=sha256_file(sys.argv[1]);
if actual!=sys.argv[2]: raise ValueError("Stage-B difficulty SHA256 changed before training")' "${DIFFICULTY_IEMOCAP_2025:?bind_verified_DIFFICULTY_IEMOCAP_2025}" "${DIFFICULTY_SHA_IEMOCAP_2025:?bind_verified_DIFFICULTY_SHA_IEMOCAP_2025}" && CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u train_cm.py --Dataset IEMOCAP --base-model GRU --graph_type hyper --graph_construct direct --multi_modal --mm_fusion_mthd concat_DHT --modals avl --norm LN --num_L 3 --num_K 4 --batch-size 16 --lr 0.0001 --l2 0.00003 --dropout 0.5 --rec-dropout 0.1 --seed 2025 --split-seed 1475 --causal --replacement-policy prefix_mean_v1 --epochs 80 --selection-protocol validation --defer-final-test --fail-on-nonfinite --diagnostic-batches 1 --num-workers 0 --auxiliary-policy legacy_epoch_snapshot --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed2025/without_cc --contrastlearning --mscl_coff 0.05 --cscl_coff 0.05 --courselearning --scheduler_steps 1 --epoch_ratio 0.15 --difficulty-path "${DIFFICULTY_IEMOCAP_2025:?bind_verified_DIFFICULTY_IEMOCAP_2025}"
```

### extension_same_step_same_mode

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import sys;from causal_provenance import sha256_file;actual=sha256_file(sys.argv[1]);
if actual!=sys.argv[2]: raise ValueError("Stage-B difficulty SHA256 changed before training")' "${DIFFICULTY_IEMOCAP_2025:?bind_verified_DIFFICULTY_IEMOCAP_2025}" "${DIFFICULTY_SHA_IEMOCAP_2025:?bind_verified_DIFFICULTY_SHA_IEMOCAP_2025}" && CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u train_cm.py --Dataset IEMOCAP --base-model GRU --graph_type hyper --graph_construct direct --multi_modal --mm_fusion_mthd concat_DHT --modals avl --norm LN --num_L 3 --num_K 4 --batch-size 16 --lr 0.0001 --l2 0.00003 --dropout 0.5 --rec-dropout 0.1 --seed 2025 --split-seed 1475 --causal --replacement-policy prefix_mean_v1 --epochs 80 --selection-protocol validation --defer-final-test --fail-on-nonfinite --diagnostic-batches 1 --num-workers 0 --auxiliary-policy same_step_same_mode --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed2025/same_step_same_mode --calibrate --rank_coff 0.005 --contrastlearning --mscl_coff 0.05 --cscl_coff 0.05 --courselearning --scheduler_steps 1 --epoch_ratio 0.15 --difficulty-path "${DIFFICULTY_IEMOCAP_2025:?bind_verified_DIFFICULTY_IEMOCAP_2025}"
```

按 ordered_stages 执行；comparison 与冻结成功后才执行 test，全部原始产物保留。

## IEMOCAP / training seed 3407

### baseline_training

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u train_cm.py --Dataset IEMOCAP --base-model GRU --graph_type hyper --graph_construct direct --multi_modal --mm_fusion_mthd concat_DHT --modals avl --norm LN --num_L 3 --num_K 4 --batch-size 16 --lr 0.0001 --l2 0.00003 --dropout 0.5 --rec-dropout 0.1 --seed 3407 --split-seed 1475 --causal --replacement-policy prefix_mean_v1 --epochs 80 --selection-protocol validation --defer-final-test --fail-on-nonfinite --diagnostic-batches 1 --num-workers 0 --auxiliary-policy legacy_epoch_snapshot --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed3407/baseline
```

### strict_teacher_admission

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u scripts/round2_verify_run.py --dataset IEMOCAP --seed 3407 --split-seed 1475 --epochs 80 --expected-provenance "${DATA_ENVIRONMENT_IEMOCAP_3407:?bind_verified_DATA_ENVIRONMENT_IEMOCAP_3407}" --variant baseline --run-dir "${BASELINE_RUN_IEMOCAP_3407:?bind_verified_BASELINE_RUN_IEMOCAP_3407}" --output "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed3407/verification/baseline.json
```

### difficulty_generation

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u caldiff/cal_diff.py --Dataset IEMOCAP --base-model GRU --graph_type hyper --graph_construct direct --multi_modal --mm_fusion_mthd concat_DHT --modals avl --norm LN --num_L 3 --num_K 4 --batch-size 16 --lr 0.0001 --l2 0.00003 --dropout 0.5 --rec-dropout 0.1 --seed 3407 --split-seed 1475 --causal --replacement-policy prefix_mean_v1 --ckpt_path "${TEACHER_BEST_IEMOCAP_3407:?bind_verified_TEACHER_BEST_IEMOCAP_3407}" --difficulty-output "${DIFFICULTY_IEMOCAP_3407:?bind_verified_DIFFICULTY_IEMOCAP_3407}"
```

### strict_difficulty_admission

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import json,sys,torch
from pathlib import Path
from causal_provenance import architecture_from_args,resolve_causal_difficulty,sha256_file
dataset,seed,path,teacher_path,inventory_path,baseline_run=sys.argv[1:]
teacher=torch.load(teacher_path,map_location='"'"'cpu'"'"')
if sha256_file(teacher_path)!=sha256_file(Path(baseline_run)/'"'"'best.pt'"'"'): raise ValueError('"'"'Teacher is not the admitted paired baseline best'"'"')
expected=json.loads(Path(inventory_path).read_text())['"'"'data_provenance'"'"']
values,artifact=resolve_causal_difficulty(path,dataset,expected,architecture_from_args(teacher['"'"'args'"'"']),expected_teacher_seed=int(seed))
source=artifact['"'"'manifest'"'"']['"'"'source_checkpoint'"'"']
if source['"'"'sha256'"'"']!=sha256_file(teacher_path) or source['"'"'epoch'"'"']!=source['"'"'best_epoch'"'"']: raise ValueError('"'"'Difficulty teacher is not the admitted baseline best'"'"')
print(json.dumps({'"'"'status'"'"':'"'"'passed'"'"','"'"'difficulty_sha256'"'"':artifact['"'"'sha256'"'"'],'"'"'teacher_sha256'"'"':source['"'"'sha256'"'"'],'"'"'train_count'"'"':len(values)}))' IEMOCAP 3407 "${DIFFICULTY_IEMOCAP_3407:?bind_verified_DIFFICULTY_IEMOCAP_3407}" "${TEACHER_BEST_IEMOCAP_3407:?bind_verified_TEACHER_BEST_IEMOCAP_3407}" "${DATA_ENVIRONMENT_IEMOCAP_3407:?bind_verified_DATA_ENVIRONMENT_IEMOCAP_3407}" "${BASELINE_RUN_IEMOCAP_3407:?bind_verified_BASELINE_RUN_IEMOCAP_3407}"
```

### full_training

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import sys;from causal_provenance import sha256_file;actual=sha256_file(sys.argv[1]);
if actual!=sys.argv[2]: raise ValueError("Stage-B difficulty SHA256 changed before training")' "${DIFFICULTY_IEMOCAP_3407:?bind_verified_DIFFICULTY_IEMOCAP_3407}" "${DIFFICULTY_SHA_IEMOCAP_3407:?bind_verified_DIFFICULTY_SHA_IEMOCAP_3407}" && CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u train_cm.py --Dataset IEMOCAP --base-model GRU --graph_type hyper --graph_construct direct --multi_modal --mm_fusion_mthd concat_DHT --modals avl --norm LN --num_L 3 --num_K 4 --batch-size 16 --lr 0.0001 --l2 0.00003 --dropout 0.5 --rec-dropout 0.1 --seed 3407 --split-seed 1475 --causal --replacement-policy prefix_mean_v1 --epochs 80 --selection-protocol validation --defer-final-test --fail-on-nonfinite --diagnostic-batches 1 --num-workers 0 --auxiliary-policy legacy_epoch_snapshot --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed3407/full --calibrate --rank_coff 0.005 --contrastlearning --mscl_coff 0.05 --cscl_coff 0.05 --courselearning --scheduler_steps 1 --epoch_ratio 0.15 --difficulty-path "${DIFFICULTY_IEMOCAP_3407:?bind_verified_DIFFICULTY_IEMOCAP_3407}"
```

### completed_full_and_pair_admission

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u scripts/round2_verify_run.py --dataset IEMOCAP --seed 3407 --split-seed 1475 --epochs 80 --expected-provenance "${DATA_ENVIRONMENT_IEMOCAP_3407:?bind_verified_DATA_ENVIRONMENT_IEMOCAP_3407}" --variant cmerc --run-dir "${CMERC_RUN_IEMOCAP_3407:?bind_verified_CMERC_RUN_IEMOCAP_3407}" --paired-run "${BASELINE_RUN_IEMOCAP_3407:?bind_verified_BASELINE_RUN_IEMOCAP_3407}" --difficulty "${DIFFICULTY_IEMOCAP_3407:?bind_verified_DIFFICULTY_IEMOCAP_3407}" --expected-difficulty-sha256 "${DIFFICULTY_SHA_IEMOCAP_3407:?bind_verified_DIFFICULTY_SHA_IEMOCAP_3407}" --output "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed3407/verification/full.json
```

### validation_evaluation_baseline

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u scripts/evaluate_checkpoint.py --dataset IEMOCAP --causal --checkpoint "${TEACHER_BEST_IEMOCAP_3407:?bind_verified_TEACHER_BEST_IEMOCAP_3407}" --device cuda:0 --split validation --split-seed 1475 --batch-size 16 --num-workers 0 --n-bins 15 --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed3407/evaluation/validation/baseline
```

### validation_evaluation_full

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u scripts/evaluate_checkpoint.py --dataset IEMOCAP --causal --checkpoint "${CMERC_BEST_IEMOCAP_3407:?bind_verified_CMERC_BEST_IEMOCAP_3407}" --device cuda:0 --split validation --split-seed 1475 --batch-size 16 --num-workers 0 --n-bins 15 --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed3407/evaluation/validation/full
```

### validation_comparison

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u scripts/round2_analysis.py --baseline-eval "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed3407/evaluation/validation/baseline --cmerc-eval "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed3407/evaluation/validation/full --baseline-training "${BASELINE_RUN_IEMOCAP_3407:?bind_verified_BASELINE_RUN_IEMOCAP_3407}"/metrics.jsonl --cmerc-training "${CMERC_RUN_IEMOCAP_3407:?bind_verified_CMERC_RUN_IEMOCAP_3407}"/metrics.jsonl --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed3407/comparison_validation
```

### validation_freeze_create

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import datetime,json,sys,torch
from pathlib import Path
from causal_provenance import sha256_file
mode,dataset,seed,comparison_path,baseline_path,full_path,baseline_eval,full_eval,freeze_path,*extra=sys.argv[1:]
seed=int(seed)
def check(condition,message):
    if not condition: raise ValueError(message)
comparison=json.loads(Path(comparison_path).read_text())
identity=dict(dataset=dataset,split='"'"'validation'"'"',split_seed=1475,replacement_policy='"'"'prefix_mean_v1'"'"')
check(all(comparison.get(key)==value for key,value in identity.items()),'"'"'Validation comparison identity mismatch'"'"')
best_hashes={}
for name,path,directory in [('"'"'baseline'"'"',baseline_path,baseline_eval),('"'"'cmerc'"'"',full_path,full_eval)]:
    checkpoint=torch.load(path,map_location='"'"'cpu'"'"'); args=checkpoint['"'"'args'"'"']
    check(args.get('"'"'Dataset'"'"')==dataset and args.get('"'"'seed'"'"')==seed and args.get('"'"'split_seed'"'"')==1475,'"'"'Frozen dataset/training seed mismatch'"'"')
    check(args.get('"'"'causal'"'"') is True and args.get('"'"'selection_protocol'"'"')=='"'"'validation'"'"' and checkpoint['"'"'epoch'"'"']==checkpoint['"'"'best_epoch'"'"'],'"'"'Frozen checkpoint is not validation-selected causal best'"'"')
    metadata=json.loads((Path(directory)/'"'"'metrics.json'"'"').read_text())
    check(all(metadata.get(key)==value for key,value in identity.items()),'"'"'Validation evaluator identity mismatch'"'"')
    check(Path(metadata['"'"'checkpoint'"'"']).resolve()==Path(path).resolve(),'"'"'Validation evaluated a different checkpoint'"'"')
    for filename in ('"'"'metrics.json'"'"','"'"'predictions.npz'"'"'):
        check(comparison['"'"'input_sha256'"'"'][name][filename]==sha256_file(Path(directory)/filename),'"'"'Validation comparison input SHA256 changed'"'"')
    best_hashes[name]=sha256_file(path)
comparison_sha=sha256_file(comparison_path)
if mode=='"'"'create'"'"':
    frozen={**comparison,'"'"'training_seed'"'"':seed,'"'"'validation_comparison_sha256'"'"':comparison_sha,'"'"'best_sha256'"'"':best_hashes,'"'"'frozen_utc'"'"':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with Path(freeze_path).open('"'"'x'"'"') as handle: json.dump(frozen,handle,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps({'"'"'status'"'"':'"'"'frozen'"'"','"'"'validation_freeze_sha256'"'"':sha256_file(freeze_path),'"'"'training_seed'"'"':seed,'"'"'best_sha256'"'"':best_hashes}))
elif mode=='"'"'check'"'"':
    check(len(extra)==1 and sha256_file(freeze_path)==extra[0],'"'"'Frozen validation JSON SHA256 changed'"'"')
    frozen=json.loads(Path(freeze_path).read_text())
    check(all(frozen.get(key)==value for key,value in identity.items()) and frozen.get('"'"'training_seed'"'"')==seed,'"'"'Frozen validation identity mismatch'"'"')
    check(frozen.get('"'"'validation_comparison_sha256'"'"')==comparison_sha and frozen.get('"'"'best_sha256'"'"')==best_hashes,'"'"'Frozen comparison or selected best SHA256 changed'"'"')
    check(frozen.get('"'"'input_sha256'"'"')==comparison.get('"'"'input_sha256'"'"'),'"'"'Frozen validation input identities changed'"'"')
    print(json.dumps({'"'"'status'"'"':'"'"'passed'"'"','"'"'validation_freeze_sha256'"'"':extra[0]}))
else: raise ValueError('"'"'Unknown freeze operation'"'"')' create IEMOCAP 3407 "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed3407/comparison_validation/comparison.json "${TEACHER_BEST_IEMOCAP_3407:?bind_verified_TEACHER_BEST_IEMOCAP_3407}" "${CMERC_BEST_IEMOCAP_3407:?bind_verified_CMERC_BEST_IEMOCAP_3407}" "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed3407/evaluation/validation/baseline "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed3407/evaluation/validation/full "${VALIDATION_FREEZE_IEMOCAP_3407:?bind_verified_VALIDATION_FREEZE_IEMOCAP_3407}"
```

### validation_freeze_check

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import datetime,json,sys,torch
from pathlib import Path
from causal_provenance import sha256_file
mode,dataset,seed,comparison_path,baseline_path,full_path,baseline_eval,full_eval,freeze_path,*extra=sys.argv[1:]
seed=int(seed)
def check(condition,message):
    if not condition: raise ValueError(message)
comparison=json.loads(Path(comparison_path).read_text())
identity=dict(dataset=dataset,split='"'"'validation'"'"',split_seed=1475,replacement_policy='"'"'prefix_mean_v1'"'"')
check(all(comparison.get(key)==value for key,value in identity.items()),'"'"'Validation comparison identity mismatch'"'"')
best_hashes={}
for name,path,directory in [('"'"'baseline'"'"',baseline_path,baseline_eval),('"'"'cmerc'"'"',full_path,full_eval)]:
    checkpoint=torch.load(path,map_location='"'"'cpu'"'"'); args=checkpoint['"'"'args'"'"']
    check(args.get('"'"'Dataset'"'"')==dataset and args.get('"'"'seed'"'"')==seed and args.get('"'"'split_seed'"'"')==1475,'"'"'Frozen dataset/training seed mismatch'"'"')
    check(args.get('"'"'causal'"'"') is True and args.get('"'"'selection_protocol'"'"')=='"'"'validation'"'"' and checkpoint['"'"'epoch'"'"']==checkpoint['"'"'best_epoch'"'"'],'"'"'Frozen checkpoint is not validation-selected causal best'"'"')
    metadata=json.loads((Path(directory)/'"'"'metrics.json'"'"').read_text())
    check(all(metadata.get(key)==value for key,value in identity.items()),'"'"'Validation evaluator identity mismatch'"'"')
    check(Path(metadata['"'"'checkpoint'"'"']).resolve()==Path(path).resolve(),'"'"'Validation evaluated a different checkpoint'"'"')
    for filename in ('"'"'metrics.json'"'"','"'"'predictions.npz'"'"'):
        check(comparison['"'"'input_sha256'"'"'][name][filename]==sha256_file(Path(directory)/filename),'"'"'Validation comparison input SHA256 changed'"'"')
    best_hashes[name]=sha256_file(path)
comparison_sha=sha256_file(comparison_path)
if mode=='"'"'create'"'"':
    frozen={**comparison,'"'"'training_seed'"'"':seed,'"'"'validation_comparison_sha256'"'"':comparison_sha,'"'"'best_sha256'"'"':best_hashes,'"'"'frozen_utc'"'"':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with Path(freeze_path).open('"'"'x'"'"') as handle: json.dump(frozen,handle,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps({'"'"'status'"'"':'"'"'frozen'"'"','"'"'validation_freeze_sha256'"'"':sha256_file(freeze_path),'"'"'training_seed'"'"':seed,'"'"'best_sha256'"'"':best_hashes}))
elif mode=='"'"'check'"'"':
    check(len(extra)==1 and sha256_file(freeze_path)==extra[0],'"'"'Frozen validation JSON SHA256 changed'"'"')
    frozen=json.loads(Path(freeze_path).read_text())
    check(all(frozen.get(key)==value for key,value in identity.items()) and frozen.get('"'"'training_seed'"'"')==seed,'"'"'Frozen validation identity mismatch'"'"')
    check(frozen.get('"'"'validation_comparison_sha256'"'"')==comparison_sha and frozen.get('"'"'best_sha256'"'"')==best_hashes,'"'"'Frozen comparison or selected best SHA256 changed'"'"')
    check(frozen.get('"'"'input_sha256'"'"')==comparison.get('"'"'input_sha256'"'"'),'"'"'Frozen validation input identities changed'"'"')
    print(json.dumps({'"'"'status'"'"':'"'"'passed'"'"','"'"'validation_freeze_sha256'"'"':extra[0]}))
else: raise ValueError('"'"'Unknown freeze operation'"'"')' check IEMOCAP 3407 "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed3407/comparison_validation/comparison.json "${TEACHER_BEST_IEMOCAP_3407:?bind_verified_TEACHER_BEST_IEMOCAP_3407}" "${CMERC_BEST_IEMOCAP_3407:?bind_verified_CMERC_BEST_IEMOCAP_3407}" "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed3407/evaluation/validation/baseline "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed3407/evaluation/validation/full "${VALIDATION_FREEZE_IEMOCAP_3407:?bind_verified_VALIDATION_FREEZE_IEMOCAP_3407}" "${VALIDATION_FREEZE_SHA_IEMOCAP_3407:?bind_verified_VALIDATION_FREEZE_SHA_IEMOCAP_3407}"
```

### test_evaluation_baseline

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import datetime,json,sys,torch
from pathlib import Path
from causal_provenance import sha256_file
mode,dataset,seed,comparison_path,baseline_path,full_path,baseline_eval,full_eval,freeze_path,*extra=sys.argv[1:]
seed=int(seed)
def check(condition,message):
    if not condition: raise ValueError(message)
comparison=json.loads(Path(comparison_path).read_text())
identity=dict(dataset=dataset,split='"'"'validation'"'"',split_seed=1475,replacement_policy='"'"'prefix_mean_v1'"'"')
check(all(comparison.get(key)==value for key,value in identity.items()),'"'"'Validation comparison identity mismatch'"'"')
best_hashes={}
for name,path,directory in [('"'"'baseline'"'"',baseline_path,baseline_eval),('"'"'cmerc'"'"',full_path,full_eval)]:
    checkpoint=torch.load(path,map_location='"'"'cpu'"'"'); args=checkpoint['"'"'args'"'"']
    check(args.get('"'"'Dataset'"'"')==dataset and args.get('"'"'seed'"'"')==seed and args.get('"'"'split_seed'"'"')==1475,'"'"'Frozen dataset/training seed mismatch'"'"')
    check(args.get('"'"'causal'"'"') is True and args.get('"'"'selection_protocol'"'"')=='"'"'validation'"'"' and checkpoint['"'"'epoch'"'"']==checkpoint['"'"'best_epoch'"'"'],'"'"'Frozen checkpoint is not validation-selected causal best'"'"')
    metadata=json.loads((Path(directory)/'"'"'metrics.json'"'"').read_text())
    check(all(metadata.get(key)==value for key,value in identity.items()),'"'"'Validation evaluator identity mismatch'"'"')
    check(Path(metadata['"'"'checkpoint'"'"']).resolve()==Path(path).resolve(),'"'"'Validation evaluated a different checkpoint'"'"')
    for filename in ('"'"'metrics.json'"'"','"'"'predictions.npz'"'"'):
        check(comparison['"'"'input_sha256'"'"'][name][filename]==sha256_file(Path(directory)/filename),'"'"'Validation comparison input SHA256 changed'"'"')
    best_hashes[name]=sha256_file(path)
comparison_sha=sha256_file(comparison_path)
if mode=='"'"'create'"'"':
    frozen={**comparison,'"'"'training_seed'"'"':seed,'"'"'validation_comparison_sha256'"'"':comparison_sha,'"'"'best_sha256'"'"':best_hashes,'"'"'frozen_utc'"'"':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with Path(freeze_path).open('"'"'x'"'"') as handle: json.dump(frozen,handle,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps({'"'"'status'"'"':'"'"'frozen'"'"','"'"'validation_freeze_sha256'"'"':sha256_file(freeze_path),'"'"'training_seed'"'"':seed,'"'"'best_sha256'"'"':best_hashes}))
elif mode=='"'"'check'"'"':
    check(len(extra)==1 and sha256_file(freeze_path)==extra[0],'"'"'Frozen validation JSON SHA256 changed'"'"')
    frozen=json.loads(Path(freeze_path).read_text())
    check(all(frozen.get(key)==value for key,value in identity.items()) and frozen.get('"'"'training_seed'"'"')==seed,'"'"'Frozen validation identity mismatch'"'"')
    check(frozen.get('"'"'validation_comparison_sha256'"'"')==comparison_sha and frozen.get('"'"'best_sha256'"'"')==best_hashes,'"'"'Frozen comparison or selected best SHA256 changed'"'"')
    check(frozen.get('"'"'input_sha256'"'"')==comparison.get('"'"'input_sha256'"'"'),'"'"'Frozen validation input identities changed'"'"')
    print(json.dumps({'"'"'status'"'"':'"'"'passed'"'"','"'"'validation_freeze_sha256'"'"':extra[0]}))
else: raise ValueError('"'"'Unknown freeze operation'"'"')' check IEMOCAP 3407 "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed3407/comparison_validation/comparison.json "${TEACHER_BEST_IEMOCAP_3407:?bind_verified_TEACHER_BEST_IEMOCAP_3407}" "${CMERC_BEST_IEMOCAP_3407:?bind_verified_CMERC_BEST_IEMOCAP_3407}" "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed3407/evaluation/validation/baseline "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed3407/evaluation/validation/full "${VALIDATION_FREEZE_IEMOCAP_3407:?bind_verified_VALIDATION_FREEZE_IEMOCAP_3407}" "${VALIDATION_FREEZE_SHA_IEMOCAP_3407:?bind_verified_VALIDATION_FREEZE_SHA_IEMOCAP_3407}" && test ! -e "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed3407/evaluation/test/baseline && CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u scripts/evaluate_checkpoint.py --dataset IEMOCAP --causal --checkpoint "${TEACHER_BEST_IEMOCAP_3407:?bind_verified_TEACHER_BEST_IEMOCAP_3407}" --device cuda:0 --split test --split-seed 1475 --batch-size 16 --num-workers 0 --n-bins 15 --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed3407/evaluation/test/baseline
```

### test_evaluation_full

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import datetime,json,sys,torch
from pathlib import Path
from causal_provenance import sha256_file
mode,dataset,seed,comparison_path,baseline_path,full_path,baseline_eval,full_eval,freeze_path,*extra=sys.argv[1:]
seed=int(seed)
def check(condition,message):
    if not condition: raise ValueError(message)
comparison=json.loads(Path(comparison_path).read_text())
identity=dict(dataset=dataset,split='"'"'validation'"'"',split_seed=1475,replacement_policy='"'"'prefix_mean_v1'"'"')
check(all(comparison.get(key)==value for key,value in identity.items()),'"'"'Validation comparison identity mismatch'"'"')
best_hashes={}
for name,path,directory in [('"'"'baseline'"'"',baseline_path,baseline_eval),('"'"'cmerc'"'"',full_path,full_eval)]:
    checkpoint=torch.load(path,map_location='"'"'cpu'"'"'); args=checkpoint['"'"'args'"'"']
    check(args.get('"'"'Dataset'"'"')==dataset and args.get('"'"'seed'"'"')==seed and args.get('"'"'split_seed'"'"')==1475,'"'"'Frozen dataset/training seed mismatch'"'"')
    check(args.get('"'"'causal'"'"') is True and args.get('"'"'selection_protocol'"'"')=='"'"'validation'"'"' and checkpoint['"'"'epoch'"'"']==checkpoint['"'"'best_epoch'"'"'],'"'"'Frozen checkpoint is not validation-selected causal best'"'"')
    metadata=json.loads((Path(directory)/'"'"'metrics.json'"'"').read_text())
    check(all(metadata.get(key)==value for key,value in identity.items()),'"'"'Validation evaluator identity mismatch'"'"')
    check(Path(metadata['"'"'checkpoint'"'"']).resolve()==Path(path).resolve(),'"'"'Validation evaluated a different checkpoint'"'"')
    for filename in ('"'"'metrics.json'"'"','"'"'predictions.npz'"'"'):
        check(comparison['"'"'input_sha256'"'"'][name][filename]==sha256_file(Path(directory)/filename),'"'"'Validation comparison input SHA256 changed'"'"')
    best_hashes[name]=sha256_file(path)
comparison_sha=sha256_file(comparison_path)
if mode=='"'"'create'"'"':
    frozen={**comparison,'"'"'training_seed'"'"':seed,'"'"'validation_comparison_sha256'"'"':comparison_sha,'"'"'best_sha256'"'"':best_hashes,'"'"'frozen_utc'"'"':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with Path(freeze_path).open('"'"'x'"'"') as handle: json.dump(frozen,handle,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps({'"'"'status'"'"':'"'"'frozen'"'"','"'"'validation_freeze_sha256'"'"':sha256_file(freeze_path),'"'"'training_seed'"'"':seed,'"'"'best_sha256'"'"':best_hashes}))
elif mode=='"'"'check'"'"':
    check(len(extra)==1 and sha256_file(freeze_path)==extra[0],'"'"'Frozen validation JSON SHA256 changed'"'"')
    frozen=json.loads(Path(freeze_path).read_text())
    check(all(frozen.get(key)==value for key,value in identity.items()) and frozen.get('"'"'training_seed'"'"')==seed,'"'"'Frozen validation identity mismatch'"'"')
    check(frozen.get('"'"'validation_comparison_sha256'"'"')==comparison_sha and frozen.get('"'"'best_sha256'"'"')==best_hashes,'"'"'Frozen comparison or selected best SHA256 changed'"'"')
    check(frozen.get('"'"'input_sha256'"'"')==comparison.get('"'"'input_sha256'"'"'),'"'"'Frozen validation input identities changed'"'"')
    print(json.dumps({'"'"'status'"'"':'"'"'passed'"'"','"'"'validation_freeze_sha256'"'"':extra[0]}))
else: raise ValueError('"'"'Unknown freeze operation'"'"')' check IEMOCAP 3407 "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed3407/comparison_validation/comparison.json "${TEACHER_BEST_IEMOCAP_3407:?bind_verified_TEACHER_BEST_IEMOCAP_3407}" "${CMERC_BEST_IEMOCAP_3407:?bind_verified_CMERC_BEST_IEMOCAP_3407}" "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed3407/evaluation/validation/baseline "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed3407/evaluation/validation/full "${VALIDATION_FREEZE_IEMOCAP_3407:?bind_verified_VALIDATION_FREEZE_IEMOCAP_3407}" "${VALIDATION_FREEZE_SHA_IEMOCAP_3407:?bind_verified_VALIDATION_FREEZE_SHA_IEMOCAP_3407}" && test ! -e "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed3407/evaluation/test/full && CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u scripts/evaluate_checkpoint.py --dataset IEMOCAP --causal --checkpoint "${CMERC_BEST_IEMOCAP_3407:?bind_verified_CMERC_BEST_IEMOCAP_3407}" --device cuda:0 --split test --split-seed 1475 --batch-size 16 --num-workers 0 --n-bins 15 --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed3407/evaluation/test/full
```

### test_comparison

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import datetime,json,sys,torch
from pathlib import Path
from causal_provenance import sha256_file
mode,dataset,seed,comparison_path,baseline_path,full_path,baseline_eval,full_eval,freeze_path,*extra=sys.argv[1:]
seed=int(seed)
def check(condition,message):
    if not condition: raise ValueError(message)
comparison=json.loads(Path(comparison_path).read_text())
identity=dict(dataset=dataset,split='"'"'validation'"'"',split_seed=1475,replacement_policy='"'"'prefix_mean_v1'"'"')
check(all(comparison.get(key)==value for key,value in identity.items()),'"'"'Validation comparison identity mismatch'"'"')
best_hashes={}
for name,path,directory in [('"'"'baseline'"'"',baseline_path,baseline_eval),('"'"'cmerc'"'"',full_path,full_eval)]:
    checkpoint=torch.load(path,map_location='"'"'cpu'"'"'); args=checkpoint['"'"'args'"'"']
    check(args.get('"'"'Dataset'"'"')==dataset and args.get('"'"'seed'"'"')==seed and args.get('"'"'split_seed'"'"')==1475,'"'"'Frozen dataset/training seed mismatch'"'"')
    check(args.get('"'"'causal'"'"') is True and args.get('"'"'selection_protocol'"'"')=='"'"'validation'"'"' and checkpoint['"'"'epoch'"'"']==checkpoint['"'"'best_epoch'"'"'],'"'"'Frozen checkpoint is not validation-selected causal best'"'"')
    metadata=json.loads((Path(directory)/'"'"'metrics.json'"'"').read_text())
    check(all(metadata.get(key)==value for key,value in identity.items()),'"'"'Validation evaluator identity mismatch'"'"')
    check(Path(metadata['"'"'checkpoint'"'"']).resolve()==Path(path).resolve(),'"'"'Validation evaluated a different checkpoint'"'"')
    for filename in ('"'"'metrics.json'"'"','"'"'predictions.npz'"'"'):
        check(comparison['"'"'input_sha256'"'"'][name][filename]==sha256_file(Path(directory)/filename),'"'"'Validation comparison input SHA256 changed'"'"')
    best_hashes[name]=sha256_file(path)
comparison_sha=sha256_file(comparison_path)
if mode=='"'"'create'"'"':
    frozen={**comparison,'"'"'training_seed'"'"':seed,'"'"'validation_comparison_sha256'"'"':comparison_sha,'"'"'best_sha256'"'"':best_hashes,'"'"'frozen_utc'"'"':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with Path(freeze_path).open('"'"'x'"'"') as handle: json.dump(frozen,handle,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps({'"'"'status'"'"':'"'"'frozen'"'"','"'"'validation_freeze_sha256'"'"':sha256_file(freeze_path),'"'"'training_seed'"'"':seed,'"'"'best_sha256'"'"':best_hashes}))
elif mode=='"'"'check'"'"':
    check(len(extra)==1 and sha256_file(freeze_path)==extra[0],'"'"'Frozen validation JSON SHA256 changed'"'"')
    frozen=json.loads(Path(freeze_path).read_text())
    check(all(frozen.get(key)==value for key,value in identity.items()) and frozen.get('"'"'training_seed'"'"')==seed,'"'"'Frozen validation identity mismatch'"'"')
    check(frozen.get('"'"'validation_comparison_sha256'"'"')==comparison_sha and frozen.get('"'"'best_sha256'"'"')==best_hashes,'"'"'Frozen comparison or selected best SHA256 changed'"'"')
    check(frozen.get('"'"'input_sha256'"'"')==comparison.get('"'"'input_sha256'"'"'),'"'"'Frozen validation input identities changed'"'"')
    print(json.dumps({'"'"'status'"'"':'"'"'passed'"'"','"'"'validation_freeze_sha256'"'"':extra[0]}))
else: raise ValueError('"'"'Unknown freeze operation'"'"')' check IEMOCAP 3407 "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed3407/comparison_validation/comparison.json "${TEACHER_BEST_IEMOCAP_3407:?bind_verified_TEACHER_BEST_IEMOCAP_3407}" "${CMERC_BEST_IEMOCAP_3407:?bind_verified_CMERC_BEST_IEMOCAP_3407}" "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed3407/evaluation/validation/baseline "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed3407/evaluation/validation/full "${VALIDATION_FREEZE_IEMOCAP_3407:?bind_verified_VALIDATION_FREEZE_IEMOCAP_3407}" "${VALIDATION_FREEZE_SHA_IEMOCAP_3407:?bind_verified_VALIDATION_FREEZE_SHA_IEMOCAP_3407}" && "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u scripts/round2_analysis.py --baseline-eval "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed3407/evaluation/test/baseline --cmerc-eval "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed3407/evaluation/test/full --baseline-training "${BASELINE_RUN_IEMOCAP_3407:?bind_verified_BASELINE_RUN_IEMOCAP_3407}"/metrics.jsonl --cmerc-training "${CMERC_RUN_IEMOCAP_3407:?bind_verified_CMERC_RUN_IEMOCAP_3407}"/metrics.jsonl --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed3407/comparison_test --validation-freeze "${VALIDATION_FREEZE_IEMOCAP_3407:?bind_verified_VALIDATION_FREEZE_IEMOCAP_3407}"
```

### extension_without_cl

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u train_cm.py --Dataset IEMOCAP --base-model GRU --graph_type hyper --graph_construct direct --multi_modal --mm_fusion_mthd concat_DHT --modals avl --norm LN --num_L 3 --num_K 4 --batch-size 16 --lr 0.0001 --l2 0.00003 --dropout 0.5 --rec-dropout 0.1 --seed 3407 --split-seed 1475 --causal --replacement-policy prefix_mean_v1 --epochs 80 --selection-protocol validation --defer-final-test --fail-on-nonfinite --diagnostic-batches 1 --num-workers 0 --auxiliary-policy legacy_epoch_snapshot --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed3407/without_cl --calibrate --rank_coff 0.005 --contrastlearning --mscl_coff 0.05 --cscl_coff 0.05
```

### extension_without_hscl

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import sys;from causal_provenance import sha256_file;actual=sha256_file(sys.argv[1]);
if actual!=sys.argv[2]: raise ValueError("Stage-B difficulty SHA256 changed before training")' "${DIFFICULTY_IEMOCAP_3407:?bind_verified_DIFFICULTY_IEMOCAP_3407}" "${DIFFICULTY_SHA_IEMOCAP_3407:?bind_verified_DIFFICULTY_SHA_IEMOCAP_3407}" && CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u train_cm.py --Dataset IEMOCAP --base-model GRU --graph_type hyper --graph_construct direct --multi_modal --mm_fusion_mthd concat_DHT --modals avl --norm LN --num_L 3 --num_K 4 --batch-size 16 --lr 0.0001 --l2 0.00003 --dropout 0.5 --rec-dropout 0.1 --seed 3407 --split-seed 1475 --causal --replacement-policy prefix_mean_v1 --epochs 80 --selection-protocol validation --defer-final-test --fail-on-nonfinite --diagnostic-batches 1 --num-workers 0 --auxiliary-policy legacy_epoch_snapshot --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed3407/without_hscl --calibrate --rank_coff 0.005 --courselearning --scheduler_steps 1 --epoch_ratio 0.15 --difficulty-path "${DIFFICULTY_IEMOCAP_3407:?bind_verified_DIFFICULTY_IEMOCAP_3407}"
```

### extension_without_cc

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import sys;from causal_provenance import sha256_file;actual=sha256_file(sys.argv[1]);
if actual!=sys.argv[2]: raise ValueError("Stage-B difficulty SHA256 changed before training")' "${DIFFICULTY_IEMOCAP_3407:?bind_verified_DIFFICULTY_IEMOCAP_3407}" "${DIFFICULTY_SHA_IEMOCAP_3407:?bind_verified_DIFFICULTY_SHA_IEMOCAP_3407}" && CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u train_cm.py --Dataset IEMOCAP --base-model GRU --graph_type hyper --graph_construct direct --multi_modal --mm_fusion_mthd concat_DHT --modals avl --norm LN --num_L 3 --num_K 4 --batch-size 16 --lr 0.0001 --l2 0.00003 --dropout 0.5 --rec-dropout 0.1 --seed 3407 --split-seed 1475 --causal --replacement-policy prefix_mean_v1 --epochs 80 --selection-protocol validation --defer-final-test --fail-on-nonfinite --diagnostic-batches 1 --num-workers 0 --auxiliary-policy legacy_epoch_snapshot --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed3407/without_cc --contrastlearning --mscl_coff 0.05 --cscl_coff 0.05 --courselearning --scheduler_steps 1 --epoch_ratio 0.15 --difficulty-path "${DIFFICULTY_IEMOCAP_3407:?bind_verified_DIFFICULTY_IEMOCAP_3407}"
```

### extension_same_step_same_mode

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import sys;from causal_provenance import sha256_file;actual=sha256_file(sys.argv[1]);
if actual!=sys.argv[2]: raise ValueError("Stage-B difficulty SHA256 changed before training")' "${DIFFICULTY_IEMOCAP_3407:?bind_verified_DIFFICULTY_IEMOCAP_3407}" "${DIFFICULTY_SHA_IEMOCAP_3407:?bind_verified_DIFFICULTY_SHA_IEMOCAP_3407}" && CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u train_cm.py --Dataset IEMOCAP --base-model GRU --graph_type hyper --graph_construct direct --multi_modal --mm_fusion_mthd concat_DHT --modals avl --norm LN --num_L 3 --num_K 4 --batch-size 16 --lr 0.0001 --l2 0.00003 --dropout 0.5 --rec-dropout 0.1 --seed 3407 --split-seed 1475 --causal --replacement-policy prefix_mean_v1 --epochs 80 --selection-protocol validation --defer-final-test --fail-on-nonfinite --diagnostic-batches 1 --num-workers 0 --auxiliary-policy same_step_same_mode --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/IEMOCAP/seed3407/same_step_same_mode --calibrate --rank_coff 0.005 --contrastlearning --mscl_coff 0.05 --cscl_coff 0.05 --courselearning --scheduler_steps 1 --epoch_ratio 0.15 --difficulty-path "${DIFFICULTY_IEMOCAP_3407:?bind_verified_DIFFICULTY_IEMOCAP_3407}"
```

按 ordered_stages 执行；comparison 与冻结成功后才执行 test，全部原始产物保留。

## MELD / training seed 1937

### baseline_training

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u train_cm.py --Dataset MELD --base-model GRU --graph_type hyper --graph_construct direct --multi_modal --mm_fusion_mthd concat_DHT --modals avl --norm LN --num_L 3 --num_K 3 --batch-size 16 --lr 0.0001 --l2 0.00003 --dropout 0.4 --rec-dropout 0.1 --seed 1937 --split-seed 1475 --causal --replacement-policy prefix_mean_v1 --epochs 40 --selection-protocol validation --defer-final-test --fail-on-nonfinite --diagnostic-batches 1 --num-workers 0 --auxiliary-policy legacy_epoch_snapshot --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed1937/baseline
```

### strict_teacher_admission

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u scripts/round2_verify_run.py --dataset MELD --seed 1937 --split-seed 1475 --epochs 40 --expected-provenance "${DATA_ENVIRONMENT_MELD_1937:?bind_verified_DATA_ENVIRONMENT_MELD_1937}" --variant baseline --run-dir "${BASELINE_RUN_MELD_1937:?bind_verified_BASELINE_RUN_MELD_1937}" --output "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed1937/verification/baseline.json
```

### difficulty_generation

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u caldiff/cal_diff.py --Dataset MELD --base-model GRU --graph_type hyper --graph_construct direct --multi_modal --mm_fusion_mthd concat_DHT --modals avl --norm LN --num_L 3 --num_K 3 --batch-size 16 --lr 0.0001 --l2 0.00003 --dropout 0.4 --rec-dropout 0.1 --seed 1937 --split-seed 1475 --causal --replacement-policy prefix_mean_v1 --ckpt_path "${TEACHER_BEST_MELD_1937:?bind_verified_TEACHER_BEST_MELD_1937}" --difficulty-output "${DIFFICULTY_MELD_1937:?bind_verified_DIFFICULTY_MELD_1937}"
```

### strict_difficulty_admission

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import json,sys,torch
from pathlib import Path
from causal_provenance import architecture_from_args,resolve_causal_difficulty,sha256_file
dataset,seed,path,teacher_path,inventory_path,baseline_run=sys.argv[1:]
teacher=torch.load(teacher_path,map_location='"'"'cpu'"'"')
if sha256_file(teacher_path)!=sha256_file(Path(baseline_run)/'"'"'best.pt'"'"'): raise ValueError('"'"'Teacher is not the admitted paired baseline best'"'"')
expected=json.loads(Path(inventory_path).read_text())['"'"'data_provenance'"'"']
values,artifact=resolve_causal_difficulty(path,dataset,expected,architecture_from_args(teacher['"'"'args'"'"']),expected_teacher_seed=int(seed))
source=artifact['"'"'manifest'"'"']['"'"'source_checkpoint'"'"']
if source['"'"'sha256'"'"']!=sha256_file(teacher_path) or source['"'"'epoch'"'"']!=source['"'"'best_epoch'"'"']: raise ValueError('"'"'Difficulty teacher is not the admitted baseline best'"'"')
print(json.dumps({'"'"'status'"'"':'"'"'passed'"'"','"'"'difficulty_sha256'"'"':artifact['"'"'sha256'"'"'],'"'"'teacher_sha256'"'"':source['"'"'sha256'"'"'],'"'"'train_count'"'"':len(values)}))' MELD 1937 "${DIFFICULTY_MELD_1937:?bind_verified_DIFFICULTY_MELD_1937}" "${TEACHER_BEST_MELD_1937:?bind_verified_TEACHER_BEST_MELD_1937}" "${DATA_ENVIRONMENT_MELD_1937:?bind_verified_DATA_ENVIRONMENT_MELD_1937}" "${BASELINE_RUN_MELD_1937:?bind_verified_BASELINE_RUN_MELD_1937}"
```

### full_training

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import sys;from causal_provenance import sha256_file;actual=sha256_file(sys.argv[1]);
if actual!=sys.argv[2]: raise ValueError("Stage-B difficulty SHA256 changed before training")' "${DIFFICULTY_MELD_1937:?bind_verified_DIFFICULTY_MELD_1937}" "${DIFFICULTY_SHA_MELD_1937:?bind_verified_DIFFICULTY_SHA_MELD_1937}" && CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u train_cm.py --Dataset MELD --base-model GRU --graph_type hyper --graph_construct direct --multi_modal --mm_fusion_mthd concat_DHT --modals avl --norm LN --num_L 3 --num_K 3 --batch-size 16 --lr 0.0001 --l2 0.00003 --dropout 0.4 --rec-dropout 0.1 --seed 1937 --split-seed 1475 --causal --replacement-policy prefix_mean_v1 --epochs 40 --selection-protocol validation --defer-final-test --fail-on-nonfinite --diagnostic-batches 1 --num-workers 0 --auxiliary-policy legacy_epoch_snapshot --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed1937/full --calibrate --rank_coff 0.002 --contrastlearning --mscl_coff 0.15 --cscl_coff 0.15 --courselearning --scheduler_steps 1 --epoch_ratio 0.4 --difficulty-path "${DIFFICULTY_MELD_1937:?bind_verified_DIFFICULTY_MELD_1937}"
```

### completed_full_and_pair_admission

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u scripts/round2_verify_run.py --dataset MELD --seed 1937 --split-seed 1475 --epochs 40 --expected-provenance "${DATA_ENVIRONMENT_MELD_1937:?bind_verified_DATA_ENVIRONMENT_MELD_1937}" --variant cmerc --run-dir "${CMERC_RUN_MELD_1937:?bind_verified_CMERC_RUN_MELD_1937}" --paired-run "${BASELINE_RUN_MELD_1937:?bind_verified_BASELINE_RUN_MELD_1937}" --difficulty "${DIFFICULTY_MELD_1937:?bind_verified_DIFFICULTY_MELD_1937}" --expected-difficulty-sha256 "${DIFFICULTY_SHA_MELD_1937:?bind_verified_DIFFICULTY_SHA_MELD_1937}" --output "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed1937/verification/full.json
```

### validation_evaluation_baseline

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u scripts/evaluate_checkpoint.py --dataset MELD --causal --checkpoint "${TEACHER_BEST_MELD_1937:?bind_verified_TEACHER_BEST_MELD_1937}" --device cuda:0 --split validation --split-seed 1475 --batch-size 16 --num-workers 0 --n-bins 15 --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed1937/evaluation/validation/baseline
```

### validation_evaluation_full

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u scripts/evaluate_checkpoint.py --dataset MELD --causal --checkpoint "${CMERC_BEST_MELD_1937:?bind_verified_CMERC_BEST_MELD_1937}" --device cuda:0 --split validation --split-seed 1475 --batch-size 16 --num-workers 0 --n-bins 15 --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed1937/evaluation/validation/full
```

### validation_comparison

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u scripts/round2_analysis.py --baseline-eval "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed1937/evaluation/validation/baseline --cmerc-eval "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed1937/evaluation/validation/full --baseline-training "${BASELINE_RUN_MELD_1937:?bind_verified_BASELINE_RUN_MELD_1937}"/metrics.jsonl --cmerc-training "${CMERC_RUN_MELD_1937:?bind_verified_CMERC_RUN_MELD_1937}"/metrics.jsonl --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed1937/comparison_validation
```

### validation_freeze_create

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import datetime,json,sys,torch
from pathlib import Path
from causal_provenance import sha256_file
mode,dataset,seed,comparison_path,baseline_path,full_path,baseline_eval,full_eval,freeze_path,*extra=sys.argv[1:]
seed=int(seed)
def check(condition,message):
    if not condition: raise ValueError(message)
comparison=json.loads(Path(comparison_path).read_text())
identity=dict(dataset=dataset,split='"'"'validation'"'"',split_seed=1475,replacement_policy='"'"'prefix_mean_v1'"'"')
check(all(comparison.get(key)==value for key,value in identity.items()),'"'"'Validation comparison identity mismatch'"'"')
best_hashes={}
for name,path,directory in [('"'"'baseline'"'"',baseline_path,baseline_eval),('"'"'cmerc'"'"',full_path,full_eval)]:
    checkpoint=torch.load(path,map_location='"'"'cpu'"'"'); args=checkpoint['"'"'args'"'"']
    check(args.get('"'"'Dataset'"'"')==dataset and args.get('"'"'seed'"'"')==seed and args.get('"'"'split_seed'"'"')==1475,'"'"'Frozen dataset/training seed mismatch'"'"')
    check(args.get('"'"'causal'"'"') is True and args.get('"'"'selection_protocol'"'"')=='"'"'validation'"'"' and checkpoint['"'"'epoch'"'"']==checkpoint['"'"'best_epoch'"'"'],'"'"'Frozen checkpoint is not validation-selected causal best'"'"')
    metadata=json.loads((Path(directory)/'"'"'metrics.json'"'"').read_text())
    check(all(metadata.get(key)==value for key,value in identity.items()),'"'"'Validation evaluator identity mismatch'"'"')
    check(Path(metadata['"'"'checkpoint'"'"']).resolve()==Path(path).resolve(),'"'"'Validation evaluated a different checkpoint'"'"')
    for filename in ('"'"'metrics.json'"'"','"'"'predictions.npz'"'"'):
        check(comparison['"'"'input_sha256'"'"'][name][filename]==sha256_file(Path(directory)/filename),'"'"'Validation comparison input SHA256 changed'"'"')
    best_hashes[name]=sha256_file(path)
comparison_sha=sha256_file(comparison_path)
if mode=='"'"'create'"'"':
    frozen={**comparison,'"'"'training_seed'"'"':seed,'"'"'validation_comparison_sha256'"'"':comparison_sha,'"'"'best_sha256'"'"':best_hashes,'"'"'frozen_utc'"'"':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with Path(freeze_path).open('"'"'x'"'"') as handle: json.dump(frozen,handle,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps({'"'"'status'"'"':'"'"'frozen'"'"','"'"'validation_freeze_sha256'"'"':sha256_file(freeze_path),'"'"'training_seed'"'"':seed,'"'"'best_sha256'"'"':best_hashes}))
elif mode=='"'"'check'"'"':
    check(len(extra)==1 and sha256_file(freeze_path)==extra[0],'"'"'Frozen validation JSON SHA256 changed'"'"')
    frozen=json.loads(Path(freeze_path).read_text())
    check(all(frozen.get(key)==value for key,value in identity.items()) and frozen.get('"'"'training_seed'"'"')==seed,'"'"'Frozen validation identity mismatch'"'"')
    check(frozen.get('"'"'validation_comparison_sha256'"'"')==comparison_sha and frozen.get('"'"'best_sha256'"'"')==best_hashes,'"'"'Frozen comparison or selected best SHA256 changed'"'"')
    check(frozen.get('"'"'input_sha256'"'"')==comparison.get('"'"'input_sha256'"'"'),'"'"'Frozen validation input identities changed'"'"')
    print(json.dumps({'"'"'status'"'"':'"'"'passed'"'"','"'"'validation_freeze_sha256'"'"':extra[0]}))
else: raise ValueError('"'"'Unknown freeze operation'"'"')' create MELD 1937 "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed1937/comparison_validation/comparison.json "${TEACHER_BEST_MELD_1937:?bind_verified_TEACHER_BEST_MELD_1937}" "${CMERC_BEST_MELD_1937:?bind_verified_CMERC_BEST_MELD_1937}" "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed1937/evaluation/validation/baseline "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed1937/evaluation/validation/full "${VALIDATION_FREEZE_MELD_1937:?bind_verified_VALIDATION_FREEZE_MELD_1937}"
```

### validation_freeze_check

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import datetime,json,sys,torch
from pathlib import Path
from causal_provenance import sha256_file
mode,dataset,seed,comparison_path,baseline_path,full_path,baseline_eval,full_eval,freeze_path,*extra=sys.argv[1:]
seed=int(seed)
def check(condition,message):
    if not condition: raise ValueError(message)
comparison=json.loads(Path(comparison_path).read_text())
identity=dict(dataset=dataset,split='"'"'validation'"'"',split_seed=1475,replacement_policy='"'"'prefix_mean_v1'"'"')
check(all(comparison.get(key)==value for key,value in identity.items()),'"'"'Validation comparison identity mismatch'"'"')
best_hashes={}
for name,path,directory in [('"'"'baseline'"'"',baseline_path,baseline_eval),('"'"'cmerc'"'"',full_path,full_eval)]:
    checkpoint=torch.load(path,map_location='"'"'cpu'"'"'); args=checkpoint['"'"'args'"'"']
    check(args.get('"'"'Dataset'"'"')==dataset and args.get('"'"'seed'"'"')==seed and args.get('"'"'split_seed'"'"')==1475,'"'"'Frozen dataset/training seed mismatch'"'"')
    check(args.get('"'"'causal'"'"') is True and args.get('"'"'selection_protocol'"'"')=='"'"'validation'"'"' and checkpoint['"'"'epoch'"'"']==checkpoint['"'"'best_epoch'"'"'],'"'"'Frozen checkpoint is not validation-selected causal best'"'"')
    metadata=json.loads((Path(directory)/'"'"'metrics.json'"'"').read_text())
    check(all(metadata.get(key)==value for key,value in identity.items()),'"'"'Validation evaluator identity mismatch'"'"')
    check(Path(metadata['"'"'checkpoint'"'"']).resolve()==Path(path).resolve(),'"'"'Validation evaluated a different checkpoint'"'"')
    for filename in ('"'"'metrics.json'"'"','"'"'predictions.npz'"'"'):
        check(comparison['"'"'input_sha256'"'"'][name][filename]==sha256_file(Path(directory)/filename),'"'"'Validation comparison input SHA256 changed'"'"')
    best_hashes[name]=sha256_file(path)
comparison_sha=sha256_file(comparison_path)
if mode=='"'"'create'"'"':
    frozen={**comparison,'"'"'training_seed'"'"':seed,'"'"'validation_comparison_sha256'"'"':comparison_sha,'"'"'best_sha256'"'"':best_hashes,'"'"'frozen_utc'"'"':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with Path(freeze_path).open('"'"'x'"'"') as handle: json.dump(frozen,handle,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps({'"'"'status'"'"':'"'"'frozen'"'"','"'"'validation_freeze_sha256'"'"':sha256_file(freeze_path),'"'"'training_seed'"'"':seed,'"'"'best_sha256'"'"':best_hashes}))
elif mode=='"'"'check'"'"':
    check(len(extra)==1 and sha256_file(freeze_path)==extra[0],'"'"'Frozen validation JSON SHA256 changed'"'"')
    frozen=json.loads(Path(freeze_path).read_text())
    check(all(frozen.get(key)==value for key,value in identity.items()) and frozen.get('"'"'training_seed'"'"')==seed,'"'"'Frozen validation identity mismatch'"'"')
    check(frozen.get('"'"'validation_comparison_sha256'"'"')==comparison_sha and frozen.get('"'"'best_sha256'"'"')==best_hashes,'"'"'Frozen comparison or selected best SHA256 changed'"'"')
    check(frozen.get('"'"'input_sha256'"'"')==comparison.get('"'"'input_sha256'"'"'),'"'"'Frozen validation input identities changed'"'"')
    print(json.dumps({'"'"'status'"'"':'"'"'passed'"'"','"'"'validation_freeze_sha256'"'"':extra[0]}))
else: raise ValueError('"'"'Unknown freeze operation'"'"')' check MELD 1937 "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed1937/comparison_validation/comparison.json "${TEACHER_BEST_MELD_1937:?bind_verified_TEACHER_BEST_MELD_1937}" "${CMERC_BEST_MELD_1937:?bind_verified_CMERC_BEST_MELD_1937}" "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed1937/evaluation/validation/baseline "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed1937/evaluation/validation/full "${VALIDATION_FREEZE_MELD_1937:?bind_verified_VALIDATION_FREEZE_MELD_1937}" "${VALIDATION_FREEZE_SHA_MELD_1937:?bind_verified_VALIDATION_FREEZE_SHA_MELD_1937}"
```

### test_evaluation_baseline

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import datetime,json,sys,torch
from pathlib import Path
from causal_provenance import sha256_file
mode,dataset,seed,comparison_path,baseline_path,full_path,baseline_eval,full_eval,freeze_path,*extra=sys.argv[1:]
seed=int(seed)
def check(condition,message):
    if not condition: raise ValueError(message)
comparison=json.loads(Path(comparison_path).read_text())
identity=dict(dataset=dataset,split='"'"'validation'"'"',split_seed=1475,replacement_policy='"'"'prefix_mean_v1'"'"')
check(all(comparison.get(key)==value for key,value in identity.items()),'"'"'Validation comparison identity mismatch'"'"')
best_hashes={}
for name,path,directory in [('"'"'baseline'"'"',baseline_path,baseline_eval),('"'"'cmerc'"'"',full_path,full_eval)]:
    checkpoint=torch.load(path,map_location='"'"'cpu'"'"'); args=checkpoint['"'"'args'"'"']
    check(args.get('"'"'Dataset'"'"')==dataset and args.get('"'"'seed'"'"')==seed and args.get('"'"'split_seed'"'"')==1475,'"'"'Frozen dataset/training seed mismatch'"'"')
    check(args.get('"'"'causal'"'"') is True and args.get('"'"'selection_protocol'"'"')=='"'"'validation'"'"' and checkpoint['"'"'epoch'"'"']==checkpoint['"'"'best_epoch'"'"'],'"'"'Frozen checkpoint is not validation-selected causal best'"'"')
    metadata=json.loads((Path(directory)/'"'"'metrics.json'"'"').read_text())
    check(all(metadata.get(key)==value for key,value in identity.items()),'"'"'Validation evaluator identity mismatch'"'"')
    check(Path(metadata['"'"'checkpoint'"'"']).resolve()==Path(path).resolve(),'"'"'Validation evaluated a different checkpoint'"'"')
    for filename in ('"'"'metrics.json'"'"','"'"'predictions.npz'"'"'):
        check(comparison['"'"'input_sha256'"'"'][name][filename]==sha256_file(Path(directory)/filename),'"'"'Validation comparison input SHA256 changed'"'"')
    best_hashes[name]=sha256_file(path)
comparison_sha=sha256_file(comparison_path)
if mode=='"'"'create'"'"':
    frozen={**comparison,'"'"'training_seed'"'"':seed,'"'"'validation_comparison_sha256'"'"':comparison_sha,'"'"'best_sha256'"'"':best_hashes,'"'"'frozen_utc'"'"':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with Path(freeze_path).open('"'"'x'"'"') as handle: json.dump(frozen,handle,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps({'"'"'status'"'"':'"'"'frozen'"'"','"'"'validation_freeze_sha256'"'"':sha256_file(freeze_path),'"'"'training_seed'"'"':seed,'"'"'best_sha256'"'"':best_hashes}))
elif mode=='"'"'check'"'"':
    check(len(extra)==1 and sha256_file(freeze_path)==extra[0],'"'"'Frozen validation JSON SHA256 changed'"'"')
    frozen=json.loads(Path(freeze_path).read_text())
    check(all(frozen.get(key)==value for key,value in identity.items()) and frozen.get('"'"'training_seed'"'"')==seed,'"'"'Frozen validation identity mismatch'"'"')
    check(frozen.get('"'"'validation_comparison_sha256'"'"')==comparison_sha and frozen.get('"'"'best_sha256'"'"')==best_hashes,'"'"'Frozen comparison or selected best SHA256 changed'"'"')
    check(frozen.get('"'"'input_sha256'"'"')==comparison.get('"'"'input_sha256'"'"'),'"'"'Frozen validation input identities changed'"'"')
    print(json.dumps({'"'"'status'"'"':'"'"'passed'"'"','"'"'validation_freeze_sha256'"'"':extra[0]}))
else: raise ValueError('"'"'Unknown freeze operation'"'"')' check MELD 1937 "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed1937/comparison_validation/comparison.json "${TEACHER_BEST_MELD_1937:?bind_verified_TEACHER_BEST_MELD_1937}" "${CMERC_BEST_MELD_1937:?bind_verified_CMERC_BEST_MELD_1937}" "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed1937/evaluation/validation/baseline "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed1937/evaluation/validation/full "${VALIDATION_FREEZE_MELD_1937:?bind_verified_VALIDATION_FREEZE_MELD_1937}" "${VALIDATION_FREEZE_SHA_MELD_1937:?bind_verified_VALIDATION_FREEZE_SHA_MELD_1937}" && test ! -e "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed1937/evaluation/test/baseline && CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u scripts/evaluate_checkpoint.py --dataset MELD --causal --checkpoint "${TEACHER_BEST_MELD_1937:?bind_verified_TEACHER_BEST_MELD_1937}" --device cuda:0 --split test --split-seed 1475 --batch-size 16 --num-workers 0 --n-bins 15 --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed1937/evaluation/test/baseline
```

### test_evaluation_full

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import datetime,json,sys,torch
from pathlib import Path
from causal_provenance import sha256_file
mode,dataset,seed,comparison_path,baseline_path,full_path,baseline_eval,full_eval,freeze_path,*extra=sys.argv[1:]
seed=int(seed)
def check(condition,message):
    if not condition: raise ValueError(message)
comparison=json.loads(Path(comparison_path).read_text())
identity=dict(dataset=dataset,split='"'"'validation'"'"',split_seed=1475,replacement_policy='"'"'prefix_mean_v1'"'"')
check(all(comparison.get(key)==value for key,value in identity.items()),'"'"'Validation comparison identity mismatch'"'"')
best_hashes={}
for name,path,directory in [('"'"'baseline'"'"',baseline_path,baseline_eval),('"'"'cmerc'"'"',full_path,full_eval)]:
    checkpoint=torch.load(path,map_location='"'"'cpu'"'"'); args=checkpoint['"'"'args'"'"']
    check(args.get('"'"'Dataset'"'"')==dataset and args.get('"'"'seed'"'"')==seed and args.get('"'"'split_seed'"'"')==1475,'"'"'Frozen dataset/training seed mismatch'"'"')
    check(args.get('"'"'causal'"'"') is True and args.get('"'"'selection_protocol'"'"')=='"'"'validation'"'"' and checkpoint['"'"'epoch'"'"']==checkpoint['"'"'best_epoch'"'"'],'"'"'Frozen checkpoint is not validation-selected causal best'"'"')
    metadata=json.loads((Path(directory)/'"'"'metrics.json'"'"').read_text())
    check(all(metadata.get(key)==value for key,value in identity.items()),'"'"'Validation evaluator identity mismatch'"'"')
    check(Path(metadata['"'"'checkpoint'"'"']).resolve()==Path(path).resolve(),'"'"'Validation evaluated a different checkpoint'"'"')
    for filename in ('"'"'metrics.json'"'"','"'"'predictions.npz'"'"'):
        check(comparison['"'"'input_sha256'"'"'][name][filename]==sha256_file(Path(directory)/filename),'"'"'Validation comparison input SHA256 changed'"'"')
    best_hashes[name]=sha256_file(path)
comparison_sha=sha256_file(comparison_path)
if mode=='"'"'create'"'"':
    frozen={**comparison,'"'"'training_seed'"'"':seed,'"'"'validation_comparison_sha256'"'"':comparison_sha,'"'"'best_sha256'"'"':best_hashes,'"'"'frozen_utc'"'"':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with Path(freeze_path).open('"'"'x'"'"') as handle: json.dump(frozen,handle,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps({'"'"'status'"'"':'"'"'frozen'"'"','"'"'validation_freeze_sha256'"'"':sha256_file(freeze_path),'"'"'training_seed'"'"':seed,'"'"'best_sha256'"'"':best_hashes}))
elif mode=='"'"'check'"'"':
    check(len(extra)==1 and sha256_file(freeze_path)==extra[0],'"'"'Frozen validation JSON SHA256 changed'"'"')
    frozen=json.loads(Path(freeze_path).read_text())
    check(all(frozen.get(key)==value for key,value in identity.items()) and frozen.get('"'"'training_seed'"'"')==seed,'"'"'Frozen validation identity mismatch'"'"')
    check(frozen.get('"'"'validation_comparison_sha256'"'"')==comparison_sha and frozen.get('"'"'best_sha256'"'"')==best_hashes,'"'"'Frozen comparison or selected best SHA256 changed'"'"')
    check(frozen.get('"'"'input_sha256'"'"')==comparison.get('"'"'input_sha256'"'"'),'"'"'Frozen validation input identities changed'"'"')
    print(json.dumps({'"'"'status'"'"':'"'"'passed'"'"','"'"'validation_freeze_sha256'"'"':extra[0]}))
else: raise ValueError('"'"'Unknown freeze operation'"'"')' check MELD 1937 "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed1937/comparison_validation/comparison.json "${TEACHER_BEST_MELD_1937:?bind_verified_TEACHER_BEST_MELD_1937}" "${CMERC_BEST_MELD_1937:?bind_verified_CMERC_BEST_MELD_1937}" "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed1937/evaluation/validation/baseline "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed1937/evaluation/validation/full "${VALIDATION_FREEZE_MELD_1937:?bind_verified_VALIDATION_FREEZE_MELD_1937}" "${VALIDATION_FREEZE_SHA_MELD_1937:?bind_verified_VALIDATION_FREEZE_SHA_MELD_1937}" && test ! -e "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed1937/evaluation/test/full && CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u scripts/evaluate_checkpoint.py --dataset MELD --causal --checkpoint "${CMERC_BEST_MELD_1937:?bind_verified_CMERC_BEST_MELD_1937}" --device cuda:0 --split test --split-seed 1475 --batch-size 16 --num-workers 0 --n-bins 15 --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed1937/evaluation/test/full
```

### test_comparison

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import datetime,json,sys,torch
from pathlib import Path
from causal_provenance import sha256_file
mode,dataset,seed,comparison_path,baseline_path,full_path,baseline_eval,full_eval,freeze_path,*extra=sys.argv[1:]
seed=int(seed)
def check(condition,message):
    if not condition: raise ValueError(message)
comparison=json.loads(Path(comparison_path).read_text())
identity=dict(dataset=dataset,split='"'"'validation'"'"',split_seed=1475,replacement_policy='"'"'prefix_mean_v1'"'"')
check(all(comparison.get(key)==value for key,value in identity.items()),'"'"'Validation comparison identity mismatch'"'"')
best_hashes={}
for name,path,directory in [('"'"'baseline'"'"',baseline_path,baseline_eval),('"'"'cmerc'"'"',full_path,full_eval)]:
    checkpoint=torch.load(path,map_location='"'"'cpu'"'"'); args=checkpoint['"'"'args'"'"']
    check(args.get('"'"'Dataset'"'"')==dataset and args.get('"'"'seed'"'"')==seed and args.get('"'"'split_seed'"'"')==1475,'"'"'Frozen dataset/training seed mismatch'"'"')
    check(args.get('"'"'causal'"'"') is True and args.get('"'"'selection_protocol'"'"')=='"'"'validation'"'"' and checkpoint['"'"'epoch'"'"']==checkpoint['"'"'best_epoch'"'"'],'"'"'Frozen checkpoint is not validation-selected causal best'"'"')
    metadata=json.loads((Path(directory)/'"'"'metrics.json'"'"').read_text())
    check(all(metadata.get(key)==value for key,value in identity.items()),'"'"'Validation evaluator identity mismatch'"'"')
    check(Path(metadata['"'"'checkpoint'"'"']).resolve()==Path(path).resolve(),'"'"'Validation evaluated a different checkpoint'"'"')
    for filename in ('"'"'metrics.json'"'"','"'"'predictions.npz'"'"'):
        check(comparison['"'"'input_sha256'"'"'][name][filename]==sha256_file(Path(directory)/filename),'"'"'Validation comparison input SHA256 changed'"'"')
    best_hashes[name]=sha256_file(path)
comparison_sha=sha256_file(comparison_path)
if mode=='"'"'create'"'"':
    frozen={**comparison,'"'"'training_seed'"'"':seed,'"'"'validation_comparison_sha256'"'"':comparison_sha,'"'"'best_sha256'"'"':best_hashes,'"'"'frozen_utc'"'"':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with Path(freeze_path).open('"'"'x'"'"') as handle: json.dump(frozen,handle,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps({'"'"'status'"'"':'"'"'frozen'"'"','"'"'validation_freeze_sha256'"'"':sha256_file(freeze_path),'"'"'training_seed'"'"':seed,'"'"'best_sha256'"'"':best_hashes}))
elif mode=='"'"'check'"'"':
    check(len(extra)==1 and sha256_file(freeze_path)==extra[0],'"'"'Frozen validation JSON SHA256 changed'"'"')
    frozen=json.loads(Path(freeze_path).read_text())
    check(all(frozen.get(key)==value for key,value in identity.items()) and frozen.get('"'"'training_seed'"'"')==seed,'"'"'Frozen validation identity mismatch'"'"')
    check(frozen.get('"'"'validation_comparison_sha256'"'"')==comparison_sha and frozen.get('"'"'best_sha256'"'"')==best_hashes,'"'"'Frozen comparison or selected best SHA256 changed'"'"')
    check(frozen.get('"'"'input_sha256'"'"')==comparison.get('"'"'input_sha256'"'"'),'"'"'Frozen validation input identities changed'"'"')
    print(json.dumps({'"'"'status'"'"':'"'"'passed'"'"','"'"'validation_freeze_sha256'"'"':extra[0]}))
else: raise ValueError('"'"'Unknown freeze operation'"'"')' check MELD 1937 "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed1937/comparison_validation/comparison.json "${TEACHER_BEST_MELD_1937:?bind_verified_TEACHER_BEST_MELD_1937}" "${CMERC_BEST_MELD_1937:?bind_verified_CMERC_BEST_MELD_1937}" "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed1937/evaluation/validation/baseline "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed1937/evaluation/validation/full "${VALIDATION_FREEZE_MELD_1937:?bind_verified_VALIDATION_FREEZE_MELD_1937}" "${VALIDATION_FREEZE_SHA_MELD_1937:?bind_verified_VALIDATION_FREEZE_SHA_MELD_1937}" && "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u scripts/round2_analysis.py --baseline-eval "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed1937/evaluation/test/baseline --cmerc-eval "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed1937/evaluation/test/full --baseline-training "${BASELINE_RUN_MELD_1937:?bind_verified_BASELINE_RUN_MELD_1937}"/metrics.jsonl --cmerc-training "${CMERC_RUN_MELD_1937:?bind_verified_CMERC_RUN_MELD_1937}"/metrics.jsonl --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed1937/comparison_test --validation-freeze "${VALIDATION_FREEZE_MELD_1937:?bind_verified_VALIDATION_FREEZE_MELD_1937}"
```

### extension_without_cl

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u train_cm.py --Dataset MELD --base-model GRU --graph_type hyper --graph_construct direct --multi_modal --mm_fusion_mthd concat_DHT --modals avl --norm LN --num_L 3 --num_K 3 --batch-size 16 --lr 0.0001 --l2 0.00003 --dropout 0.4 --rec-dropout 0.1 --seed 1937 --split-seed 1475 --causal --replacement-policy prefix_mean_v1 --epochs 40 --selection-protocol validation --defer-final-test --fail-on-nonfinite --diagnostic-batches 1 --num-workers 0 --auxiliary-policy legacy_epoch_snapshot --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed1937/without_cl --calibrate --rank_coff 0.002 --contrastlearning --mscl_coff 0.15 --cscl_coff 0.15
```

### extension_without_hscl

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import sys;from causal_provenance import sha256_file;actual=sha256_file(sys.argv[1]);
if actual!=sys.argv[2]: raise ValueError("Stage-B difficulty SHA256 changed before training")' "${DIFFICULTY_MELD_1937:?bind_verified_DIFFICULTY_MELD_1937}" "${DIFFICULTY_SHA_MELD_1937:?bind_verified_DIFFICULTY_SHA_MELD_1937}" && CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u train_cm.py --Dataset MELD --base-model GRU --graph_type hyper --graph_construct direct --multi_modal --mm_fusion_mthd concat_DHT --modals avl --norm LN --num_L 3 --num_K 3 --batch-size 16 --lr 0.0001 --l2 0.00003 --dropout 0.4 --rec-dropout 0.1 --seed 1937 --split-seed 1475 --causal --replacement-policy prefix_mean_v1 --epochs 40 --selection-protocol validation --defer-final-test --fail-on-nonfinite --diagnostic-batches 1 --num-workers 0 --auxiliary-policy legacy_epoch_snapshot --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed1937/without_hscl --calibrate --rank_coff 0.002 --courselearning --scheduler_steps 1 --epoch_ratio 0.4 --difficulty-path "${DIFFICULTY_MELD_1937:?bind_verified_DIFFICULTY_MELD_1937}"
```

### extension_without_cc

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import sys;from causal_provenance import sha256_file;actual=sha256_file(sys.argv[1]);
if actual!=sys.argv[2]: raise ValueError("Stage-B difficulty SHA256 changed before training")' "${DIFFICULTY_MELD_1937:?bind_verified_DIFFICULTY_MELD_1937}" "${DIFFICULTY_SHA_MELD_1937:?bind_verified_DIFFICULTY_SHA_MELD_1937}" && CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u train_cm.py --Dataset MELD --base-model GRU --graph_type hyper --graph_construct direct --multi_modal --mm_fusion_mthd concat_DHT --modals avl --norm LN --num_L 3 --num_K 3 --batch-size 16 --lr 0.0001 --l2 0.00003 --dropout 0.4 --rec-dropout 0.1 --seed 1937 --split-seed 1475 --causal --replacement-policy prefix_mean_v1 --epochs 40 --selection-protocol validation --defer-final-test --fail-on-nonfinite --diagnostic-batches 1 --num-workers 0 --auxiliary-policy legacy_epoch_snapshot --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed1937/without_cc --contrastlearning --mscl_coff 0.15 --cscl_coff 0.15 --courselearning --scheduler_steps 1 --epoch_ratio 0.4 --difficulty-path "${DIFFICULTY_MELD_1937:?bind_verified_DIFFICULTY_MELD_1937}"
```

### extension_same_step_same_mode

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import sys;from causal_provenance import sha256_file;actual=sha256_file(sys.argv[1]);
if actual!=sys.argv[2]: raise ValueError("Stage-B difficulty SHA256 changed before training")' "${DIFFICULTY_MELD_1937:?bind_verified_DIFFICULTY_MELD_1937}" "${DIFFICULTY_SHA_MELD_1937:?bind_verified_DIFFICULTY_SHA_MELD_1937}" && CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u train_cm.py --Dataset MELD --base-model GRU --graph_type hyper --graph_construct direct --multi_modal --mm_fusion_mthd concat_DHT --modals avl --norm LN --num_L 3 --num_K 3 --batch-size 16 --lr 0.0001 --l2 0.00003 --dropout 0.4 --rec-dropout 0.1 --seed 1937 --split-seed 1475 --causal --replacement-policy prefix_mean_v1 --epochs 40 --selection-protocol validation --defer-final-test --fail-on-nonfinite --diagnostic-batches 1 --num-workers 0 --auxiliary-policy same_step_same_mode --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed1937/same_step_same_mode --calibrate --rank_coff 0.002 --contrastlearning --mscl_coff 0.15 --cscl_coff 0.15 --courselearning --scheduler_steps 1 --epoch_ratio 0.4 --difficulty-path "${DIFFICULTY_MELD_1937:?bind_verified_DIFFICULTY_MELD_1937}"
```

按 ordered_stages 执行；comparison 与冻结成功后才执行 test，全部原始产物保留。

## MELD / training seed 2025

### baseline_training

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u train_cm.py --Dataset MELD --base-model GRU --graph_type hyper --graph_construct direct --multi_modal --mm_fusion_mthd concat_DHT --modals avl --norm LN --num_L 3 --num_K 3 --batch-size 16 --lr 0.0001 --l2 0.00003 --dropout 0.4 --rec-dropout 0.1 --seed 2025 --split-seed 1475 --causal --replacement-policy prefix_mean_v1 --epochs 40 --selection-protocol validation --defer-final-test --fail-on-nonfinite --diagnostic-batches 1 --num-workers 0 --auxiliary-policy legacy_epoch_snapshot --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed2025/baseline
```

### strict_teacher_admission

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u scripts/round2_verify_run.py --dataset MELD --seed 2025 --split-seed 1475 --epochs 40 --expected-provenance "${DATA_ENVIRONMENT_MELD_2025:?bind_verified_DATA_ENVIRONMENT_MELD_2025}" --variant baseline --run-dir "${BASELINE_RUN_MELD_2025:?bind_verified_BASELINE_RUN_MELD_2025}" --output "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed2025/verification/baseline.json
```

### difficulty_generation

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u caldiff/cal_diff.py --Dataset MELD --base-model GRU --graph_type hyper --graph_construct direct --multi_modal --mm_fusion_mthd concat_DHT --modals avl --norm LN --num_L 3 --num_K 3 --batch-size 16 --lr 0.0001 --l2 0.00003 --dropout 0.4 --rec-dropout 0.1 --seed 2025 --split-seed 1475 --causal --replacement-policy prefix_mean_v1 --ckpt_path "${TEACHER_BEST_MELD_2025:?bind_verified_TEACHER_BEST_MELD_2025}" --difficulty-output "${DIFFICULTY_MELD_2025:?bind_verified_DIFFICULTY_MELD_2025}"
```

### strict_difficulty_admission

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import json,sys,torch
from pathlib import Path
from causal_provenance import architecture_from_args,resolve_causal_difficulty,sha256_file
dataset,seed,path,teacher_path,inventory_path,baseline_run=sys.argv[1:]
teacher=torch.load(teacher_path,map_location='"'"'cpu'"'"')
if sha256_file(teacher_path)!=sha256_file(Path(baseline_run)/'"'"'best.pt'"'"'): raise ValueError('"'"'Teacher is not the admitted paired baseline best'"'"')
expected=json.loads(Path(inventory_path).read_text())['"'"'data_provenance'"'"']
values,artifact=resolve_causal_difficulty(path,dataset,expected,architecture_from_args(teacher['"'"'args'"'"']),expected_teacher_seed=int(seed))
source=artifact['"'"'manifest'"'"']['"'"'source_checkpoint'"'"']
if source['"'"'sha256'"'"']!=sha256_file(teacher_path) or source['"'"'epoch'"'"']!=source['"'"'best_epoch'"'"']: raise ValueError('"'"'Difficulty teacher is not the admitted baseline best'"'"')
print(json.dumps({'"'"'status'"'"':'"'"'passed'"'"','"'"'difficulty_sha256'"'"':artifact['"'"'sha256'"'"'],'"'"'teacher_sha256'"'"':source['"'"'sha256'"'"'],'"'"'train_count'"'"':len(values)}))' MELD 2025 "${DIFFICULTY_MELD_2025:?bind_verified_DIFFICULTY_MELD_2025}" "${TEACHER_BEST_MELD_2025:?bind_verified_TEACHER_BEST_MELD_2025}" "${DATA_ENVIRONMENT_MELD_2025:?bind_verified_DATA_ENVIRONMENT_MELD_2025}" "${BASELINE_RUN_MELD_2025:?bind_verified_BASELINE_RUN_MELD_2025}"
```

### full_training

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import sys;from causal_provenance import sha256_file;actual=sha256_file(sys.argv[1]);
if actual!=sys.argv[2]: raise ValueError("Stage-B difficulty SHA256 changed before training")' "${DIFFICULTY_MELD_2025:?bind_verified_DIFFICULTY_MELD_2025}" "${DIFFICULTY_SHA_MELD_2025:?bind_verified_DIFFICULTY_SHA_MELD_2025}" && CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u train_cm.py --Dataset MELD --base-model GRU --graph_type hyper --graph_construct direct --multi_modal --mm_fusion_mthd concat_DHT --modals avl --norm LN --num_L 3 --num_K 3 --batch-size 16 --lr 0.0001 --l2 0.00003 --dropout 0.4 --rec-dropout 0.1 --seed 2025 --split-seed 1475 --causal --replacement-policy prefix_mean_v1 --epochs 40 --selection-protocol validation --defer-final-test --fail-on-nonfinite --diagnostic-batches 1 --num-workers 0 --auxiliary-policy legacy_epoch_snapshot --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed2025/full --calibrate --rank_coff 0.002 --contrastlearning --mscl_coff 0.15 --cscl_coff 0.15 --courselearning --scheduler_steps 1 --epoch_ratio 0.4 --difficulty-path "${DIFFICULTY_MELD_2025:?bind_verified_DIFFICULTY_MELD_2025}"
```

### completed_full_and_pair_admission

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u scripts/round2_verify_run.py --dataset MELD --seed 2025 --split-seed 1475 --epochs 40 --expected-provenance "${DATA_ENVIRONMENT_MELD_2025:?bind_verified_DATA_ENVIRONMENT_MELD_2025}" --variant cmerc --run-dir "${CMERC_RUN_MELD_2025:?bind_verified_CMERC_RUN_MELD_2025}" --paired-run "${BASELINE_RUN_MELD_2025:?bind_verified_BASELINE_RUN_MELD_2025}" --difficulty "${DIFFICULTY_MELD_2025:?bind_verified_DIFFICULTY_MELD_2025}" --expected-difficulty-sha256 "${DIFFICULTY_SHA_MELD_2025:?bind_verified_DIFFICULTY_SHA_MELD_2025}" --output "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed2025/verification/full.json
```

### validation_evaluation_baseline

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u scripts/evaluate_checkpoint.py --dataset MELD --causal --checkpoint "${TEACHER_BEST_MELD_2025:?bind_verified_TEACHER_BEST_MELD_2025}" --device cuda:0 --split validation --split-seed 1475 --batch-size 16 --num-workers 0 --n-bins 15 --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed2025/evaluation/validation/baseline
```

### validation_evaluation_full

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u scripts/evaluate_checkpoint.py --dataset MELD --causal --checkpoint "${CMERC_BEST_MELD_2025:?bind_verified_CMERC_BEST_MELD_2025}" --device cuda:0 --split validation --split-seed 1475 --batch-size 16 --num-workers 0 --n-bins 15 --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed2025/evaluation/validation/full
```

### validation_comparison

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u scripts/round2_analysis.py --baseline-eval "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed2025/evaluation/validation/baseline --cmerc-eval "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed2025/evaluation/validation/full --baseline-training "${BASELINE_RUN_MELD_2025:?bind_verified_BASELINE_RUN_MELD_2025}"/metrics.jsonl --cmerc-training "${CMERC_RUN_MELD_2025:?bind_verified_CMERC_RUN_MELD_2025}"/metrics.jsonl --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed2025/comparison_validation
```

### validation_freeze_create

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import datetime,json,sys,torch
from pathlib import Path
from causal_provenance import sha256_file
mode,dataset,seed,comparison_path,baseline_path,full_path,baseline_eval,full_eval,freeze_path,*extra=sys.argv[1:]
seed=int(seed)
def check(condition,message):
    if not condition: raise ValueError(message)
comparison=json.loads(Path(comparison_path).read_text())
identity=dict(dataset=dataset,split='"'"'validation'"'"',split_seed=1475,replacement_policy='"'"'prefix_mean_v1'"'"')
check(all(comparison.get(key)==value for key,value in identity.items()),'"'"'Validation comparison identity mismatch'"'"')
best_hashes={}
for name,path,directory in [('"'"'baseline'"'"',baseline_path,baseline_eval),('"'"'cmerc'"'"',full_path,full_eval)]:
    checkpoint=torch.load(path,map_location='"'"'cpu'"'"'); args=checkpoint['"'"'args'"'"']
    check(args.get('"'"'Dataset'"'"')==dataset and args.get('"'"'seed'"'"')==seed and args.get('"'"'split_seed'"'"')==1475,'"'"'Frozen dataset/training seed mismatch'"'"')
    check(args.get('"'"'causal'"'"') is True and args.get('"'"'selection_protocol'"'"')=='"'"'validation'"'"' and checkpoint['"'"'epoch'"'"']==checkpoint['"'"'best_epoch'"'"'],'"'"'Frozen checkpoint is not validation-selected causal best'"'"')
    metadata=json.loads((Path(directory)/'"'"'metrics.json'"'"').read_text())
    check(all(metadata.get(key)==value for key,value in identity.items()),'"'"'Validation evaluator identity mismatch'"'"')
    check(Path(metadata['"'"'checkpoint'"'"']).resolve()==Path(path).resolve(),'"'"'Validation evaluated a different checkpoint'"'"')
    for filename in ('"'"'metrics.json'"'"','"'"'predictions.npz'"'"'):
        check(comparison['"'"'input_sha256'"'"'][name][filename]==sha256_file(Path(directory)/filename),'"'"'Validation comparison input SHA256 changed'"'"')
    best_hashes[name]=sha256_file(path)
comparison_sha=sha256_file(comparison_path)
if mode=='"'"'create'"'"':
    frozen={**comparison,'"'"'training_seed'"'"':seed,'"'"'validation_comparison_sha256'"'"':comparison_sha,'"'"'best_sha256'"'"':best_hashes,'"'"'frozen_utc'"'"':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with Path(freeze_path).open('"'"'x'"'"') as handle: json.dump(frozen,handle,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps({'"'"'status'"'"':'"'"'frozen'"'"','"'"'validation_freeze_sha256'"'"':sha256_file(freeze_path),'"'"'training_seed'"'"':seed,'"'"'best_sha256'"'"':best_hashes}))
elif mode=='"'"'check'"'"':
    check(len(extra)==1 and sha256_file(freeze_path)==extra[0],'"'"'Frozen validation JSON SHA256 changed'"'"')
    frozen=json.loads(Path(freeze_path).read_text())
    check(all(frozen.get(key)==value for key,value in identity.items()) and frozen.get('"'"'training_seed'"'"')==seed,'"'"'Frozen validation identity mismatch'"'"')
    check(frozen.get('"'"'validation_comparison_sha256'"'"')==comparison_sha and frozen.get('"'"'best_sha256'"'"')==best_hashes,'"'"'Frozen comparison or selected best SHA256 changed'"'"')
    check(frozen.get('"'"'input_sha256'"'"')==comparison.get('"'"'input_sha256'"'"'),'"'"'Frozen validation input identities changed'"'"')
    print(json.dumps({'"'"'status'"'"':'"'"'passed'"'"','"'"'validation_freeze_sha256'"'"':extra[0]}))
else: raise ValueError('"'"'Unknown freeze operation'"'"')' create MELD 2025 "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed2025/comparison_validation/comparison.json "${TEACHER_BEST_MELD_2025:?bind_verified_TEACHER_BEST_MELD_2025}" "${CMERC_BEST_MELD_2025:?bind_verified_CMERC_BEST_MELD_2025}" "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed2025/evaluation/validation/baseline "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed2025/evaluation/validation/full "${VALIDATION_FREEZE_MELD_2025:?bind_verified_VALIDATION_FREEZE_MELD_2025}"
```

### validation_freeze_check

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import datetime,json,sys,torch
from pathlib import Path
from causal_provenance import sha256_file
mode,dataset,seed,comparison_path,baseline_path,full_path,baseline_eval,full_eval,freeze_path,*extra=sys.argv[1:]
seed=int(seed)
def check(condition,message):
    if not condition: raise ValueError(message)
comparison=json.loads(Path(comparison_path).read_text())
identity=dict(dataset=dataset,split='"'"'validation'"'"',split_seed=1475,replacement_policy='"'"'prefix_mean_v1'"'"')
check(all(comparison.get(key)==value for key,value in identity.items()),'"'"'Validation comparison identity mismatch'"'"')
best_hashes={}
for name,path,directory in [('"'"'baseline'"'"',baseline_path,baseline_eval),('"'"'cmerc'"'"',full_path,full_eval)]:
    checkpoint=torch.load(path,map_location='"'"'cpu'"'"'); args=checkpoint['"'"'args'"'"']
    check(args.get('"'"'Dataset'"'"')==dataset and args.get('"'"'seed'"'"')==seed and args.get('"'"'split_seed'"'"')==1475,'"'"'Frozen dataset/training seed mismatch'"'"')
    check(args.get('"'"'causal'"'"') is True and args.get('"'"'selection_protocol'"'"')=='"'"'validation'"'"' and checkpoint['"'"'epoch'"'"']==checkpoint['"'"'best_epoch'"'"'],'"'"'Frozen checkpoint is not validation-selected causal best'"'"')
    metadata=json.loads((Path(directory)/'"'"'metrics.json'"'"').read_text())
    check(all(metadata.get(key)==value for key,value in identity.items()),'"'"'Validation evaluator identity mismatch'"'"')
    check(Path(metadata['"'"'checkpoint'"'"']).resolve()==Path(path).resolve(),'"'"'Validation evaluated a different checkpoint'"'"')
    for filename in ('"'"'metrics.json'"'"','"'"'predictions.npz'"'"'):
        check(comparison['"'"'input_sha256'"'"'][name][filename]==sha256_file(Path(directory)/filename),'"'"'Validation comparison input SHA256 changed'"'"')
    best_hashes[name]=sha256_file(path)
comparison_sha=sha256_file(comparison_path)
if mode=='"'"'create'"'"':
    frozen={**comparison,'"'"'training_seed'"'"':seed,'"'"'validation_comparison_sha256'"'"':comparison_sha,'"'"'best_sha256'"'"':best_hashes,'"'"'frozen_utc'"'"':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with Path(freeze_path).open('"'"'x'"'"') as handle: json.dump(frozen,handle,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps({'"'"'status'"'"':'"'"'frozen'"'"','"'"'validation_freeze_sha256'"'"':sha256_file(freeze_path),'"'"'training_seed'"'"':seed,'"'"'best_sha256'"'"':best_hashes}))
elif mode=='"'"'check'"'"':
    check(len(extra)==1 and sha256_file(freeze_path)==extra[0],'"'"'Frozen validation JSON SHA256 changed'"'"')
    frozen=json.loads(Path(freeze_path).read_text())
    check(all(frozen.get(key)==value for key,value in identity.items()) and frozen.get('"'"'training_seed'"'"')==seed,'"'"'Frozen validation identity mismatch'"'"')
    check(frozen.get('"'"'validation_comparison_sha256'"'"')==comparison_sha and frozen.get('"'"'best_sha256'"'"')==best_hashes,'"'"'Frozen comparison or selected best SHA256 changed'"'"')
    check(frozen.get('"'"'input_sha256'"'"')==comparison.get('"'"'input_sha256'"'"'),'"'"'Frozen validation input identities changed'"'"')
    print(json.dumps({'"'"'status'"'"':'"'"'passed'"'"','"'"'validation_freeze_sha256'"'"':extra[0]}))
else: raise ValueError('"'"'Unknown freeze operation'"'"')' check MELD 2025 "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed2025/comparison_validation/comparison.json "${TEACHER_BEST_MELD_2025:?bind_verified_TEACHER_BEST_MELD_2025}" "${CMERC_BEST_MELD_2025:?bind_verified_CMERC_BEST_MELD_2025}" "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed2025/evaluation/validation/baseline "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed2025/evaluation/validation/full "${VALIDATION_FREEZE_MELD_2025:?bind_verified_VALIDATION_FREEZE_MELD_2025}" "${VALIDATION_FREEZE_SHA_MELD_2025:?bind_verified_VALIDATION_FREEZE_SHA_MELD_2025}"
```

### test_evaluation_baseline

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import datetime,json,sys,torch
from pathlib import Path
from causal_provenance import sha256_file
mode,dataset,seed,comparison_path,baseline_path,full_path,baseline_eval,full_eval,freeze_path,*extra=sys.argv[1:]
seed=int(seed)
def check(condition,message):
    if not condition: raise ValueError(message)
comparison=json.loads(Path(comparison_path).read_text())
identity=dict(dataset=dataset,split='"'"'validation'"'"',split_seed=1475,replacement_policy='"'"'prefix_mean_v1'"'"')
check(all(comparison.get(key)==value for key,value in identity.items()),'"'"'Validation comparison identity mismatch'"'"')
best_hashes={}
for name,path,directory in [('"'"'baseline'"'"',baseline_path,baseline_eval),('"'"'cmerc'"'"',full_path,full_eval)]:
    checkpoint=torch.load(path,map_location='"'"'cpu'"'"'); args=checkpoint['"'"'args'"'"']
    check(args.get('"'"'Dataset'"'"')==dataset and args.get('"'"'seed'"'"')==seed and args.get('"'"'split_seed'"'"')==1475,'"'"'Frozen dataset/training seed mismatch'"'"')
    check(args.get('"'"'causal'"'"') is True and args.get('"'"'selection_protocol'"'"')=='"'"'validation'"'"' and checkpoint['"'"'epoch'"'"']==checkpoint['"'"'best_epoch'"'"'],'"'"'Frozen checkpoint is not validation-selected causal best'"'"')
    metadata=json.loads((Path(directory)/'"'"'metrics.json'"'"').read_text())
    check(all(metadata.get(key)==value for key,value in identity.items()),'"'"'Validation evaluator identity mismatch'"'"')
    check(Path(metadata['"'"'checkpoint'"'"']).resolve()==Path(path).resolve(),'"'"'Validation evaluated a different checkpoint'"'"')
    for filename in ('"'"'metrics.json'"'"','"'"'predictions.npz'"'"'):
        check(comparison['"'"'input_sha256'"'"'][name][filename]==sha256_file(Path(directory)/filename),'"'"'Validation comparison input SHA256 changed'"'"')
    best_hashes[name]=sha256_file(path)
comparison_sha=sha256_file(comparison_path)
if mode=='"'"'create'"'"':
    frozen={**comparison,'"'"'training_seed'"'"':seed,'"'"'validation_comparison_sha256'"'"':comparison_sha,'"'"'best_sha256'"'"':best_hashes,'"'"'frozen_utc'"'"':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with Path(freeze_path).open('"'"'x'"'"') as handle: json.dump(frozen,handle,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps({'"'"'status'"'"':'"'"'frozen'"'"','"'"'validation_freeze_sha256'"'"':sha256_file(freeze_path),'"'"'training_seed'"'"':seed,'"'"'best_sha256'"'"':best_hashes}))
elif mode=='"'"'check'"'"':
    check(len(extra)==1 and sha256_file(freeze_path)==extra[0],'"'"'Frozen validation JSON SHA256 changed'"'"')
    frozen=json.loads(Path(freeze_path).read_text())
    check(all(frozen.get(key)==value for key,value in identity.items()) and frozen.get('"'"'training_seed'"'"')==seed,'"'"'Frozen validation identity mismatch'"'"')
    check(frozen.get('"'"'validation_comparison_sha256'"'"')==comparison_sha and frozen.get('"'"'best_sha256'"'"')==best_hashes,'"'"'Frozen comparison or selected best SHA256 changed'"'"')
    check(frozen.get('"'"'input_sha256'"'"')==comparison.get('"'"'input_sha256'"'"'),'"'"'Frozen validation input identities changed'"'"')
    print(json.dumps({'"'"'status'"'"':'"'"'passed'"'"','"'"'validation_freeze_sha256'"'"':extra[0]}))
else: raise ValueError('"'"'Unknown freeze operation'"'"')' check MELD 2025 "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed2025/comparison_validation/comparison.json "${TEACHER_BEST_MELD_2025:?bind_verified_TEACHER_BEST_MELD_2025}" "${CMERC_BEST_MELD_2025:?bind_verified_CMERC_BEST_MELD_2025}" "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed2025/evaluation/validation/baseline "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed2025/evaluation/validation/full "${VALIDATION_FREEZE_MELD_2025:?bind_verified_VALIDATION_FREEZE_MELD_2025}" "${VALIDATION_FREEZE_SHA_MELD_2025:?bind_verified_VALIDATION_FREEZE_SHA_MELD_2025}" && test ! -e "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed2025/evaluation/test/baseline && CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u scripts/evaluate_checkpoint.py --dataset MELD --causal --checkpoint "${TEACHER_BEST_MELD_2025:?bind_verified_TEACHER_BEST_MELD_2025}" --device cuda:0 --split test --split-seed 1475 --batch-size 16 --num-workers 0 --n-bins 15 --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed2025/evaluation/test/baseline
```

### test_evaluation_full

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import datetime,json,sys,torch
from pathlib import Path
from causal_provenance import sha256_file
mode,dataset,seed,comparison_path,baseline_path,full_path,baseline_eval,full_eval,freeze_path,*extra=sys.argv[1:]
seed=int(seed)
def check(condition,message):
    if not condition: raise ValueError(message)
comparison=json.loads(Path(comparison_path).read_text())
identity=dict(dataset=dataset,split='"'"'validation'"'"',split_seed=1475,replacement_policy='"'"'prefix_mean_v1'"'"')
check(all(comparison.get(key)==value for key,value in identity.items()),'"'"'Validation comparison identity mismatch'"'"')
best_hashes={}
for name,path,directory in [('"'"'baseline'"'"',baseline_path,baseline_eval),('"'"'cmerc'"'"',full_path,full_eval)]:
    checkpoint=torch.load(path,map_location='"'"'cpu'"'"'); args=checkpoint['"'"'args'"'"']
    check(args.get('"'"'Dataset'"'"')==dataset and args.get('"'"'seed'"'"')==seed and args.get('"'"'split_seed'"'"')==1475,'"'"'Frozen dataset/training seed mismatch'"'"')
    check(args.get('"'"'causal'"'"') is True and args.get('"'"'selection_protocol'"'"')=='"'"'validation'"'"' and checkpoint['"'"'epoch'"'"']==checkpoint['"'"'best_epoch'"'"'],'"'"'Frozen checkpoint is not validation-selected causal best'"'"')
    metadata=json.loads((Path(directory)/'"'"'metrics.json'"'"').read_text())
    check(all(metadata.get(key)==value for key,value in identity.items()),'"'"'Validation evaluator identity mismatch'"'"')
    check(Path(metadata['"'"'checkpoint'"'"']).resolve()==Path(path).resolve(),'"'"'Validation evaluated a different checkpoint'"'"')
    for filename in ('"'"'metrics.json'"'"','"'"'predictions.npz'"'"'):
        check(comparison['"'"'input_sha256'"'"'][name][filename]==sha256_file(Path(directory)/filename),'"'"'Validation comparison input SHA256 changed'"'"')
    best_hashes[name]=sha256_file(path)
comparison_sha=sha256_file(comparison_path)
if mode=='"'"'create'"'"':
    frozen={**comparison,'"'"'training_seed'"'"':seed,'"'"'validation_comparison_sha256'"'"':comparison_sha,'"'"'best_sha256'"'"':best_hashes,'"'"'frozen_utc'"'"':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with Path(freeze_path).open('"'"'x'"'"') as handle: json.dump(frozen,handle,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps({'"'"'status'"'"':'"'"'frozen'"'"','"'"'validation_freeze_sha256'"'"':sha256_file(freeze_path),'"'"'training_seed'"'"':seed,'"'"'best_sha256'"'"':best_hashes}))
elif mode=='"'"'check'"'"':
    check(len(extra)==1 and sha256_file(freeze_path)==extra[0],'"'"'Frozen validation JSON SHA256 changed'"'"')
    frozen=json.loads(Path(freeze_path).read_text())
    check(all(frozen.get(key)==value for key,value in identity.items()) and frozen.get('"'"'training_seed'"'"')==seed,'"'"'Frozen validation identity mismatch'"'"')
    check(frozen.get('"'"'validation_comparison_sha256'"'"')==comparison_sha and frozen.get('"'"'best_sha256'"'"')==best_hashes,'"'"'Frozen comparison or selected best SHA256 changed'"'"')
    check(frozen.get('"'"'input_sha256'"'"')==comparison.get('"'"'input_sha256'"'"'),'"'"'Frozen validation input identities changed'"'"')
    print(json.dumps({'"'"'status'"'"':'"'"'passed'"'"','"'"'validation_freeze_sha256'"'"':extra[0]}))
else: raise ValueError('"'"'Unknown freeze operation'"'"')' check MELD 2025 "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed2025/comparison_validation/comparison.json "${TEACHER_BEST_MELD_2025:?bind_verified_TEACHER_BEST_MELD_2025}" "${CMERC_BEST_MELD_2025:?bind_verified_CMERC_BEST_MELD_2025}" "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed2025/evaluation/validation/baseline "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed2025/evaluation/validation/full "${VALIDATION_FREEZE_MELD_2025:?bind_verified_VALIDATION_FREEZE_MELD_2025}" "${VALIDATION_FREEZE_SHA_MELD_2025:?bind_verified_VALIDATION_FREEZE_SHA_MELD_2025}" && test ! -e "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed2025/evaluation/test/full && CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u scripts/evaluate_checkpoint.py --dataset MELD --causal --checkpoint "${CMERC_BEST_MELD_2025:?bind_verified_CMERC_BEST_MELD_2025}" --device cuda:0 --split test --split-seed 1475 --batch-size 16 --num-workers 0 --n-bins 15 --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed2025/evaluation/test/full
```

### test_comparison

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import datetime,json,sys,torch
from pathlib import Path
from causal_provenance import sha256_file
mode,dataset,seed,comparison_path,baseline_path,full_path,baseline_eval,full_eval,freeze_path,*extra=sys.argv[1:]
seed=int(seed)
def check(condition,message):
    if not condition: raise ValueError(message)
comparison=json.loads(Path(comparison_path).read_text())
identity=dict(dataset=dataset,split='"'"'validation'"'"',split_seed=1475,replacement_policy='"'"'prefix_mean_v1'"'"')
check(all(comparison.get(key)==value for key,value in identity.items()),'"'"'Validation comparison identity mismatch'"'"')
best_hashes={}
for name,path,directory in [('"'"'baseline'"'"',baseline_path,baseline_eval),('"'"'cmerc'"'"',full_path,full_eval)]:
    checkpoint=torch.load(path,map_location='"'"'cpu'"'"'); args=checkpoint['"'"'args'"'"']
    check(args.get('"'"'Dataset'"'"')==dataset and args.get('"'"'seed'"'"')==seed and args.get('"'"'split_seed'"'"')==1475,'"'"'Frozen dataset/training seed mismatch'"'"')
    check(args.get('"'"'causal'"'"') is True and args.get('"'"'selection_protocol'"'"')=='"'"'validation'"'"' and checkpoint['"'"'epoch'"'"']==checkpoint['"'"'best_epoch'"'"'],'"'"'Frozen checkpoint is not validation-selected causal best'"'"')
    metadata=json.loads((Path(directory)/'"'"'metrics.json'"'"').read_text())
    check(all(metadata.get(key)==value for key,value in identity.items()),'"'"'Validation evaluator identity mismatch'"'"')
    check(Path(metadata['"'"'checkpoint'"'"']).resolve()==Path(path).resolve(),'"'"'Validation evaluated a different checkpoint'"'"')
    for filename in ('"'"'metrics.json'"'"','"'"'predictions.npz'"'"'):
        check(comparison['"'"'input_sha256'"'"'][name][filename]==sha256_file(Path(directory)/filename),'"'"'Validation comparison input SHA256 changed'"'"')
    best_hashes[name]=sha256_file(path)
comparison_sha=sha256_file(comparison_path)
if mode=='"'"'create'"'"':
    frozen={**comparison,'"'"'training_seed'"'"':seed,'"'"'validation_comparison_sha256'"'"':comparison_sha,'"'"'best_sha256'"'"':best_hashes,'"'"'frozen_utc'"'"':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with Path(freeze_path).open('"'"'x'"'"') as handle: json.dump(frozen,handle,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps({'"'"'status'"'"':'"'"'frozen'"'"','"'"'validation_freeze_sha256'"'"':sha256_file(freeze_path),'"'"'training_seed'"'"':seed,'"'"'best_sha256'"'"':best_hashes}))
elif mode=='"'"'check'"'"':
    check(len(extra)==1 and sha256_file(freeze_path)==extra[0],'"'"'Frozen validation JSON SHA256 changed'"'"')
    frozen=json.loads(Path(freeze_path).read_text())
    check(all(frozen.get(key)==value for key,value in identity.items()) and frozen.get('"'"'training_seed'"'"')==seed,'"'"'Frozen validation identity mismatch'"'"')
    check(frozen.get('"'"'validation_comparison_sha256'"'"')==comparison_sha and frozen.get('"'"'best_sha256'"'"')==best_hashes,'"'"'Frozen comparison or selected best SHA256 changed'"'"')
    check(frozen.get('"'"'input_sha256'"'"')==comparison.get('"'"'input_sha256'"'"'),'"'"'Frozen validation input identities changed'"'"')
    print(json.dumps({'"'"'status'"'"':'"'"'passed'"'"','"'"'validation_freeze_sha256'"'"':extra[0]}))
else: raise ValueError('"'"'Unknown freeze operation'"'"')' check MELD 2025 "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed2025/comparison_validation/comparison.json "${TEACHER_BEST_MELD_2025:?bind_verified_TEACHER_BEST_MELD_2025}" "${CMERC_BEST_MELD_2025:?bind_verified_CMERC_BEST_MELD_2025}" "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed2025/evaluation/validation/baseline "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed2025/evaluation/validation/full "${VALIDATION_FREEZE_MELD_2025:?bind_verified_VALIDATION_FREEZE_MELD_2025}" "${VALIDATION_FREEZE_SHA_MELD_2025:?bind_verified_VALIDATION_FREEZE_SHA_MELD_2025}" && "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u scripts/round2_analysis.py --baseline-eval "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed2025/evaluation/test/baseline --cmerc-eval "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed2025/evaluation/test/full --baseline-training "${BASELINE_RUN_MELD_2025:?bind_verified_BASELINE_RUN_MELD_2025}"/metrics.jsonl --cmerc-training "${CMERC_RUN_MELD_2025:?bind_verified_CMERC_RUN_MELD_2025}"/metrics.jsonl --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed2025/comparison_test --validation-freeze "${VALIDATION_FREEZE_MELD_2025:?bind_verified_VALIDATION_FREEZE_MELD_2025}"
```

### extension_without_cl

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u train_cm.py --Dataset MELD --base-model GRU --graph_type hyper --graph_construct direct --multi_modal --mm_fusion_mthd concat_DHT --modals avl --norm LN --num_L 3 --num_K 3 --batch-size 16 --lr 0.0001 --l2 0.00003 --dropout 0.4 --rec-dropout 0.1 --seed 2025 --split-seed 1475 --causal --replacement-policy prefix_mean_v1 --epochs 40 --selection-protocol validation --defer-final-test --fail-on-nonfinite --diagnostic-batches 1 --num-workers 0 --auxiliary-policy legacy_epoch_snapshot --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed2025/without_cl --calibrate --rank_coff 0.002 --contrastlearning --mscl_coff 0.15 --cscl_coff 0.15
```

### extension_without_hscl

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import sys;from causal_provenance import sha256_file;actual=sha256_file(sys.argv[1]);
if actual!=sys.argv[2]: raise ValueError("Stage-B difficulty SHA256 changed before training")' "${DIFFICULTY_MELD_2025:?bind_verified_DIFFICULTY_MELD_2025}" "${DIFFICULTY_SHA_MELD_2025:?bind_verified_DIFFICULTY_SHA_MELD_2025}" && CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u train_cm.py --Dataset MELD --base-model GRU --graph_type hyper --graph_construct direct --multi_modal --mm_fusion_mthd concat_DHT --modals avl --norm LN --num_L 3 --num_K 3 --batch-size 16 --lr 0.0001 --l2 0.00003 --dropout 0.4 --rec-dropout 0.1 --seed 2025 --split-seed 1475 --causal --replacement-policy prefix_mean_v1 --epochs 40 --selection-protocol validation --defer-final-test --fail-on-nonfinite --diagnostic-batches 1 --num-workers 0 --auxiliary-policy legacy_epoch_snapshot --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed2025/without_hscl --calibrate --rank_coff 0.002 --courselearning --scheduler_steps 1 --epoch_ratio 0.4 --difficulty-path "${DIFFICULTY_MELD_2025:?bind_verified_DIFFICULTY_MELD_2025}"
```

### extension_without_cc

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import sys;from causal_provenance import sha256_file;actual=sha256_file(sys.argv[1]);
if actual!=sys.argv[2]: raise ValueError("Stage-B difficulty SHA256 changed before training")' "${DIFFICULTY_MELD_2025:?bind_verified_DIFFICULTY_MELD_2025}" "${DIFFICULTY_SHA_MELD_2025:?bind_verified_DIFFICULTY_SHA_MELD_2025}" && CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u train_cm.py --Dataset MELD --base-model GRU --graph_type hyper --graph_construct direct --multi_modal --mm_fusion_mthd concat_DHT --modals avl --norm LN --num_L 3 --num_K 3 --batch-size 16 --lr 0.0001 --l2 0.00003 --dropout 0.4 --rec-dropout 0.1 --seed 2025 --split-seed 1475 --causal --replacement-policy prefix_mean_v1 --epochs 40 --selection-protocol validation --defer-final-test --fail-on-nonfinite --diagnostic-batches 1 --num-workers 0 --auxiliary-policy legacy_epoch_snapshot --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed2025/without_cc --contrastlearning --mscl_coff 0.15 --cscl_coff 0.15 --courselearning --scheduler_steps 1 --epoch_ratio 0.4 --difficulty-path "${DIFFICULTY_MELD_2025:?bind_verified_DIFFICULTY_MELD_2025}"
```

### extension_same_step_same_mode

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import sys;from causal_provenance import sha256_file;actual=sha256_file(sys.argv[1]);
if actual!=sys.argv[2]: raise ValueError("Stage-B difficulty SHA256 changed before training")' "${DIFFICULTY_MELD_2025:?bind_verified_DIFFICULTY_MELD_2025}" "${DIFFICULTY_SHA_MELD_2025:?bind_verified_DIFFICULTY_SHA_MELD_2025}" && CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u train_cm.py --Dataset MELD --base-model GRU --graph_type hyper --graph_construct direct --multi_modal --mm_fusion_mthd concat_DHT --modals avl --norm LN --num_L 3 --num_K 3 --batch-size 16 --lr 0.0001 --l2 0.00003 --dropout 0.4 --rec-dropout 0.1 --seed 2025 --split-seed 1475 --causal --replacement-policy prefix_mean_v1 --epochs 40 --selection-protocol validation --defer-final-test --fail-on-nonfinite --diagnostic-batches 1 --num-workers 0 --auxiliary-policy same_step_same_mode --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed2025/same_step_same_mode --calibrate --rank_coff 0.002 --contrastlearning --mscl_coff 0.15 --cscl_coff 0.15 --courselearning --scheduler_steps 1 --epoch_ratio 0.4 --difficulty-path "${DIFFICULTY_MELD_2025:?bind_verified_DIFFICULTY_MELD_2025}"
```

按 ordered_stages 执行；comparison 与冻结成功后才执行 test，全部原始产物保留。

## MELD / training seed 3407

### baseline_training

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u train_cm.py --Dataset MELD --base-model GRU --graph_type hyper --graph_construct direct --multi_modal --mm_fusion_mthd concat_DHT --modals avl --norm LN --num_L 3 --num_K 3 --batch-size 16 --lr 0.0001 --l2 0.00003 --dropout 0.4 --rec-dropout 0.1 --seed 3407 --split-seed 1475 --causal --replacement-policy prefix_mean_v1 --epochs 40 --selection-protocol validation --defer-final-test --fail-on-nonfinite --diagnostic-batches 1 --num-workers 0 --auxiliary-policy legacy_epoch_snapshot --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed3407/baseline
```

### strict_teacher_admission

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u scripts/round2_verify_run.py --dataset MELD --seed 3407 --split-seed 1475 --epochs 40 --expected-provenance "${DATA_ENVIRONMENT_MELD_3407:?bind_verified_DATA_ENVIRONMENT_MELD_3407}" --variant baseline --run-dir "${BASELINE_RUN_MELD_3407:?bind_verified_BASELINE_RUN_MELD_3407}" --output "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed3407/verification/baseline.json
```

### difficulty_generation

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u caldiff/cal_diff.py --Dataset MELD --base-model GRU --graph_type hyper --graph_construct direct --multi_modal --mm_fusion_mthd concat_DHT --modals avl --norm LN --num_L 3 --num_K 3 --batch-size 16 --lr 0.0001 --l2 0.00003 --dropout 0.4 --rec-dropout 0.1 --seed 3407 --split-seed 1475 --causal --replacement-policy prefix_mean_v1 --ckpt_path "${TEACHER_BEST_MELD_3407:?bind_verified_TEACHER_BEST_MELD_3407}" --difficulty-output "${DIFFICULTY_MELD_3407:?bind_verified_DIFFICULTY_MELD_3407}"
```

### strict_difficulty_admission

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import json,sys,torch
from pathlib import Path
from causal_provenance import architecture_from_args,resolve_causal_difficulty,sha256_file
dataset,seed,path,teacher_path,inventory_path,baseline_run=sys.argv[1:]
teacher=torch.load(teacher_path,map_location='"'"'cpu'"'"')
if sha256_file(teacher_path)!=sha256_file(Path(baseline_run)/'"'"'best.pt'"'"'): raise ValueError('"'"'Teacher is not the admitted paired baseline best'"'"')
expected=json.loads(Path(inventory_path).read_text())['"'"'data_provenance'"'"']
values,artifact=resolve_causal_difficulty(path,dataset,expected,architecture_from_args(teacher['"'"'args'"'"']),expected_teacher_seed=int(seed))
source=artifact['"'"'manifest'"'"']['"'"'source_checkpoint'"'"']
if source['"'"'sha256'"'"']!=sha256_file(teacher_path) or source['"'"'epoch'"'"']!=source['"'"'best_epoch'"'"']: raise ValueError('"'"'Difficulty teacher is not the admitted baseline best'"'"')
print(json.dumps({'"'"'status'"'"':'"'"'passed'"'"','"'"'difficulty_sha256'"'"':artifact['"'"'sha256'"'"'],'"'"'teacher_sha256'"'"':source['"'"'sha256'"'"'],'"'"'train_count'"'"':len(values)}))' MELD 3407 "${DIFFICULTY_MELD_3407:?bind_verified_DIFFICULTY_MELD_3407}" "${TEACHER_BEST_MELD_3407:?bind_verified_TEACHER_BEST_MELD_3407}" "${DATA_ENVIRONMENT_MELD_3407:?bind_verified_DATA_ENVIRONMENT_MELD_3407}" "${BASELINE_RUN_MELD_3407:?bind_verified_BASELINE_RUN_MELD_3407}"
```

### full_training

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import sys;from causal_provenance import sha256_file;actual=sha256_file(sys.argv[1]);
if actual!=sys.argv[2]: raise ValueError("Stage-B difficulty SHA256 changed before training")' "${DIFFICULTY_MELD_3407:?bind_verified_DIFFICULTY_MELD_3407}" "${DIFFICULTY_SHA_MELD_3407:?bind_verified_DIFFICULTY_SHA_MELD_3407}" && CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u train_cm.py --Dataset MELD --base-model GRU --graph_type hyper --graph_construct direct --multi_modal --mm_fusion_mthd concat_DHT --modals avl --norm LN --num_L 3 --num_K 3 --batch-size 16 --lr 0.0001 --l2 0.00003 --dropout 0.4 --rec-dropout 0.1 --seed 3407 --split-seed 1475 --causal --replacement-policy prefix_mean_v1 --epochs 40 --selection-protocol validation --defer-final-test --fail-on-nonfinite --diagnostic-batches 1 --num-workers 0 --auxiliary-policy legacy_epoch_snapshot --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed3407/full --calibrate --rank_coff 0.002 --contrastlearning --mscl_coff 0.15 --cscl_coff 0.15 --courselearning --scheduler_steps 1 --epoch_ratio 0.4 --difficulty-path "${DIFFICULTY_MELD_3407:?bind_verified_DIFFICULTY_MELD_3407}"
```

### completed_full_and_pair_admission

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u scripts/round2_verify_run.py --dataset MELD --seed 3407 --split-seed 1475 --epochs 40 --expected-provenance "${DATA_ENVIRONMENT_MELD_3407:?bind_verified_DATA_ENVIRONMENT_MELD_3407}" --variant cmerc --run-dir "${CMERC_RUN_MELD_3407:?bind_verified_CMERC_RUN_MELD_3407}" --paired-run "${BASELINE_RUN_MELD_3407:?bind_verified_BASELINE_RUN_MELD_3407}" --difficulty "${DIFFICULTY_MELD_3407:?bind_verified_DIFFICULTY_MELD_3407}" --expected-difficulty-sha256 "${DIFFICULTY_SHA_MELD_3407:?bind_verified_DIFFICULTY_SHA_MELD_3407}" --output "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed3407/verification/full.json
```

### validation_evaluation_baseline

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u scripts/evaluate_checkpoint.py --dataset MELD --causal --checkpoint "${TEACHER_BEST_MELD_3407:?bind_verified_TEACHER_BEST_MELD_3407}" --device cuda:0 --split validation --split-seed 1475 --batch-size 16 --num-workers 0 --n-bins 15 --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed3407/evaluation/validation/baseline
```

### validation_evaluation_full

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u scripts/evaluate_checkpoint.py --dataset MELD --causal --checkpoint "${CMERC_BEST_MELD_3407:?bind_verified_CMERC_BEST_MELD_3407}" --device cuda:0 --split validation --split-seed 1475 --batch-size 16 --num-workers 0 --n-bins 15 --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed3407/evaluation/validation/full
```

### validation_comparison

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u scripts/round2_analysis.py --baseline-eval "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed3407/evaluation/validation/baseline --cmerc-eval "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed3407/evaluation/validation/full --baseline-training "${BASELINE_RUN_MELD_3407:?bind_verified_BASELINE_RUN_MELD_3407}"/metrics.jsonl --cmerc-training "${CMERC_RUN_MELD_3407:?bind_verified_CMERC_RUN_MELD_3407}"/metrics.jsonl --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed3407/comparison_validation
```

### validation_freeze_create

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import datetime,json,sys,torch
from pathlib import Path
from causal_provenance import sha256_file
mode,dataset,seed,comparison_path,baseline_path,full_path,baseline_eval,full_eval,freeze_path,*extra=sys.argv[1:]
seed=int(seed)
def check(condition,message):
    if not condition: raise ValueError(message)
comparison=json.loads(Path(comparison_path).read_text())
identity=dict(dataset=dataset,split='"'"'validation'"'"',split_seed=1475,replacement_policy='"'"'prefix_mean_v1'"'"')
check(all(comparison.get(key)==value for key,value in identity.items()),'"'"'Validation comparison identity mismatch'"'"')
best_hashes={}
for name,path,directory in [('"'"'baseline'"'"',baseline_path,baseline_eval),('"'"'cmerc'"'"',full_path,full_eval)]:
    checkpoint=torch.load(path,map_location='"'"'cpu'"'"'); args=checkpoint['"'"'args'"'"']
    check(args.get('"'"'Dataset'"'"')==dataset and args.get('"'"'seed'"'"')==seed and args.get('"'"'split_seed'"'"')==1475,'"'"'Frozen dataset/training seed mismatch'"'"')
    check(args.get('"'"'causal'"'"') is True and args.get('"'"'selection_protocol'"'"')=='"'"'validation'"'"' and checkpoint['"'"'epoch'"'"']==checkpoint['"'"'best_epoch'"'"'],'"'"'Frozen checkpoint is not validation-selected causal best'"'"')
    metadata=json.loads((Path(directory)/'"'"'metrics.json'"'"').read_text())
    check(all(metadata.get(key)==value for key,value in identity.items()),'"'"'Validation evaluator identity mismatch'"'"')
    check(Path(metadata['"'"'checkpoint'"'"']).resolve()==Path(path).resolve(),'"'"'Validation evaluated a different checkpoint'"'"')
    for filename in ('"'"'metrics.json'"'"','"'"'predictions.npz'"'"'):
        check(comparison['"'"'input_sha256'"'"'][name][filename]==sha256_file(Path(directory)/filename),'"'"'Validation comparison input SHA256 changed'"'"')
    best_hashes[name]=sha256_file(path)
comparison_sha=sha256_file(comparison_path)
if mode=='"'"'create'"'"':
    frozen={**comparison,'"'"'training_seed'"'"':seed,'"'"'validation_comparison_sha256'"'"':comparison_sha,'"'"'best_sha256'"'"':best_hashes,'"'"'frozen_utc'"'"':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with Path(freeze_path).open('"'"'x'"'"') as handle: json.dump(frozen,handle,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps({'"'"'status'"'"':'"'"'frozen'"'"','"'"'validation_freeze_sha256'"'"':sha256_file(freeze_path),'"'"'training_seed'"'"':seed,'"'"'best_sha256'"'"':best_hashes}))
elif mode=='"'"'check'"'"':
    check(len(extra)==1 and sha256_file(freeze_path)==extra[0],'"'"'Frozen validation JSON SHA256 changed'"'"')
    frozen=json.loads(Path(freeze_path).read_text())
    check(all(frozen.get(key)==value for key,value in identity.items()) and frozen.get('"'"'training_seed'"'"')==seed,'"'"'Frozen validation identity mismatch'"'"')
    check(frozen.get('"'"'validation_comparison_sha256'"'"')==comparison_sha and frozen.get('"'"'best_sha256'"'"')==best_hashes,'"'"'Frozen comparison or selected best SHA256 changed'"'"')
    check(frozen.get('"'"'input_sha256'"'"')==comparison.get('"'"'input_sha256'"'"'),'"'"'Frozen validation input identities changed'"'"')
    print(json.dumps({'"'"'status'"'"':'"'"'passed'"'"','"'"'validation_freeze_sha256'"'"':extra[0]}))
else: raise ValueError('"'"'Unknown freeze operation'"'"')' create MELD 3407 "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed3407/comparison_validation/comparison.json "${TEACHER_BEST_MELD_3407:?bind_verified_TEACHER_BEST_MELD_3407}" "${CMERC_BEST_MELD_3407:?bind_verified_CMERC_BEST_MELD_3407}" "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed3407/evaluation/validation/baseline "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed3407/evaluation/validation/full "${VALIDATION_FREEZE_MELD_3407:?bind_verified_VALIDATION_FREEZE_MELD_3407}"
```

### validation_freeze_check

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import datetime,json,sys,torch
from pathlib import Path
from causal_provenance import sha256_file
mode,dataset,seed,comparison_path,baseline_path,full_path,baseline_eval,full_eval,freeze_path,*extra=sys.argv[1:]
seed=int(seed)
def check(condition,message):
    if not condition: raise ValueError(message)
comparison=json.loads(Path(comparison_path).read_text())
identity=dict(dataset=dataset,split='"'"'validation'"'"',split_seed=1475,replacement_policy='"'"'prefix_mean_v1'"'"')
check(all(comparison.get(key)==value for key,value in identity.items()),'"'"'Validation comparison identity mismatch'"'"')
best_hashes={}
for name,path,directory in [('"'"'baseline'"'"',baseline_path,baseline_eval),('"'"'cmerc'"'"',full_path,full_eval)]:
    checkpoint=torch.load(path,map_location='"'"'cpu'"'"'); args=checkpoint['"'"'args'"'"']
    check(args.get('"'"'Dataset'"'"')==dataset and args.get('"'"'seed'"'"')==seed and args.get('"'"'split_seed'"'"')==1475,'"'"'Frozen dataset/training seed mismatch'"'"')
    check(args.get('"'"'causal'"'"') is True and args.get('"'"'selection_protocol'"'"')=='"'"'validation'"'"' and checkpoint['"'"'epoch'"'"']==checkpoint['"'"'best_epoch'"'"'],'"'"'Frozen checkpoint is not validation-selected causal best'"'"')
    metadata=json.loads((Path(directory)/'"'"'metrics.json'"'"').read_text())
    check(all(metadata.get(key)==value for key,value in identity.items()),'"'"'Validation evaluator identity mismatch'"'"')
    check(Path(metadata['"'"'checkpoint'"'"']).resolve()==Path(path).resolve(),'"'"'Validation evaluated a different checkpoint'"'"')
    for filename in ('"'"'metrics.json'"'"','"'"'predictions.npz'"'"'):
        check(comparison['"'"'input_sha256'"'"'][name][filename]==sha256_file(Path(directory)/filename),'"'"'Validation comparison input SHA256 changed'"'"')
    best_hashes[name]=sha256_file(path)
comparison_sha=sha256_file(comparison_path)
if mode=='"'"'create'"'"':
    frozen={**comparison,'"'"'training_seed'"'"':seed,'"'"'validation_comparison_sha256'"'"':comparison_sha,'"'"'best_sha256'"'"':best_hashes,'"'"'frozen_utc'"'"':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with Path(freeze_path).open('"'"'x'"'"') as handle: json.dump(frozen,handle,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps({'"'"'status'"'"':'"'"'frozen'"'"','"'"'validation_freeze_sha256'"'"':sha256_file(freeze_path),'"'"'training_seed'"'"':seed,'"'"'best_sha256'"'"':best_hashes}))
elif mode=='"'"'check'"'"':
    check(len(extra)==1 and sha256_file(freeze_path)==extra[0],'"'"'Frozen validation JSON SHA256 changed'"'"')
    frozen=json.loads(Path(freeze_path).read_text())
    check(all(frozen.get(key)==value for key,value in identity.items()) and frozen.get('"'"'training_seed'"'"')==seed,'"'"'Frozen validation identity mismatch'"'"')
    check(frozen.get('"'"'validation_comparison_sha256'"'"')==comparison_sha and frozen.get('"'"'best_sha256'"'"')==best_hashes,'"'"'Frozen comparison or selected best SHA256 changed'"'"')
    check(frozen.get('"'"'input_sha256'"'"')==comparison.get('"'"'input_sha256'"'"'),'"'"'Frozen validation input identities changed'"'"')
    print(json.dumps({'"'"'status'"'"':'"'"'passed'"'"','"'"'validation_freeze_sha256'"'"':extra[0]}))
else: raise ValueError('"'"'Unknown freeze operation'"'"')' check MELD 3407 "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed3407/comparison_validation/comparison.json "${TEACHER_BEST_MELD_3407:?bind_verified_TEACHER_BEST_MELD_3407}" "${CMERC_BEST_MELD_3407:?bind_verified_CMERC_BEST_MELD_3407}" "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed3407/evaluation/validation/baseline "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed3407/evaluation/validation/full "${VALIDATION_FREEZE_MELD_3407:?bind_verified_VALIDATION_FREEZE_MELD_3407}" "${VALIDATION_FREEZE_SHA_MELD_3407:?bind_verified_VALIDATION_FREEZE_SHA_MELD_3407}"
```

### test_evaluation_baseline

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import datetime,json,sys,torch
from pathlib import Path
from causal_provenance import sha256_file
mode,dataset,seed,comparison_path,baseline_path,full_path,baseline_eval,full_eval,freeze_path,*extra=sys.argv[1:]
seed=int(seed)
def check(condition,message):
    if not condition: raise ValueError(message)
comparison=json.loads(Path(comparison_path).read_text())
identity=dict(dataset=dataset,split='"'"'validation'"'"',split_seed=1475,replacement_policy='"'"'prefix_mean_v1'"'"')
check(all(comparison.get(key)==value for key,value in identity.items()),'"'"'Validation comparison identity mismatch'"'"')
best_hashes={}
for name,path,directory in [('"'"'baseline'"'"',baseline_path,baseline_eval),('"'"'cmerc'"'"',full_path,full_eval)]:
    checkpoint=torch.load(path,map_location='"'"'cpu'"'"'); args=checkpoint['"'"'args'"'"']
    check(args.get('"'"'Dataset'"'"')==dataset and args.get('"'"'seed'"'"')==seed and args.get('"'"'split_seed'"'"')==1475,'"'"'Frozen dataset/training seed mismatch'"'"')
    check(args.get('"'"'causal'"'"') is True and args.get('"'"'selection_protocol'"'"')=='"'"'validation'"'"' and checkpoint['"'"'epoch'"'"']==checkpoint['"'"'best_epoch'"'"'],'"'"'Frozen checkpoint is not validation-selected causal best'"'"')
    metadata=json.loads((Path(directory)/'"'"'metrics.json'"'"').read_text())
    check(all(metadata.get(key)==value for key,value in identity.items()),'"'"'Validation evaluator identity mismatch'"'"')
    check(Path(metadata['"'"'checkpoint'"'"']).resolve()==Path(path).resolve(),'"'"'Validation evaluated a different checkpoint'"'"')
    for filename in ('"'"'metrics.json'"'"','"'"'predictions.npz'"'"'):
        check(comparison['"'"'input_sha256'"'"'][name][filename]==sha256_file(Path(directory)/filename),'"'"'Validation comparison input SHA256 changed'"'"')
    best_hashes[name]=sha256_file(path)
comparison_sha=sha256_file(comparison_path)
if mode=='"'"'create'"'"':
    frozen={**comparison,'"'"'training_seed'"'"':seed,'"'"'validation_comparison_sha256'"'"':comparison_sha,'"'"'best_sha256'"'"':best_hashes,'"'"'frozen_utc'"'"':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with Path(freeze_path).open('"'"'x'"'"') as handle: json.dump(frozen,handle,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps({'"'"'status'"'"':'"'"'frozen'"'"','"'"'validation_freeze_sha256'"'"':sha256_file(freeze_path),'"'"'training_seed'"'"':seed,'"'"'best_sha256'"'"':best_hashes}))
elif mode=='"'"'check'"'"':
    check(len(extra)==1 and sha256_file(freeze_path)==extra[0],'"'"'Frozen validation JSON SHA256 changed'"'"')
    frozen=json.loads(Path(freeze_path).read_text())
    check(all(frozen.get(key)==value for key,value in identity.items()) and frozen.get('"'"'training_seed'"'"')==seed,'"'"'Frozen validation identity mismatch'"'"')
    check(frozen.get('"'"'validation_comparison_sha256'"'"')==comparison_sha and frozen.get('"'"'best_sha256'"'"')==best_hashes,'"'"'Frozen comparison or selected best SHA256 changed'"'"')
    check(frozen.get('"'"'input_sha256'"'"')==comparison.get('"'"'input_sha256'"'"'),'"'"'Frozen validation input identities changed'"'"')
    print(json.dumps({'"'"'status'"'"':'"'"'passed'"'"','"'"'validation_freeze_sha256'"'"':extra[0]}))
else: raise ValueError('"'"'Unknown freeze operation'"'"')' check MELD 3407 "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed3407/comparison_validation/comparison.json "${TEACHER_BEST_MELD_3407:?bind_verified_TEACHER_BEST_MELD_3407}" "${CMERC_BEST_MELD_3407:?bind_verified_CMERC_BEST_MELD_3407}" "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed3407/evaluation/validation/baseline "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed3407/evaluation/validation/full "${VALIDATION_FREEZE_MELD_3407:?bind_verified_VALIDATION_FREEZE_MELD_3407}" "${VALIDATION_FREEZE_SHA_MELD_3407:?bind_verified_VALIDATION_FREEZE_SHA_MELD_3407}" && test ! -e "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed3407/evaluation/test/baseline && CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u scripts/evaluate_checkpoint.py --dataset MELD --causal --checkpoint "${TEACHER_BEST_MELD_3407:?bind_verified_TEACHER_BEST_MELD_3407}" --device cuda:0 --split test --split-seed 1475 --batch-size 16 --num-workers 0 --n-bins 15 --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed3407/evaluation/test/baseline
```

### test_evaluation_full

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import datetime,json,sys,torch
from pathlib import Path
from causal_provenance import sha256_file
mode,dataset,seed,comparison_path,baseline_path,full_path,baseline_eval,full_eval,freeze_path,*extra=sys.argv[1:]
seed=int(seed)
def check(condition,message):
    if not condition: raise ValueError(message)
comparison=json.loads(Path(comparison_path).read_text())
identity=dict(dataset=dataset,split='"'"'validation'"'"',split_seed=1475,replacement_policy='"'"'prefix_mean_v1'"'"')
check(all(comparison.get(key)==value for key,value in identity.items()),'"'"'Validation comparison identity mismatch'"'"')
best_hashes={}
for name,path,directory in [('"'"'baseline'"'"',baseline_path,baseline_eval),('"'"'cmerc'"'"',full_path,full_eval)]:
    checkpoint=torch.load(path,map_location='"'"'cpu'"'"'); args=checkpoint['"'"'args'"'"']
    check(args.get('"'"'Dataset'"'"')==dataset and args.get('"'"'seed'"'"')==seed and args.get('"'"'split_seed'"'"')==1475,'"'"'Frozen dataset/training seed mismatch'"'"')
    check(args.get('"'"'causal'"'"') is True and args.get('"'"'selection_protocol'"'"')=='"'"'validation'"'"' and checkpoint['"'"'epoch'"'"']==checkpoint['"'"'best_epoch'"'"'],'"'"'Frozen checkpoint is not validation-selected causal best'"'"')
    metadata=json.loads((Path(directory)/'"'"'metrics.json'"'"').read_text())
    check(all(metadata.get(key)==value for key,value in identity.items()),'"'"'Validation evaluator identity mismatch'"'"')
    check(Path(metadata['"'"'checkpoint'"'"']).resolve()==Path(path).resolve(),'"'"'Validation evaluated a different checkpoint'"'"')
    for filename in ('"'"'metrics.json'"'"','"'"'predictions.npz'"'"'):
        check(comparison['"'"'input_sha256'"'"'][name][filename]==sha256_file(Path(directory)/filename),'"'"'Validation comparison input SHA256 changed'"'"')
    best_hashes[name]=sha256_file(path)
comparison_sha=sha256_file(comparison_path)
if mode=='"'"'create'"'"':
    frozen={**comparison,'"'"'training_seed'"'"':seed,'"'"'validation_comparison_sha256'"'"':comparison_sha,'"'"'best_sha256'"'"':best_hashes,'"'"'frozen_utc'"'"':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with Path(freeze_path).open('"'"'x'"'"') as handle: json.dump(frozen,handle,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps({'"'"'status'"'"':'"'"'frozen'"'"','"'"'validation_freeze_sha256'"'"':sha256_file(freeze_path),'"'"'training_seed'"'"':seed,'"'"'best_sha256'"'"':best_hashes}))
elif mode=='"'"'check'"'"':
    check(len(extra)==1 and sha256_file(freeze_path)==extra[0],'"'"'Frozen validation JSON SHA256 changed'"'"')
    frozen=json.loads(Path(freeze_path).read_text())
    check(all(frozen.get(key)==value for key,value in identity.items()) and frozen.get('"'"'training_seed'"'"')==seed,'"'"'Frozen validation identity mismatch'"'"')
    check(frozen.get('"'"'validation_comparison_sha256'"'"')==comparison_sha and frozen.get('"'"'best_sha256'"'"')==best_hashes,'"'"'Frozen comparison or selected best SHA256 changed'"'"')
    check(frozen.get('"'"'input_sha256'"'"')==comparison.get('"'"'input_sha256'"'"'),'"'"'Frozen validation input identities changed'"'"')
    print(json.dumps({'"'"'status'"'"':'"'"'passed'"'"','"'"'validation_freeze_sha256'"'"':extra[0]}))
else: raise ValueError('"'"'Unknown freeze operation'"'"')' check MELD 3407 "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed3407/comparison_validation/comparison.json "${TEACHER_BEST_MELD_3407:?bind_verified_TEACHER_BEST_MELD_3407}" "${CMERC_BEST_MELD_3407:?bind_verified_CMERC_BEST_MELD_3407}" "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed3407/evaluation/validation/baseline "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed3407/evaluation/validation/full "${VALIDATION_FREEZE_MELD_3407:?bind_verified_VALIDATION_FREEZE_MELD_3407}" "${VALIDATION_FREEZE_SHA_MELD_3407:?bind_verified_VALIDATION_FREEZE_SHA_MELD_3407}" && test ! -e "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed3407/evaluation/test/full && CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u scripts/evaluate_checkpoint.py --dataset MELD --causal --checkpoint "${CMERC_BEST_MELD_3407:?bind_verified_CMERC_BEST_MELD_3407}" --device cuda:0 --split test --split-seed 1475 --batch-size 16 --num-workers 0 --n-bins 15 --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed3407/evaluation/test/full
```

### test_comparison

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import datetime,json,sys,torch
from pathlib import Path
from causal_provenance import sha256_file
mode,dataset,seed,comparison_path,baseline_path,full_path,baseline_eval,full_eval,freeze_path,*extra=sys.argv[1:]
seed=int(seed)
def check(condition,message):
    if not condition: raise ValueError(message)
comparison=json.loads(Path(comparison_path).read_text())
identity=dict(dataset=dataset,split='"'"'validation'"'"',split_seed=1475,replacement_policy='"'"'prefix_mean_v1'"'"')
check(all(comparison.get(key)==value for key,value in identity.items()),'"'"'Validation comparison identity mismatch'"'"')
best_hashes={}
for name,path,directory in [('"'"'baseline'"'"',baseline_path,baseline_eval),('"'"'cmerc'"'"',full_path,full_eval)]:
    checkpoint=torch.load(path,map_location='"'"'cpu'"'"'); args=checkpoint['"'"'args'"'"']
    check(args.get('"'"'Dataset'"'"')==dataset and args.get('"'"'seed'"'"')==seed and args.get('"'"'split_seed'"'"')==1475,'"'"'Frozen dataset/training seed mismatch'"'"')
    check(args.get('"'"'causal'"'"') is True and args.get('"'"'selection_protocol'"'"')=='"'"'validation'"'"' and checkpoint['"'"'epoch'"'"']==checkpoint['"'"'best_epoch'"'"'],'"'"'Frozen checkpoint is not validation-selected causal best'"'"')
    metadata=json.loads((Path(directory)/'"'"'metrics.json'"'"').read_text())
    check(all(metadata.get(key)==value for key,value in identity.items()),'"'"'Validation evaluator identity mismatch'"'"')
    check(Path(metadata['"'"'checkpoint'"'"']).resolve()==Path(path).resolve(),'"'"'Validation evaluated a different checkpoint'"'"')
    for filename in ('"'"'metrics.json'"'"','"'"'predictions.npz'"'"'):
        check(comparison['"'"'input_sha256'"'"'][name][filename]==sha256_file(Path(directory)/filename),'"'"'Validation comparison input SHA256 changed'"'"')
    best_hashes[name]=sha256_file(path)
comparison_sha=sha256_file(comparison_path)
if mode=='"'"'create'"'"':
    frozen={**comparison,'"'"'training_seed'"'"':seed,'"'"'validation_comparison_sha256'"'"':comparison_sha,'"'"'best_sha256'"'"':best_hashes,'"'"'frozen_utc'"'"':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with Path(freeze_path).open('"'"'x'"'"') as handle: json.dump(frozen,handle,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps({'"'"'status'"'"':'"'"'frozen'"'"','"'"'validation_freeze_sha256'"'"':sha256_file(freeze_path),'"'"'training_seed'"'"':seed,'"'"'best_sha256'"'"':best_hashes}))
elif mode=='"'"'check'"'"':
    check(len(extra)==1 and sha256_file(freeze_path)==extra[0],'"'"'Frozen validation JSON SHA256 changed'"'"')
    frozen=json.loads(Path(freeze_path).read_text())
    check(all(frozen.get(key)==value for key,value in identity.items()) and frozen.get('"'"'training_seed'"'"')==seed,'"'"'Frozen validation identity mismatch'"'"')
    check(frozen.get('"'"'validation_comparison_sha256'"'"')==comparison_sha and frozen.get('"'"'best_sha256'"'"')==best_hashes,'"'"'Frozen comparison or selected best SHA256 changed'"'"')
    check(frozen.get('"'"'input_sha256'"'"')==comparison.get('"'"'input_sha256'"'"'),'"'"'Frozen validation input identities changed'"'"')
    print(json.dumps({'"'"'status'"'"':'"'"'passed'"'"','"'"'validation_freeze_sha256'"'"':extra[0]}))
else: raise ValueError('"'"'Unknown freeze operation'"'"')' check MELD 3407 "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed3407/comparison_validation/comparison.json "${TEACHER_BEST_MELD_3407:?bind_verified_TEACHER_BEST_MELD_3407}" "${CMERC_BEST_MELD_3407:?bind_verified_CMERC_BEST_MELD_3407}" "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed3407/evaluation/validation/baseline "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed3407/evaluation/validation/full "${VALIDATION_FREEZE_MELD_3407:?bind_verified_VALIDATION_FREEZE_MELD_3407}" "${VALIDATION_FREEZE_SHA_MELD_3407:?bind_verified_VALIDATION_FREEZE_SHA_MELD_3407}" && "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u scripts/round2_analysis.py --baseline-eval "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed3407/evaluation/test/baseline --cmerc-eval "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed3407/evaluation/test/full --baseline-training "${BASELINE_RUN_MELD_3407:?bind_verified_BASELINE_RUN_MELD_3407}"/metrics.jsonl --cmerc-training "${CMERC_RUN_MELD_3407:?bind_verified_CMERC_RUN_MELD_3407}"/metrics.jsonl --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed3407/comparison_test --validation-freeze "${VALIDATION_FREEZE_MELD_3407:?bind_verified_VALIDATION_FREEZE_MELD_3407}"
```

### extension_without_cl

```bash
CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u train_cm.py --Dataset MELD --base-model GRU --graph_type hyper --graph_construct direct --multi_modal --mm_fusion_mthd concat_DHT --modals avl --norm LN --num_L 3 --num_K 3 --batch-size 16 --lr 0.0001 --l2 0.00003 --dropout 0.4 --rec-dropout 0.1 --seed 3407 --split-seed 1475 --causal --replacement-policy prefix_mean_v1 --epochs 40 --selection-protocol validation --defer-final-test --fail-on-nonfinite --diagnostic-batches 1 --num-workers 0 --auxiliary-policy legacy_epoch_snapshot --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed3407/without_cl --calibrate --rank_coff 0.002 --contrastlearning --mscl_coff 0.15 --cscl_coff 0.15
```

### extension_without_hscl

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import sys;from causal_provenance import sha256_file;actual=sha256_file(sys.argv[1]);
if actual!=sys.argv[2]: raise ValueError("Stage-B difficulty SHA256 changed before training")' "${DIFFICULTY_MELD_3407:?bind_verified_DIFFICULTY_MELD_3407}" "${DIFFICULTY_SHA_MELD_3407:?bind_verified_DIFFICULTY_SHA_MELD_3407}" && CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u train_cm.py --Dataset MELD --base-model GRU --graph_type hyper --graph_construct direct --multi_modal --mm_fusion_mthd concat_DHT --modals avl --norm LN --num_L 3 --num_K 3 --batch-size 16 --lr 0.0001 --l2 0.00003 --dropout 0.4 --rec-dropout 0.1 --seed 3407 --split-seed 1475 --causal --replacement-policy prefix_mean_v1 --epochs 40 --selection-protocol validation --defer-final-test --fail-on-nonfinite --diagnostic-batches 1 --num-workers 0 --auxiliary-policy legacy_epoch_snapshot --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed3407/without_hscl --calibrate --rank_coff 0.002 --courselearning --scheduler_steps 1 --epoch_ratio 0.4 --difficulty-path "${DIFFICULTY_MELD_3407:?bind_verified_DIFFICULTY_MELD_3407}"
```

### extension_without_cc

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import sys;from causal_provenance import sha256_file;actual=sha256_file(sys.argv[1]);
if actual!=sys.argv[2]: raise ValueError("Stage-B difficulty SHA256 changed before training")' "${DIFFICULTY_MELD_3407:?bind_verified_DIFFICULTY_MELD_3407}" "${DIFFICULTY_SHA_MELD_3407:?bind_verified_DIFFICULTY_SHA_MELD_3407}" && CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u train_cm.py --Dataset MELD --base-model GRU --graph_type hyper --graph_construct direct --multi_modal --mm_fusion_mthd concat_DHT --modals avl --norm LN --num_L 3 --num_K 3 --batch-size 16 --lr 0.0001 --l2 0.00003 --dropout 0.4 --rec-dropout 0.1 --seed 3407 --split-seed 1475 --causal --replacement-policy prefix_mean_v1 --epochs 40 --selection-protocol validation --defer-final-test --fail-on-nonfinite --diagnostic-batches 1 --num-workers 0 --auxiliary-policy legacy_epoch_snapshot --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed3407/without_cc --contrastlearning --mscl_coff 0.15 --cscl_coff 0.15 --courselearning --scheduler_steps 1 --epoch_ratio 0.4 --difficulty-path "${DIFFICULTY_MELD_3407:?bind_verified_DIFFICULTY_MELD_3407}"
```

### extension_same_step_same_mode

```bash
"${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u -c 'import sys;from causal_provenance import sha256_file;actual=sha256_file(sys.argv[1]);
if actual!=sys.argv[2]: raise ValueError("Stage-B difficulty SHA256 changed before training")' "${DIFFICULTY_MELD_3407:?bind_verified_DIFFICULTY_MELD_3407}" "${DIFFICULTY_SHA_MELD_3407:?bind_verified_DIFFICULTY_SHA_MELD_3407}" && CUDA_VISIBLE_DEVICES="${GPU_UUID:?recheck_health_and_free_capacity_then_bind_GPU_UUID}" "${CMERC_PYTHON:?bind_verified_CMERC_PYTHON}" -u train_cm.py --Dataset MELD --base-model GRU --graph_type hyper --graph_construct direct --multi_modal --mm_fusion_mthd concat_DHT --modals avl --norm LN --num_L 3 --num_K 3 --batch-size 16 --lr 0.0001 --l2 0.00003 --dropout 0.4 --rec-dropout 0.1 --seed 3407 --split-seed 1475 --causal --replacement-policy prefix_mean_v1 --epochs 40 --selection-protocol validation --defer-final-test --fail-on-nonfinite --diagnostic-batches 1 --num-workers 0 --auxiliary-policy same_step_same_mode --output-dir "${FUTURE_OUTPUT_ROOT:?bind_verified_FUTURE_OUTPUT_ROOT}"/MELD/seed3407/same_step_same_mode --calibrate --rank_coff 0.002 --contrastlearning --mscl_coff 0.15 --cscl_coff 0.15 --courselearning --scheduler_steps 1 --epoch_ratio 0.4 --difficulty-path "${DIFFICULTY_MELD_3407:?bind_verified_DIFFICULTY_MELD_3407}"
```

按 ordered_stages 执行；comparison 与冻结成功后才执行 test，全部原始产物保留。
