"""XP-only durable 20-run queue, acceptance, backup/export and registration."""
import json,os,subprocess,time
from pathlib import Path
B=Path(__file__).resolve().parents[1];CODE=B/'code';CMERC=B.parents[2]
PYTHON='/data2/yb/conda_envs/cmerc-xp-repro/bin/python'
state={'status':'running','pid':os.getpid(),'started_at':time.time(),'phases':{}}
def save():
 p=B/'state/pipeline.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(p)
def execute(name,command):
 state['phase']=name;entry={'command':command,'started_at':time.time(),'status':'running'};state['phases'][name]=entry;save()
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',CMERC_ENV_PREFIX='/data2/yb/conda_envs/cmerc-xp-repro',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 for key in ['CMERC_GRADIENT_PROBE','CMERC_STOP_AFTER_FIRST_BEST','CMERC_GPU_RESUME_TEST']:env.pop(key,None)
 env['LD_LIBRARY_PATH']='/data2/yb/conda_envs/cmerc-xp-repro/lib/python3.10/site-packages/nvidia/cusparse/lib'+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
 log=B/'logs'/f'{name}_supervisor.log'
 with log.open('x') as h:
  child=subprocess.Popen(command,cwd=CODE,env=env,stdin=subprocess.DEVNULL,stdout=h,stderr=subprocess.STDOUT,start_new_session=True)
  entry['pid']=child.pid;entry['log']=str(log);save();rc=child.wait()
 entry.update(status='completed' if rc==0 else 'failed',exit_code=rc,finished_at=time.time());save()
 if rc:raise RuntimeError(f'{name} failed exit{rc}; see {log}')
def register():
 acceptance=json.loads((B/'results/acceptance.json').read_text());assert acceptance['status']=='passed'
 assert (CMERC/'experiment_records/04_xp_mainline_validation/README.md').exists()
 original=(B/'BASELINE_REGISTRY.md').read_text()
 target=CMERC/'BASELINE_REGISTRY.md'
 if target.exists():
  if str(B) in target.read_text():return
  with target.open('a') as h:h.write('\n\n'+original)
 else:target.write_text(original)
 index=CMERC/'experiment_records/README.md';text=index.read_text()
 if '04_xp_mainline_validation' not in text:
  marker='| V100 多 seed V2'
  rows=text.splitlines();out=[]
  for line in rows:
   out.append(line)
   if line.startswith(marker):out.append('| XP 主研究基线 | 本地 `user23` / TITAN Xp | 5 seed × 2 数据集 × baseline/CMERC，共20 run；validation/continuous/ATen/capacity_safe | 已完成并通过验收 | [04_xp_mainline_validation](04_xp_mainline_validation/README.md) |')
  text='\n'.join(out)+'\n'
  text=text.replace('# 最近三组 CMERC 实验档案','# CMERC 实验档案',1)
  text+='\n## 当前主研究基线\n\n04 为 XP 主研究基线：非因果、验证集选模、固定外部课程、五 seed 配对。20项完整性与协议验收通过后登记，分数不作为登记门槛。历史01/02/03原目录不变；不同协议/环境的统计分别展示。大型权重、逐样本预测和原始日志保留原批次及独立备份，公开04仅提供代码、配置和审查材料。\n'
  index.write_text(text)
 state['registration']={'status':'completed','root_registry':str(target),'public_archive':str(CMERC/'experiment_records/04_xp_mainline_validation'),'completed_at':time.time()};save()
try:
 assert json.loads((B/'preflight/status.json').read_text())['status']=='passed'
 execute('formal',[PYTHON,str(CODE/'scripts/multiseed_scheduler.py'),'--config',str(B/'configs/xp_mainline_validation.json'),'--run-root',str(B/'runs'),'--log-root',str(B/'logs'),'--result-root',str(B/'results'),'--state',str(B/'state/queue.json')])
 execute('acceptance',[PYTHON,str(B/'tools/finalize_mainline.py')])
 execute('backup_export',[PYTHON,str(B/'tools/export_mainline.py')])
 register()
 state.update(status='completed',phase='completed',finished_at=time.time());save()
except BaseException as error:
 state.update(status='failed',error=repr(error),finished_at=time.time());save();raise
print(json.dumps({'status':state['status'],'batch_root':str(B)}),flush=True)
