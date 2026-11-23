"""Strict streaming projection: only prelisted role IDs/counts and lineage metadata.

Unknown JSON subtrees and unselected CSV fields are skipped without parsing their
values. Never returns raw text, emotion labels, class support or mask-row values.
"""
import csv
import json
from collections import Counter
from pathlib import Path

RUN=Path(__file__).resolve().parents[2]
WORK=next(p for p in RUN.parents if (p/'CMERC').is_dir())
BASE=WORK/'experiments/xp_caldist_behavioral_calibration'
FORMAL=BASE/'formal_fivefold_fiveseed/20261109T174024Z'
LINEAGE=BASE/'independent_calibration_protocol/20261109T100307Z/results/sample_lineage.csv'
ROLE_FIELDS={'dialogue_ids','dialogue_count','utterance_count','supervision_eligible_count','unknown_masked_count','group_ids'}
META_FIELDS={'dataset','dialogue_id','session_id','episode_id','original_split',
 'used_for_original_model_training','used_for_checkpoint_selection','used_for_calibration_development',
 'previously_evaluated_as_test','not_previously_evaluated','historical_exposure'}


class Reader:
    def __init__(self,stream):self.f=stream;self.pending=''
    def take(self):
        if self.pending:
            c=self.pending;self.pending='';return c
        return self.f.read(1)
    def unget(self,c):
        if self.pending:raise ValueError('Reader pushback conflict')
        self.pending=c
    def nonspace(self):
        c=self.take()
        while c and c.isspace():c=self.take()
        return c
    def string(self,capture):
        values=['"'] if capture else None
        escaped=False
        while True:
            c=self.take()
            if not c:raise ValueError('Unclosed JSON string')
            if capture:values.append(c)
            if escaped:escaped=False
            elif c=='\\':escaped=True
            elif c=='"':return ''.join(values) if capture else None
    def value(self,capture):
        c=self.nonspace()
        if c=='"':return self.string(capture)
        if c in '{[':
            stack=[c];pieces=[c] if capture else None
            while stack:
                c=self.take()
                if not c:raise ValueError('Unclosed JSON subtree')
                if c=='"':
                    text=self.string(capture)
                    if capture:pieces.append(text)
                    continue
                if capture:pieces.append(c)
                if c in '{[':stack.append(c)
                elif c in '}]':stack.pop()
            return ''.join(pieces) if capture else None
        pieces=[c] if capture else None
        while True:
            c=self.take()
            if not c or c in ',}':
                self.unget(c);break
            if capture:pieces.append(c)
        return ''.join(pieces) if capture else None


def keep(path,key):
    if not path:return 'object' if key=='protocols' else None
    if path==('protocols',):return 'object' if key in ('IEMOCAP','MELD') else None
    if len(path)==2:return 'object' if key=='folds' else 'leaf' if key=='group_unit' else None
    if len(path)==3:return 'object' if key in ('fold1','fold2','fold3','fold4','fold5') else None
    if len(path)==4:return 'object' if key=='roles' else None
    if len(path)==5:return 'object' if key in ('training','development','calibration','evaluation') else None
    if len(path)==6:return 'leaf' if key in ROLE_FIELDS else None
    return None


def project_json(path):
    with path.open() as f:
        reader=Reader(f)
        def obj(location):
            if reader.nonspace()!='{':raise ValueError('Expected metadata object')
            result={}
            while True:
                c=reader.nonspace()
                if c=='}':return result
                if c!='"':raise ValueError('Expected metadata key')
                key=json.loads(reader.string(True))
                if reader.nonspace()!=':':raise ValueError('Expected colon')
                mode=keep(location,key)
                if mode=='object':result[key]=obj(location+(key,))
                elif mode=='leaf':result[key]=json.loads(reader.value(True))
                else:reader.value(False)
                c=reader.nonspace()
                if c=='}':return result
                if c!=',':raise ValueError('Expected comma')
        return obj(())


def project_csv(path,columns):
    """Skip unselected fields while scanning; never construct a full data row."""
    with path.open(newline='') as f:
        header=next(csv.reader([f.readline()]))
        indices={i:key for i,key in enumerate(header) if key in columns}
        index=0;pieces=[];record={};quoted=False;pending=''
        def commit():
            nonlocal pieces
            if index in indices:record[indices[index]]=''.join(pieces)
            pieces=[]
        while True:
            c=pending or f.read(1);pending=''
            if not c:
                if record or pieces or index:
                    commit();yield record
                return
            if c=='"':
                if quoted:
                    following=f.read(1)
                    if following=='"':
                        if index in indices:pieces.append('"')
                    else:quoted=False;pending=following
                else:quoted=True
            elif c==',' and not quoted:
                commit();index+=1
            elif c in '\r\n' and not quoted:
                if c=='\r':
                    following=f.read(1)
                    if following!='\n':pending=following
                commit()
                if index+1!=len(header):raise ValueError('CSV metadata shape mismatch')
                yield record
                record={};index=0
            elif index in indices:pieces.append(c)


def main():
    output=RUN/'results/exposure'
    approved=project_json(FORMAL/'configs/approved_split_manifests.json')['protocols']
    # Only decoded metadata fields are retained; forbidden label field never enters records.
    lineage=list(project_csv(LINEAGE,META_FIELDS))
    assert all(set(r)<=META_FIELDS for r in lineage)
    by_dataset={}
    for r in lineage:
        d=by_dataset.setdefault(r['dataset'],{})
        did=r['dialogue_id']
        if did in d and d[did]['original_split']!=r['original_split']:
            raise ValueError('A lineage Dialogue has ambiguous source split')
        d[did]=r
    folded=[];rotation=[];aliases=[]
    for dataset,p in approved.items():
        eval_union=set();eval_counts=Counter();calibration_union=set()
        for fold,entry in p['folds'].items():
            role_sets={role:set(meta['dialogue_ids']) for role,meta in entry['roles'].items()}
            assert sum(map(len,role_sets.values()))==len(set.union(*role_sets.values()))
            for role,meta in entry['roles'].items():
                original=Counter()
                for did in meta['dialogue_ids']:
                    original[by_dataset[dataset].get(did,{}).get('original_split','UNVERIFIED')]+=1
                folded.append({'dataset':dataset,'fold':fold,'role':role,'dialogues':len(role_sets[role]),
                    'utterances':meta.get('utterance_count'),'metadata_count_matches':len(role_sets[role])==meta.get('dialogue_count'),
                    'original_split_dialogue_counts':dict(original)})
                if role=='evaluation':
                    aliases.append({'dataset':dataset,'fold':fold,'original_split_dialogue_counts':dict(original)})
                    for did in role_sets[role]:eval_counts[did]+=1
                    eval_union|=role_sets[role]
                if role=='calibration':calibration_union|=role_sets[role]
            eval_ids=role_sets['evaluation'];other_roles={r:set() for r in ('training','development','calibration')}
            for other,other_entry in p['folds'].items():
                if other==fold:continue
                for role in other_roles:other_roles[role]|=set(other_entry['roles'][role]['dialogue_ids'])
            rotation.append({'dataset':dataset,'fold':fold,'evaluation_dialogues':len(eval_ids),
                'evaluation_in_other_fold_training':len(eval_ids&other_roles['training']),
                'evaluation_in_other_fold_development':len(eval_ids&other_roles['development']),
                'evaluation_in_other_fold_train_or_dev':len(eval_ids&(other_roles['training']|other_roles['development'])),
                'evaluation_in_other_fold_calibration':len(eval_ids&other_roles['calibration']),
                'same_fold_disjoint':not any(eval_ids&role_sets[r] for r in ('training','development','calibration'))})
        assert all(n==1 for n in eval_counts.values())
        split_orig=Counter(by_dataset[dataset].get(did,{}).get('original_split','UNVERIFIED') for did in eval_union)
        flagcounts={key:sum(str(by_dataset[dataset].get(did,{}).get(key,'')).lower()=='true' for did in eval_union)
            for key in ('used_for_original_model_training','used_for_checkpoint_selection','used_for_calibration_development','previously_evaluated_as_test','not_previously_evaluated')}
        output_summary={'evaluation_union_dialogues':len(eval_union),'calibration_union_dialogues':len(calibration_union),
            'eval_union_in_any_calibration':len(eval_union&calibration_union),'eval_original_split_counts':dict(split_orig),
            'lineage_exposure_true_dialogue_counts':flagcounts,'lineage_mapping_unverified':split_orig.get('UNVERIFIED',0)}
        by_dataset[dataset]={'summary':output_summary,'lineage_ids':by_dataset[dataset]}
    summary={ds:data['summary'] for ds,data in by_dataset.items()}
    (output/'role_counts.json').write_text(json.dumps(folded,indent=2)+'\n')
    (output/'fold_rotation_exposure.json').write_text(json.dumps(rotation,indent=2)+'\n')
    (output/'eval_alias_counts.json').write_text(json.dumps(aliases,indent=2)+'\n')
    (output/'exposure_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    # Source build_protocol_audit.py explicitly defines MELD_{split}_{raw_id:04d}.
    # Plan B's manifest supplies actual feature-ID -> raw-ID metadata. No guessed offsets.
    bridges={};pilot_dialogues={}
    plan_b_legacy=BASE/'plan_b_raw_input_pilot/20261120T143131Z'
    for ds in ('IEMOCAP','MELD'):
        bridge={}
        data_manifest=plan_b_legacy/'local_only/data'/ds/'manifest.csv'
        for row in project_csv(data_manifest,{'dataset','split','dialogue_id','raw_dialogue_id','inner_subset'}):
            formal_id=row['dialogue_id'] if ds=='IEMOCAP' else f"MELD_{row['split']}_{int(row['raw_dialogue_id']):04d}"
            record={'split':row['split'],'feature_dialogue_id':row['dialogue_id'],'inner_subset':row['inner_subset']}
            if formal_id in bridge and bridge[formal_id]!=record:raise ValueError('Ambiguous metadata-only source bridge')
            bridge[formal_id]=record
        bridges[ds]=bridge
        feature_to_formal={r['feature_dialogue_id']:did for did,r in bridge.items() if r['split']=='validation'}
        pilot_rows=list(project_csv(plan_b_legacy/'results/pilot'/ds/'sample_manifest.csv',{'dialogue_id'}))
        if len(pilot_rows)!=120:raise ValueError('Expected historical fixed120 metadata rows')
        if any(r['dialogue_id'] not in feature_to_formal for r in pilot_rows):raise ValueError('Unverified historical pilot ID')
        pilot_dialogues[ds]={feature_to_formal[r['dialogue_id']] for r in pilot_rows}
    # Safe Join to newer Plan B's Train-only role metadata, no old scoring labels/text.
    plan_b=BASE/'plan_b_raw_input_pilot/20261122T041838Z_meld_real_scoring/local_only/SPLIT_DIALOGUES.csv'
    plan_b_rows=list(project_csv(plan_b,{'dialogue_id','role'}))
    plan_b_roles={r['dialogue_id']:r['role'] for r in plan_b_rows}
    overlaps=[]
    for ds in ('IEMOCAP','MELD'):
        for fold,entry in approved[ds]['folds'].items():
            counts=Counter();old_exposure=Counter();unmapped=0
            eval_ids=set(entry['roles']['evaluation']['dialogue_ids'])
            for did in eval_ids:
                b=bridges[ds].get(did)
                if b is None:
                    if by_dataset[ds]['lineage_ids'][did]['original_split']!='test':unmapped+=1
                    continue
                if b['split']=='validation':old_exposure['prior_full_validation']+=1
                else:
                    old_exposure['prior_train_'+b['inner_subset']]+=1
                    if ds=='MELD':
                        fid=b['feature_dialogue_id']
                        if fid in plan_b_roles:counts[plan_b_roles[fid]]+=1
                        else:unmapped+=1
            overlaps.append({'dataset':ds,'fold':fold,'evaluation_dialogues':len(eval_ids),
                'older_plan_b_baseline_exposure_counts':dict(old_exposure),
                'older_plan_b_validation_pilot_target_dialogues':len(eval_ids&pilot_dialogues[ds]),
                'newer_meld_plan_b_role_counts':dict(counts),'non_test_mapping_unverified':unmapped})
    (output/'plan_b_overlap.json').write_text(json.dumps(overlaps,indent=2)+'\n')
    print(json.dumps({'datasets':summary,'rotation':rotation,'plan_b_overlap':overlaps},indent=2))


if __name__=='__main__':main()
