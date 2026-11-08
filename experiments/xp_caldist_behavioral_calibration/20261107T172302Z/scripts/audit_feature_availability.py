"""Record zero-vector metadata on validation only; do not change any experiment."""
import argparse
import csv
import json
import pickle
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args();root=args.root.resolve()
    base=root.parents[2]/'CMERC/experiments/xp_mainline_validation/20261016T050823Z';rows=[];summary=[]
    specs=[('IEMOCAP','IEMOCAP_features.pkl','iemocap_features_roberta.pkl',[2,3,4,5]),('MELD','MELD_features_raw1.pkl','meld_features_roberta.pkl',[3,4,5,6])]
    for dataset,raw,roberta,indices in specs:
        config=json.loads((base/f'configs/effective/{dataset}_cmerc_validation_seed1475.json').read_text());keys=config['provenance']['split']['validation_ids']
        with (base/'code/data'/raw).open('rb') as handle:payload=pickle.load(handle,encoding='latin1')
        with (base/'code/data'/roberta).open('rb') as handle:text=pickle.load(handle,encoding='latin1')
        for modality,collections in [('audio',[payload[4]]),('visual',[payload[5]]),('text_representation',[text[i] for i in indices])]:
            count=0
            for key in keys:
                for index in range(len(payload[2][key])):
                    vectors=[np.asarray(collection[key][index]) for collection in collections]
                    zero=all(np.all(vector==0) for vector in vectors);count+=zero
                    rows.append(dict(dataset=dataset,modality=modality,dialogue_id=str(key),utterance_index=index,
                                     all_zero_features=bool(zero),zero_layers=sum(np.all(vector==0) for vector in vectors),
                                     layer_l2_norms=json.dumps([float(np.linalg.norm(vector)) for vector in vectors])))
            summary.append(dict(dataset=dataset,modality=modality,all_zero_utterances=int(count)))
    with (root/'results/validation_feature_availability.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (root/'results/feature_availability_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))

if __name__=='__main__':main()
