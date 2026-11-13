"""Record the exact per-feature PCG64 stream keys used by the launched Pilot."""
import csv
import hashlib
import json
import os
from pathlib import Path

from run_behavior import MODALITIES, NAMES, OUT


def key_seed(key):
    canonical = json.dumps(key, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    return int.from_bytes(hashlib.sha256(canonical.encode('utf-8')).digest()[:16], 'big')


for dataset in ('IEMOCAP', 'MELD'):
    source = OUT / dataset / 'perturbed_predictions.csv'
    temp = source.with_suffix('.csv.tmp')
    with source.open(newline='') as inp, temp.open('w', newline='') as out:
        reader = csv.DictReader(inp)
        fields = list(reader.fieldnames) + ['rng_seeds']
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        count = 0
        for row in reader:
            missing = row['visual_missing_skipped'].lower() == 'true'
            keys = [[dataset, int(row['seed']), row['dialogue_id'], int(row['utterance_index']),
                     NAMES[index], row['corruption'], float(row['strength']), int(row['repeat_id'])]
                    for index in MODALITIES[row['modality']]
                    if not (index == 4 and missing)]
            row['rng_key'] = json.dumps(keys, separators=(',', ':'))
            row['rng_seeds'] = json.dumps([key_seed(key) for key in keys], separators=(',', ':'))
            writer.writerow(row)
            count += 1
    os.replace(temp, source)
    print(dataset, count, 'random stream records corrected')
