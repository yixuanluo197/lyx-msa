#!/usr/bin/env python3
"""Read-only independent checks of saved predictions, summaries and report cells."""
import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

CLASSES = {
    'IEMOCAP': ['happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated'],
    'MELD': ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger'],
}
THRESHOLDS = [(0.8, '0p8'), (0.9, '0p9'), (0.95, '0p95')]


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            result.update(block)
    return result.hexdigest()


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def boolean(value):
    assert value in ('True', 'False'), value
    return value == 'True'


def equal(value, expected):
    if expected is None:
        assert value == 'N/A', (value, expected)
    else:
        assert math.isclose(float(value), expected, abs_tol=1e-10), (value, expected)


def ratio(a, b):
    return a/b if b else None


def key(row):
    return row['dataset'], row['dialogue_id'], int(row['utterance_index'])


def check_probabilities(row, prefix=''):
    names = CLASSES[row['dataset']]
    probs = [float(row[f'{prefix}probability_{name}']) for name in names]
    assert all(math.isfinite(p) and 0 <= p <= 1 for p in probs)
    assert math.isclose(sum(probs), 1, abs_tol=2e-6)
    predicted = max(range(len(probs)), key=probs.__getitem__)
    assert int(row[f'{prefix}predicted_label']) == predicted
    equal(row[f'{prefix}predicted_confidence'], probs[predicted])
    equal(row[f'{prefix}true_label_probability'], probs[int(row['true_label'])])
    assert boolean(row[f'{prefix}correct']) == (predicted == int(row['true_label']))


def verify_full_tables(output, full):
    """Recalculate experiments 2, 3, 5 without importing their implementation."""
    outcomes=rows(output/'exp2_paired_outcomes.csv')
    new_details=rows(output/'exp2_new_high_confidence_errors.csv')
    remaining=rows(output/'exp3_cm_remaining_high_confidence_errors.csv')
    high_error=lambda row,t:not boolean(row['correct']) and float(row['predicted_confidence'])>=t
    for row in rows(output/'exp2_summary.csv'):
        dataset,t=row['dataset'],float(row['threshold'])
        pairs=list(zip(full[dataset,'baseline'],full[dataset,'cmerc']))
        selected=[(b,c) for b,c in pairs if high_error(b,t)]
        new=[(b,c) for b,c in pairs if not high_error(b,t) and high_error(c,t)]
        counts=Counter('corrected' if boolean(c['correct']) else 'still_wrong_confidence_lower' if float(c['predicted_confidence'])<float(b['predicted_confidence']) else 'still_wrong_confidence_not_lower' for b,c in selected)
        equal(row['valid_test_utterances'],len(pairs))
        equal(row['baseline_high_confidence_errors'],len(selected))
        equal(row['new_cmerc_high_confidence_errors'],len(new))
        equal(row['new_error_eligible_count'],len(pairs)-len(selected))
        equal(row['new_error_ratio'],ratio(len(new),len(pairs)-len(selected)))
        for outcome in ('corrected','still_wrong_confidence_lower','still_wrong_confidence_not_lower'):
            equal(row[outcome+'_count'],counts[outcome])
            equal(row[outcome+'_ratio'],ratio(counts[outcome],len(selected)))
        actual=[r for r in outcomes if r['dataset']==dataset and float(r['threshold'])==t]
        assert len(actual)==len(selected)
        assert {key(r) for r in actual}=={key(b) for b,c in selected}
        assert Counter(r['outcome'] for r in actual)==counts
        actual=[r for r in new_details if r['dataset']==dataset and float(r['threshold'])==t]
        assert len(actual)==len(new) and {key(r) for r in actual}=={key(b) for b,c in new}
        expected=[c for b,c in pairs if high_error(c,t)]
        actual=[r for r in remaining if r['dataset']==dataset and float(r['threshold'])==t]
        assert len(actual)==len(expected) and {key(r) for r in actual}=={key(c) for c in expected}
    for row in rows(output/'exp3_class_summary.csv'):
        dataset,t,label=row['dataset'],float(row['threshold']),row['true_label']
        records=[r for r in full[dataset,'cmerc'] if r['true_label']==label]
        errors=sum(not boolean(r['correct']) for r in records)
        high=sum(float(r['predicted_confidence'])>=t for r in records)
        high_wrong=sum(high_error(r,t) for r in records)
        for name,value in [('class_support',len(records)),('total_errors',errors),('high_confidence_error_count',high_wrong),('high_confidence_prediction_count',high)]:
            equal(row[name],value)
        equal(row['class_error_rate'],ratio(errors,len(records)))
        equal(row['high_confidence_error_rate'],ratio(high_wrong,high))
        equal(row['high_confidence_error_fraction_of_class'],ratio(high_wrong,len(records)))
    bins=[(0,.5),(.5,.6),(.6,.7),(.7,.8),(.8,.9),(.9,.95),(.95,1)]
    distribution=rows(output/'exp5_error_confidence_distribution.csv')
    overall=rows(output/'exp5_overall_summary.csv')
    for (dataset,model),records in full.items():
        errors=[float(r['predicted_confidence']) for r in records if not boolean(r['correct'])]
        table=next(r for r in overall if (r['dataset'],r['model'])==(dataset,model))
        equal(table['valid_test_utterances'],len(records))
        equal(table['total_errors'],len(errors))
        equal(table['accuracy'],1-len(errors)/len(records))
        support=Counter(r['true_label'] for r in records)
        predicted=Counter(r['predicted_label'] for r in records)
        tp=Counter(r['true_label'] for r in records if boolean(r['correct']))
        wf1=sum(n*2*tp[label]/(n+predicted[label]) for label,n in support.items())/len(records)
        equal(table['weighted_f1'],wf1)
        equal(table['mean_error_confidence'],statistics.mean(errors) if errors else None)
        equal(table['median_error_confidence'],statistics.median(errors) if errors else None)
        for t,suffix in THRESHOLDS:
            high=sum(float(r['predicted_confidence'])>=t for r in records)
            wrong=sum(p>=t for p in errors)
            equal(table['errors_ge_'+suffix],wrong)
            equal(table['high_confidence_predictions_ge_'+suffix],high)
            equal(table['high_confidence_error_rate_ge_'+suffix],ratio(wrong,high))
        bin_rows=[r for r in distribution if r['dataset']==dataset]
        assert len(bin_rows)==7
        total=0
        for row,(lo,hi) in zip(bin_rows,bins):
            count=sum(lo<=p and (p<hi if hi<1 else p<=hi) for p in errors)
            equal(row[model+'_error_count'],count)
            equal(row[model+'_ratio_of_errors'],ratio(count,len(errors)))
            total+=count
        assert total==len(errors)


def verify(output, include_reports=True):
    manifest = json.loads((output/'run_manifest.json').read_text())
    exp1 = Path(manifest['exp1_root'])
    full = defaultdict(list)
    for row in rows(output/'full_predictions_reused.csv'):
        check_probabilities(row)
        full[row['dataset'], row['model']].append(row)
    for (dataset, model), records in full.items():
        original = rows(exp1/dataset/model/'all_predictions.csv')
        assert len(records) == len(original)
        assert len({key(r) for r in records}) == len(records)
        for old, new in zip(original, records):
            assert key(old) == key(new)
            assert old['text'] == new['utterance_text']
            assert old['true_label'] == new['true_label']
            assert old['predicted_label'] == new['predicted_label']
            for name in CLASSES[dataset]:
                assert old[f'probability_{name}'] == new[f'probability_{name}']
        assert len(records) == {'IEMOCAP':1623, 'MELD':2610}[dataset]
    for dataset in CLASSES:
        left, right = full[dataset,'baseline'], full[dataset,'cmerc']
        assert len(left) == len(right)
        for a,b in zip(left,right):
            assert key(a) == key(b)
            assert a['utterance_text'] == b['utterance_text']
            assert a['true_label'] == b['true_label']
    verify_full_tables(output,full)
    full_maps = {group:{key(r):r for r in records} for group,records in full.items()}
    masked = defaultdict(list)
    for row in rows(output/'exp4_modality_ablation_predictions.csv'):
        check_probabilities(row,'full_')
        check_probabilities(row,'masked_')
        group = row['dataset'],row['model']
        source = full_maps[group][key(row)]
        assert row['utterance_text'] == source['utterance_text']
        assert row['true_label'] == source['true_label']
        assert row['full_predicted_label'] == source['predicted_label']
        assert row['full_predicted_confidence'] == source['predicted_confidence']
        masked[group+(row['input_condition'],)].append(row)
    assert len(masked) == 12
    for (dataset,model,condition),records in masked.items():
        assert [key(r) for r in records] == [key(r) for r in full[dataset,model]]
        assert len({key(r) for r in records}) == len(records)
        assert condition in ('w/o_text','w/o_audio','w/o_visual')
        assert all(r['removed_modality'] == condition[4:] for r in records)
    for row in rows(output/'exp4_modality_ablation_summary.csv'):
        dataset,model,condition = row['dataset'],row['model'],row['input_condition']
        records = full[dataset,model] if condition=='full' else masked[dataset,model,condition]
        prefix = '' if condition=='full' else 'masked_'
        errors = [r for r in records if not boolean(r[prefix+'correct'])]
        equal(row['valid_utterances'],len(records))
        equal(row['total_errors'],len(errors))
        equal(row['mean_error_confidence'],statistics.mean(float(r[prefix+'predicted_confidence']) for r in errors) if errors else None)
        for threshold,suffix in THRESHOLDS:
            high = [r for r in records if float(r[prefix+'predicted_confidence'])>=threshold]
            wrong = [r for r in high if not boolean(r[prefix+'correct'])]
            equal(row['errors_ge_'+suffix],len(wrong))
            equal(row['high_confidence_predictions_ge_'+suffix],len(high))
            equal(row['high_confidence_error_rate_ge_'+suffix],ratio(len(wrong),len(high)))
    new_details = rows(output/'exp4_new_high_confidence_errors.csv')
    existing = rows(output/'exp4_existing_errors.csv')
    for row in rows(output/'exp4_full_correct_to_masked_error.csv'):
        group = row['dataset'],row['model'],row['input_condition']
        records = masked[group]
        threshold = float(row['threshold'])
        full_correct = [r for r in records if boolean(r['full_correct'])]
        new = [r for r in full_correct if not boolean(r['masked_correct']) and float(r['masked_predicted_confidence'])>=threshold]
        full_wrong = [r for r in records if not boolean(r['full_correct'])]
        still_wrong = [r for r in full_wrong if not boolean(r['masked_correct'])]
        high_wrong = [r for r in still_wrong if float(r['masked_predicted_confidence'])>=threshold]
        for field,value in [('full_correct_samples',len(full_correct)),('full_wrong_samples',len(full_wrong)),('new_high_confidence_error_count',len(new)),('full_wrong_still_wrong_count',len(still_wrong)),('full_wrong_still_wrong_high_confidence_count',len(high_wrong))]:
            equal(row[field],value)
        equal(row['ratio'],ratio(len(new),len(full_correct)))
        equal(row['full_wrong_still_wrong_high_confidence_ratio'],ratio(len(high_wrong),len(full_wrong)))
        selected = [r for r in new_details if (r['dataset'],r['model'],r['input_condition'])==group and float(r['threshold'])==threshold]
        assert {key(r) for r in selected} == {key(r) for r in new}
        selected_existing = [r for r in existing if (r['dataset'],r['model'],r['input_condition'])==group]
        assert {key(r) for r in selected_existing} == {key(r) for r in still_wrong}
    # Explicitly check masking actually happened and no full pass was submitted.
    forward_calls = 0
    for info in manifest['models'].values():
        audit = info['audit']
        assert audit['full_forward_calls']==0 and audit['batch_size']==16
        forward_calls += audit['masked_forward_calls']
        for batch in audit['batches']:
            for condition,item in batch['conditions'].items():
                assert item['mean_axis']==0 and item['padding_included_in_mean']
                assert item['non_target_modalities_unchanged'] and item['target_changed_elements']>0
    assert forward_calls == manifest['new_forward_calls'] == 120
    for dataset in CLASSES:
        a=manifest['models'][dataset+'/baseline']['audit']['batches']
        b=manifest['models'][dataset+'/cmerc']['audit']['batches']
        for left,right in zip(a,b):
            assert left['dialogue_ids']==right['dialogue_ids'] and left['lengths']==right['lengths']
            for condition in left['conditions']:
                assert left['conditions'][condition]['input_sha256']==right['conditions'][condition]['input_sha256']
    for path, expected in manifest['input_files_before_sha256'].items():
        assert digest(path)==expected, path
    assert (output/'SUMMARY.md').read_bytes().startswith((exp1/'SUMMARY.md').read_bytes())
    for filename,count in manifest['table_rows'].items():
        assert len(rows(output/(filename+'.csv')))==count
    result = dict(status='passed', valid_utterances_per_model={'IEMOCAP':1623,'MELD':2610},
                  full_prediction_rows=sum(map(len,full.values())),masked_prediction_rows=sum(map(len,masked.values())),
                  masked_forward_calls=forward_calls,original_hashes_verified=len(manifest['input_files_before_sha256']),
                  full_probabilities_reused_exactly=True, exp4_independent_recalculation=True,
                  exp235_independent_recalculation=True,
                  original_summary_prefix_unchanged=True)
    if include_reports:
        result.update(verify_reports(output))
    return result


def verify_reports(output):
    from docx import Document
    report = json.loads((output/'report_manifest.json').read_text())
    string_digest = lambda value:hashlib.sha256(json.dumps(value,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    for item in report['outputs'].values():
        assert digest(item['path'])==item['sha256']
        with ZipFile(item['path']) as archive:
            assert archive.testzip() is None
    spreadsheet = next(Path(item['path']) for name,item in report['outputs'].items() if name.endswith('.xlsx'))
    ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with ZipFile(spreadsheet) as archive:
        strings = [''.join(si.itertext()) for si in ET.fromstring(archive.read('xl/sharedStrings.xml'))]
        for sheet in report['sheets']:
            xml=ET.fromstring(archive.read(f"xl/worksheets/sheet{sheet['worksheet_index']}.xml"))
            actual=[]
            for row in xml.findall('s:sheetData/s:row',ns):
                values=[]
                for cell in row.findall('s:c',ns):
                    assert cell.find('s:f',ns) is None, 'Unexpected Excel formula'
                    raw=cell.findtext('s:v',namespaces=ns)
                    values.append(strings[int(raw)] if cell.attrib.get('t')=='s' else raw)
                actual.append(values)
            source=rows(sheet['source_csv'])
            headers=sheet['columns']
            expected=[[source[i][h] if source[i][h] not in ('','None','null','NaN','nan','N/A') else 'N/A' for h in headers] for i in sheet['source_row_indices']]
            # Excel XML uses escape sequences for literal C1 control characters.
            def unescape(value):
                return re.sub(r'_x([0-9A-Fa-f]{4})_',lambda m:chr(int(m[1],16)),value)
            actual=[[unescape(v) for v in row] for row in actual]
            assert actual[0]==headers
            assert actual[1:]==expected, sheet['name']
            assert len(expected)==sheet['row_count']
            assert string_digest(expected)==sheet['rendered_rows_sha256']
    word = next(Path(item['path']) for name,item in report['outputs'].items() if name.endswith('.docx'))
    document=Document(word)
    for table in report['word_tables']:
        actual=[[cell.text for cell in row.cells] for row in document.tables[table['document_table_index']].rows]
        source=rows(table['source_csv'])
        expected=[[source[i][h] if source[i][h] else 'N/A' for h in table['columns']] for i in table['source_row_indices']]
        assert actual[0]==table['columns'] and actual[1:]==expected
        assert string_digest(actual[1:])==table['rendered_rows_sha256']
    return dict(excel_sheets_verified=len(report['sheets']),word_tables_verified=len(report['word_tables']),
                excel_cells_equal_csv=True,word_summary_cells_equal_csv=True,
                original_word_template_found=report['mother_document']['found'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--without-reports',action='store_true')
    options=parser.parse_args()
    print(json.dumps(verify(options.output_dir.resolve(),not options.without_reports),ensure_ascii=False,indent=2))
