#!/usr/bin/env python3
"""Prepare Plan B raw inputs and official visual features, with no legacy text/audio features."""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import pickle
import platform
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/data2/yb/paper/lyx-msa')
HISTORY = ROOT / 'experiments/xp_caldist_behavioral_calibration/xp04_random_feature_behavior_calibration/raw_data_pilot'
XP = ROOT / 'CMERC/experiments/xp_mainline_validation/20261016T050823Z'
MELD_RAW = Path('/data2/yb/OpenDataLab___MELD/raw/MELD/MELD.Raw')
SPECS = {
    'IEMOCAP': dict(seed=1475, filename='IEMOCAP_features.pkl', train=5283, validation=527,
                    fit=4747, dev=536, speakers=2,
                    class_names=['happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated'],
                    sha256='ceb5cc9a45d1792998438e27f86a12642320aa6315546e4425316140ea503fb3'),
    'MELD': dict(seed=67137, filename='MELD_features_raw1.pkl', train=9989, validation=1109,
                 fit=8989, dev=1000, speakers=9,
                 class_names=['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger'],
                 sha256='d043b3ec1813c8917e2067af01e933ff0892d33d19d93595098aaf88c4b200b0'),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def savej(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def array_hash(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(str(array.dtype).encode() + str(array.shape).encode() + array.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--recovery-json', type=Path,
                        help='Optional approved same-split recovered raw-media mapping, keyed dataset/split/raw dialogue/raw utterance')
    args = parser.parse_args()
    out = args.out.resolve()
    recovery = json.loads(args.recovery_json.read_text()) if args.recovery_json else {}
    sources, artifact_hashes, summaries, missing_all = {}, {}, {}, []
    if args.recovery_json:
        sources[str(args.recovery_json.resolve())] = sha256(args.recovery_json)
    all_media_rows, text_conflicts = [], []
    limited_recovery_path = out / 'provenance/meld_limited_recovery.json'
    if limited_recovery_path.is_file():
        sources[str(limited_recovery_path)] = sha256(limited_recovery_path)
    for dataset, spec in SPECS.items():
        destination = out / 'local_only/data' / dataset
        dataset_path = destination / 'dataset.pt'
        destination.mkdir(parents=True, exist_ok=True)
        cfg_path = ROOT / f'experiments/xp_caldist_behavioral_calibration/20261107T172302Z/runs/{dataset}/seed{spec["seed"]}/xp04_effective_config.json'
        cfg = json.loads(cfg_path.read_text())
        assert cfg['Dataset'] == dataset and cfg['seed'] == spec['seed']
        # Test split identifiers are deliberately never selected.
        ids = {split: cfg['provenance']['split'][split + '_ids'] for split in ['train', 'validation']}
        assert len(set(ids['train'])) == len(ids['train'])
        assert len(set(ids['validation'])) == len(ids['validation'])
        assert set(ids['train']).isdisjoint(ids['validation'])
        sources[str(cfg_path)] = sha256(cfg_path)
        base_path = XP / 'code/data' / spec['filename']
        sources[str(base_path.resolve())] = sha256(base_path)
        assert sources[str(base_path.resolve())] == spec['sha256']
        # Shared official container materializes every field during unpickling.
        # ONLY slots 0 IDs, 1 speakers, 2 emotion labels, 5 visual and 6 raw text
        # are selected below, exclusively for effective Train and Validation.
        # Legacy slot 3 text features and slot 4 audio features are never indexed.
        with base_path.open('rb') as stream:
            base = pickle.load(stream, encoding='latin1')
        inner_path = HISTORY / '20261118T125439Z/local_only' / dataset / 'train_manifest.csv'
        inner_rows = csv_rows(inner_path)
        sources[str(inner_path)] = sha256(inner_path)
        assert len(inner_rows) == spec['train']
        inner = {(row['dialogue_id'], int(row['utterance_index'])): row['inner_subset'] for row in inner_rows}
        fit_ids = [dialogue for dialogue in ids['train'] if inner[(str(dialogue), 0)] == 'fit']
        dev_ids = [dialogue for dialogue in ids['train'] if inner[(str(dialogue), 0)] == 'inner_dev']
        assert set(fit_ids).isdisjoint(dev_ids) and set(fit_ids) | set(dev_ids) == set(ids['train'])
        result = dict(dataset=dataset, seed=spec['seed'], class_names=spec['class_names'],
                      splits={}, fit_dialogue_ids=list(map(str, fit_ids)),
                      inner_dev_dialogue_ids=list(map(str, dev_ids)))
        summary = dict(seed=spec['seed'], class_names=spec['class_names'], source_pickle_slots=[0, 1, 2, 5, 6],
                       shared_pickle_deserialized=True, legacy_text_feature_slot_indexed=False,
                       legacy_audio_feature_slot_indexed=False, test_dictionary_values_indexed=False,
                       previous_teacher_pt_loaded=False, train_inner_indices_from_csv_only=True,
                       text_source=('official base pkl slot 6' if dataset == 'IEMOCAP' else
                                    'original Train/dev CSV, exact raw Dialogue ID+Utterance ID alignment'),
                       text_preprocessing='none: preserve chosen original strings, including original Unicode/control characters',
                       official_visual_dim=342, speaker_dim=spec['speakers'], splits={})
        manifest, dialogue_summaries, speaker_names = [], [], defaultdict(set)
        for split in ['train', 'validation']:
            old_manifest_path = HISTORY / '20261117T024213Z/local_only' / dataset / f'manifest_{split}.csv'
            old_rows = csv_rows(old_manifest_path)
            sources[str(old_manifest_path)] = sha256(old_manifest_path)
            old_lookup = {(row['dialogue_id'], int(row['utterance_index'])): row for row in old_rows}
            assert len(old_lookup) == spec[split]
            metadata = {}
            if dataset == 'MELD':
                csv_path = MELD_RAW / ('train_sent_emo.csv' if split == 'train' else 'dev_sent_emo.csv')
                sources[str(csv_path)] = sha256(csv_path)
                metadata = {(int(row['Dialogue_ID']), int(row['Utterance_ID'])): row for row in csv_rows(csv_path)}
                assert len(metadata) == spec[split]
            dialogues, split_rows, raw_uid_differences = [], [], 0
            for dialogue in ids[split]:
                uid = [str(value) for value in base[0][dialogue]]
                n = len(uid)
                texts = list(base[6][dialogue])
                labels = np.asarray(base[2][dialogue], dtype=np.int64)
                visual = np.asarray(base[5][dialogue], dtype=np.float32)
                if dataset == 'IEMOCAP':
                    original_speakers = base[1][dialogue]
                    assert all(speaker in ('M', 'F') for speaker in original_speakers)
                    speaker = np.array([[1., 0.] if value == 'M' else [0., 1.] for value in original_speakers], dtype=np.float32)
                else:
                    speaker = np.asarray(base[1][dialogue], dtype=np.float32)
                assert n == len(set(uid)) == len(texts)
                assert labels.shape == (n,) and ((labels >= 0) & (labels < len(spec['class_names']))).all()
                assert visual.shape == (n, 342) and np.isfinite(visual).all()
                assert speaker.shape == (n, spec['speakers']) and np.isfinite(speaker).all()
                assert np.all((speaker == 0) | (speaker == 1)) and np.all(speaker.sum(axis=1) == 1)
                audio = []
                dialogue_media = []
                for index, text in enumerate(texts):
                    assert isinstance(text, str)
                    previous = old_lookup[(str(dialogue), index)]
                    assert previous['raw_utterance_id'] == uid[index]
                    assert int(previous['speaker_id']) == int(speaker[index].argmax())
                    if dataset == 'MELD':
                        raw_dialogue = int(dialogue) - (1039 if split == 'validation' else 0)
                        assert str(raw_dialogue) == previous['raw_dialogue_id']
                        raw_meta = metadata[(raw_dialogue, int(uid[index]))]
                        if text != raw_meta['Utterance']:
                            text_conflicts.append(dict(dataset=dataset, split=split, dialogue_id=str(dialogue),
                                                       utterance_id=uid[index], official_pkl_text=text,
                                                       raw_csv_text=raw_meta['Utterance'],
                                                       decision='Use exact-ID raw CSV text for new baseline; preserve legacy discrepancy'))
                        text = raw_meta['Utterance']
                        texts[index] = text
                        assert spec['class_names'][labels[index]] == raw_meta['Emotion'], 'Raw CSV emotion alignment changed'
                        speaker_name = raw_meta['Speaker']
                        speaker_names[speaker_name].add(int(speaker[index].argmax()))
                    else:
                        raw_dialogue = str(dialogue)
                        speaker_name = original_speakers[index]
                        assert uid[index].startswith(str(dialogue) + '_')
                        assert uid[index].rsplit('_', 1)[-1][0] == speaker_name
                    media = Path(previous['media_path'])
                    known_bad = dataset == 'MELD' and ((split, raw_dialogue, uid[index]) in
                                [('train', 125, '3'), ('validation', 110, '7')])
                    key = f'{dataset}/{split}/{raw_dialogue}/{uid[index]}'
                    if key in recovery:
                        recovered = recovery[key]
                        assert recovered['same_split'] and recovered['adopted']
                        media = Path(recovered['path'])
                        assert media.is_file() and sha256(media) == recovered['sha256']
                        known_bad = False
                    usable = media.is_file() and not known_bad
                    audio_path = str(media) if usable else None
                    audio.append(audio_path)
                    row = dict(dataset=dataset, split=split, dialogue_id=str(dialogue),
                               utterance_index=index, utterance_id=uid[index],
                               raw_dialogue_id=str(raw_dialogue), speaker_id=int(speaker[index].argmax()),
                               speaker_name=speaker_name, label=int(labels[index]),
                               text_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest(),
                               visual_sha256=array_hash(visual[index]), audio_path=audio_path,
                               original_media_path=previous['media_path'],
                               historical_media_sha256=previous['media_sha256'],
                               audio_missing_reason=('known_corrupt_original_media' if known_bad and split == 'train' else
                                                     'known_missing_original_media' if known_bad else
                                                     'path_missing' if not usable else ''),
                               inner_subset=inner[(str(dialogue), index)] if split == 'train' else 'official_validation')
                    if split == 'train':
                        assert row['inner_subset'] == inner[(str(dialogue), 0)]
                        if dataset == 'IEMOCAP':
                            assert row['text_sha256'] == inner_rows[len(split_rows)]['original_text_sha256']
                    if dataset == 'MELD':
                        raw_uid_differences += int(uid[index] != str(index))
                    manifest.append(row)
                    split_rows.append(row)
                    dialogue_media.append(row)
                    if not usable:
                        missing_all.append(row)
                dialogues.append(dict(dialogue_id=str(dialogue), utterance_ids=uid,
                                      speaker=torch.from_numpy(speaker), labels=torch.from_numpy(labels),
                                      texts=texts, audio_paths=audio, visual=torch.from_numpy(visual)))
                dialogue_summaries.append(dict(dataset=dataset, split=split, dialogue_id=str(dialogue),
                                                utterances=n, audio_missing=sum(path is None for path in audio),
                                                visual_sha256=array_hash(visual), labels_sha256=array_hash(labels),
                                                speaker_sha256=array_hash(speaker),
                                                text_order_sha256=hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()))
            assert len(split_rows) == spec[split]
            result['splits'][split] = dialogues
            summary['splits'][split] = dict(dialogues=len(dialogues), utterances=len(split_rows),
                                            label_counts=dict(Counter(row['label'] for row in split_rows)),
                                            audio_present=sum(row['audio_path'] is not None for row in split_rows),
                                            audio_missing=sum(row['audio_path'] is None for row in split_rows),
                                            raw_utterance_id_differs_from_array_index=raw_uid_differences)
        assert sum(len(dialogue['texts']) for dialogue in result['splits']['train'] if dialogue['dialogue_id'] in result['fit_dialogue_ids']) == spec['fit']
        assert sum(len(dialogue['texts']) for dialogue in result['splits']['train'] if dialogue['dialogue_id'] in result['inner_dev_dialogue_ids']) == spec['dev']
        summary.update(fit_utterances=spec['fit'], inner_dev_utterances=spec['dev'],
                       fit_dialogues=len(fit_ids), inner_dev_dialogues=len(dev_ids),
                       raw_speaker_to_official_slot={key: sorted(value) for key, value in speaker_names.items()})
        if dataset == 'MELD':
            by_dialogue = defaultdict(lambda: defaultdict(set))
            for row in manifest:
                by_dialogue[(row['split'], row['dialogue_id'])][row['speaker_name']].add(row['speaker_id'])
            summary['speaker_slots_are_dialogue_local'] = True
            summary['within_dialogue_csv_speaker_slot_conflicts'] = [
                dict(split=split, dialogue_id=dialogue, speaker_name=name, official_slots=sorted(slots))
                for (split, dialogue), names in by_dialogue.items() for name, slots in names.items() if len(slots) > 1]
            summary['speaker_conflict_decision'] = 'Keep original official Speaker ID; do not change slot0 to CSV-implied slot1 at Train556/utt6'
        if dataset_path.exists():
            existing = torch.load(dataset_path, map_location='cpu', weights_only=False)
            for key in ['dataset', 'seed', 'class_names', 'fit_dialogue_ids', 'inner_dev_dialogue_ids']:
                assert existing[key] == result[key], 'Prepared metadata changed on resume'
            for split in ['train', 'validation']:
                assert len(existing['splits'][split]) == len(result['splits'][split])
                for old, new in zip(existing['splits'][split], result['splits'][split]):
                    for key in ['dialogue_id', 'utterance_ids', 'audio_paths']:
                        assert old[key] == new[key], 'Prepared raw input changed on resume'
                    for key in ['speaker', 'labels', 'visual']:
                        assert torch.equal(old[key], new[key]), 'Prepared tensors changed on resume'
            text_changed = any(old['texts'] != new['texts'] for split in ['train', 'validation']
                               for old, new in zip(existing['splits'][split], result['splits'][split]))
            if text_changed:
                assert dataset == 'MELD', 'Only authorized MELD raw CSV canonicalization may replace text'
                changed = [(old['dialogue_id'], old['utterance_ids'][index])
                           for split in ['train', 'validation']
                           for old, new in zip(existing['splits'][split], result['splits'][split])
                           for index in range(len(old['texts'])) if old['texts'][index] != new['texts'][index]]
                assert changed == [('556', '6')], 'Unexpected raw text source change'
                temporary = dataset_path.with_suffix('.tmp.pt')
                torch.save(result, temporary)
                os.replace(temporary, dataset_path)
                print('Applied authorized canonical raw CSV text at MELD Train556/utt6', flush=True)
            else:
                print('Verified identical existing dataset without overwriting: ' + str(dataset_path), flush=True)
        else:
            torch.save(result, dataset_path)
        artifact_hashes[str(dataset_path)] = sha256(dataset_path)
        for filename, rows in [('manifest.csv', manifest), ('dialogue_summary.csv', dialogue_summaries)]:
            path = destination / filename
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            artifact_hashes[str(path)] = sha256(path)
        summaries[dataset] = summary
        all_media_rows.extend(manifest)
        print(json.dumps(dict(dataset=dataset, path=str(dataset_path), **summary['splits'])), flush=True)
    media_paths = sorted({row['audio_path'] for row in all_media_rows if row['audio_path'] is not None})
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        media_hashes = dict(zip(media_paths, pool.map(sha256, media_paths)))
    for row in all_media_rows:
        if row['audio_path'] is not None and row['audio_path'] == row['original_media_path']:
            assert media_hashes[row['audio_path']] == row['historical_media_sha256'], 'Historical raw media changed'
    savej(out / 'local_only/data/media_sha256.json', media_hashes)
    savej(out / 'local_only/data/audio_missing.json', missing_all)
    savej(out / 'local_only/data/text_source_conflicts.json', text_conflicts)
    artifact_hashes[str(out / 'local_only/data/media_sha256.json')] = sha256(out / 'local_only/data/media_sha256.json')
    artifact_hashes[str(out / 'local_only/data/audio_missing.json')] = sha256(out / 'local_only/data/audio_missing.json')
    artifact_hashes[str(out / 'local_only/data/text_source_conflicts.json')] = sha256(out / 'local_only/data/text_source_conflicts.json')
    assert all(sha256(path) == digest for path, digest in sources.items()), 'Input changed during preparation'
    provenance = dict(status='COMPLETE_WITH_RETAINED_MISSING_AUDIO' if missing_all else 'COMPLETE',
                      datasets=summaries, source_sha256=sources, artifacts_sha256=artifact_hashes,
                      media_files_hashed=len(media_hashes), media_hashes='local_only/data/media_sha256.json',
                      audio_missing='local_only/data/audio_missing.json',
                      text_source_conflicts='local_only/data/text_source_conflicts.json',
                      text_source_conflict_count=len(text_conflicts),
                      shared_official_pickle_materializes_test=True, test_ids_or_values_indexed=False,
                      old_text_or_audio_features_used=False, emotion_labels_used_only_train_validation=True,
                      previous_teacher_pt_loaded=False, original_sources_modified=False,
                      environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__))
    savej(out / 'provenance/data_preparation.json', provenance)
    lines = ['# Plan B 数据准备报告', '',
             '仅准备原始文本、原始音频路径、官方视觉特征、原标签与 Speaker；未启动训练。', '',
             '| 数据集 | Train / 对话 | Validation / 对话 | Fit | Inner dev | 视觉 | Speaker | 缺失音频 Train / Val |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for dataset, summary in summaries.items():
        train, val = summary['splits']['train'], summary['splits']['validation']
        lines.append(f'| {dataset} | {train["utterances"]} / {train["dialogues"]} | {val["utterances"]} / {val["dialogues"]} | {summary["fit_utterances"]} / {summary["fit_dialogues"]} 对话 | {summary["inner_dev_utterances"]} / {summary["inner_dev_dialogues"]} 对话 | 342 | {summary["speaker_dim"]} | {train["audio_missing"]} / {val["audio_missing"]} |')
    lines += ['', '有效 Train/Validation 划分沿用 Seed1475 / Seed67137 的 XP04 frozen config；Train 内 holdout 完全沿用 20261118T125439Z 已固定的 Dialogue 分组 CSV，只读取索引与文本哈希，不加载旧教师特征 `.pt`。Fit、Inner dev 与官方 Validation 之间无 Dialogue 重叠。', '',
              '基础官方 pkl 仅索引 0（真实话语 ID）、1（Speaker）、2（情绪标签）、5（视觉）及 6（原始文本）槽位。共享 pkl 反序列化会物化包括 Test 的容器，但代码没有选择 Test 标识或值，未索引旧文本槽位 3、旧音频槽位 4。', '',
              f'IEMOCAP 文本逐字符保留官方 pkl 原始字符串；MELD 全部使用原始 Train/dev CSV 按精确 Dialogue ID+真实 Utterance ID 对齐的文本，不做 ASCII 删除或情绪表达修改。发现 {len(text_conflicts)} 条历史文本来源冲突，即 Train dialogue556/真实utt6；按本轮新 baseline 指令采用原始 CSV 文本，旧 pkl 文本仅在本地 text_source_conflicts.json 完整留档。Validation 文本全部与旧 pkl 一致，Train/Validation 标签全部与 CSV 一致。保留原始标点/控制字符。IEMOCAP 真实话语 ID 中 M/F 与 Speaker 一致。Speaker 使用原官方 one-hot（IEMOCAP M=[1,0]、F=[0,1]；MELD 原 9 槽）。', '',
              'Train dialogue556/真实utt6 还存在唯一 Speaker 元数据冲突：原始 CSV 为 Ross，在该对话其他行对应 slot1；该行官方 pkl 为 slot0。按冻结决定保留用户要求的原 Speaker ID（slot0），保留原始 CSV 文本，不删除话语、不修改标签或划分，并将这一历史异常单独披露。不能声称全部 Speaker 与 CSV 一致；该异常仅在 Train，不在 Validation pilot。MELD Speaker 槽位是 Dialogue 内编码，同一角色在不同 Dialogue 中可使用不同槽。', '',
              'MELD Validation 的 XP04 Dialogue ID 使用原始 dev ID +1039。真实 Utterance ID 与数组位置分别保留，不用数组位置猜测媒体名称。IEMOCAP 音频指向原 WAV，MELD 指向同划分原 MP4 的音轨；本阶段不变换、合成或填充音频。', '',
              '已知原 MELD Train dia125_utt3 损坏、dev dia110_utt7 缺失。未成功恢复的条目保留完整文本、视觉、标签及顺序，audio_paths=None，详见本地 audio_missing.json；不删除话语、不用历史音频特征替代、不从 Test 借同名媒体。有限同版本恢复检查由音频准备负责人记录。', '',
              f'完整现存许可媒体逐文件 SHA-256 共 {len(media_hashes)} 个，且与历史原始媒体 SHA 一致；所有输入配置、官方 pkl、历史 ID 清单、输出 dataset.pt 均已哈希留档。见 `provenance/data_preparation.json`；媒体哈希和完整数据仅保留在 `local_only/data/`。', '',
              '接口：`dataset.pt={dataset,seed,class_names,splits:{train,validation},fit_dialogue_ids,inner_dev_dialogue_ids}`；每个 dialogue 包含 dialogue_id、utterance_ids、speaker、labels、texts、audio_paths、visual。所有 Dialogue ID 与 Utterance ID 输出为字符串；标签 LongTensor，Speaker/视觉 FloatTensor。', '',
              'IEMOCAP 上游视觉/标签/文本源保持历史版本；先前 RoBERTa 上游训练划分重叠风险不转化为本轮旧文本特征输入，本轮没有使用任何旧文本或旧音频表示。官方视觉提取器的历史生产元数据仍存在既有来源限制。', '']
    (out / 'reports/DATA_PREPARATION_REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status=provenance['status'], media_files_hashed=len(media_hashes),
                          missing_audio=len(missing_all))), flush=True)


if __name__ == '__main__':
    main()
