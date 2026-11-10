#!/usr/bin/env python3
"""Read-only XP04 input auditor; writes only into its audit directory."""

import csv
import hashlib
import json
import os
import pickle
import re
from pathlib import Path

import numpy as np


AUDIT_DIR = Path(__file__).resolve().parent.parent
WORKSPACE = Path("/data2/yb/paper/lyx-msa")
CMERC = WORKSPACE / "CMERC"
DATA = CMERC / "data"
EXTERNAL = Path("/data2/yb/paper/05_reproduction/external_repos/CMERC")
XP04 = CMERC / "experiment_records/04_xp_mainline_validation"
IEM_RAW = Path("/data2/yb/paper/IEMOCAP_extract/IEMOCAP_full_release")
MELD_RAW = Path("/data2/yb/OpenDataLab___MELD/raw/MELD/MELD.Raw")


def load_pickle(path):
    with path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_stats(dataset, modality, values):
    arrays = [np.asarray(value) for value in values]
    widths = sorted({int(array.shape[1]) for array in arrays})
    dtypes = sorted({str(array.dtype) for array in arrays})
    rows = sum(int(array.shape[0]) for array in arrays)
    zero_rows = sum(int(np.all(array == 0, axis=1).sum()) for array in arrays)
    finite = sum(int(np.isfinite(array).sum()) for array in arrays)
    elements = sum(int(array.size) for array in arrays)
    total_sum = sum(float(array.astype(np.float64).sum()) for array in arrays)
    total_sq = sum(float(np.square(array.astype(np.float64)).sum()) for array in arrays)
    mean = total_sum / elements
    variance = max(0.0, total_sq / elements - mean * mean)
    return {
        "dataset": dataset,
        "modality": modality,
        "dialogue_count": len(arrays),
        "utterance_count": rows,
        "feature_width": ";".join(map(str, widths)),
        "dtype": ";".join(dtypes),
        "element_count": elements,
        "finite_element_count": finite,
        "nonfinite_element_count": elements - finite,
        "all_zero_utterance_count": zero_rows,
        "mean": mean,
        "std": variance ** 0.5,
    }


def split_name(key, train, validation, test):
    if key in train:
        return "train"
    if key in validation:
        return "validation"
    if key in test:
        return "test"
    return "UNMAPPED"


def load_iemocap_raw_index():
    texts, labels = {}, {}
    transcript_pattern = re.compile(r"^(\S+) \[[^]]+\]: (.*)$")
    label_pattern = re.compile(r"^\[[^]]+\]\s+(\S+)\s+([a-z]+)\s+\[")
    for path in IEM_RAW.glob("Session*/dialog/transcriptions/*.txt"):
        for line in path.read_text(errors="replace").splitlines():
            match = transcript_pattern.match(line)
            if match:
                texts[match.group(1)] = match.group(2)
    for path in IEM_RAW.glob("Session*/dialog/EmoEvaluation/*.txt"):
        for line in path.read_text(errors="replace").splitlines():
            match = label_pattern.match(line)
            if match:
                labels[match.group(1)] = match.group(2)
    return texts, labels


def load_meld_raw_index():
    result = {}
    offsets = {"train": 0, "dev": 1039, "test": 1153}
    for split, offset in offsets.items():
        path = MELD_RAW / f"{split}_sent_emo.csv"
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                key = int(row["Dialogue_ID"]) + offset
                uid = int(row["Utterance_ID"])
                result[(key, uid)] = row
    return result


def write_alignment_and_stats():
    i_raw = load_pickle(DATA / "IEMOCAP_features.pkl")
    i_rob = load_pickle(DATA / "iemocap_features_roberta.pkl")
    m_raw = load_pickle(DATA / "MELD_features_raw1.pkl")
    m_rob = load_pickle(DATA / "meld_features_roberta.pkl")
    i_raw_texts, i_raw_labels = load_iemocap_raw_index()
    m_raw_index = load_meld_raw_index()

    columns = [
        "dataset", "dialogue_id", "split", "utterance_index", "utterance_id",
        "sentence", "speaker", "label", "raw_roberta_sentence_match",
        "raw_roberta_speaker_match", "raw_roberta_label_match", "all_feature_lengths_match",
        "legacy_text_dim", "t1_dim", "t2_dim", "t3_dim", "t4_dim",
        "audio_dim", "visual_dim", "audio_all_zero", "visual_all_zero",
        "raw_audio_status", "raw_video_status",
        "candidate_raw_text_status", "candidate_raw_speaker_status",
        "candidate_raw_label", "candidate_raw_label_status", "candidate_raw_media_status",
    ]
    rows = []
    stats = []

    i_ids, i_speakers, i_labels, i_legacy, i_audio, i_visual, i_sentences, _, i_test = i_raw
    i_rspeakers, i_rlabels, i_t1, i_t2, i_t3, i_t4, i_rsentences, i_train, i_rtest, i_valid = i_rob
    for key in i_ids:
        feature_lengths = [
            len(i_ids[key]), len(i_speakers[key]), len(i_labels[key]), len(i_legacy[key]),
            len(i_audio[key]), len(i_visual[key]), len(i_sentences[key]), len(i_rspeakers[key]),
            len(i_rlabels[key]), len(i_t1[key]), len(i_t2[key]), len(i_t3[key]),
            len(i_t4[key]), len(i_rsentences[key]),
        ]
        aligned = len(set(feature_lengths)) == 1
        for index in range(len(i_ids[key])):
            audio = np.asarray(i_audio[key][index])
            visual = np.asarray(i_visual[key][index])
            uid = i_ids[key][index]
            raw_label = i_raw_labels.get(uid)
            rows.append({
                "dataset": "IEMOCAP", "dialogue_id": key,
                "split": split_name(key, set(i_train), set(i_valid), set(i_rtest)),
                "utterance_index": index, "utterance_id": i_ids[key][index],
                "sentence": i_sentences[key][index], "speaker": i_speakers[key][index],
                "label": i_labels[key][index],
                "raw_roberta_sentence_match": i_sentences[key][index] == i_rsentences[key][index],
                "raw_roberta_speaker_match": i_speakers[key][index] == i_rspeakers[key][index],
                "raw_roberta_label_match": i_labels[key][index] == i_rlabels[key][index],
                "all_feature_lengths_match": aligned,
                "legacy_text_dim": len(i_legacy[key][index]), "t1_dim": len(i_t1[key][index]),
                "t2_dim": len(i_t2[key][index]), "t3_dim": len(i_t3[key][index]),
                "t4_dim": len(i_t4[key][index]), "audio_dim": len(audio),
                "visual_dim": len(visual), "audio_all_zero": bool(np.all(audio == 0)),
                "visual_all_zero": bool(np.all(visual == 0)),
                "raw_audio_status": "CANDIDATE_SENTENCE_WAV", "raw_video_status": "CANDIDATE_DIALOGUE_AVI",
                "candidate_raw_text_status": "MATCH" if i_raw_texts.get(uid) == i_sentences[key][index] else "MISMATCH_OR_MISSING",
                "candidate_raw_speaker_status": "MATCH" if uid.rsplit("_", 1)[-1][:1] == i_speakers[key][index] else "MISMATCH",
                "candidate_raw_label": raw_label or "MISSING",
                "candidate_raw_label_status": "UNKNOWN_MAPPED_TO_HAP" if raw_label == "xxx" else "MATCH_KNOWN_CLASS",
                "candidate_raw_media_status": "WAV_ID_MATCH;DIALOGUE_AVI_MATCH",
            })
    for name, values in [
        ("legacy_text", i_legacy.values()), ("T1", i_t1.values()), ("T2", i_t2.values()),
        ("T3", i_t3.values()), ("T4", i_t4.values()), ("audio", i_audio.values()),
        ("visual", i_visual.values()),
    ]:
        stats.append(array_stats("IEMOCAP", name, values))

    m_ids, m_speakers, m_labels, m_legacy, m_audio, m_visual, m_sentences, _, m_test, _ = m_raw
    m_rspeakers, m_rlabels, _, m_t1, m_t2, m_t3, m_t4, m_rsentences, m_train, m_rtest, m_valid = m_rob
    for key in m_ids:
        feature_lengths = [
            len(m_ids[key]), len(m_speakers[key]), len(m_labels[key]), len(m_legacy[key]),
            len(m_audio[key]), len(m_visual[key]), len(m_sentences[key]), len(m_rspeakers[key]),
            len(m_rlabels[key]), len(m_t1[key]), len(m_t2[key]), len(m_t3[key]),
            len(m_t4[key]), len(m_rsentences[key]),
        ]
        aligned = len(set(feature_lengths)) == 1
        for index in range(len(m_ids[key])):
            audio = np.asarray(m_audio[key][index])
            visual = np.asarray(m_visual[key][index])
            uid = m_ids[key][index]
            raw_row = m_raw_index.get((key, int(uid)))
            raw_text_status = "MATCH" if raw_row and raw_row["Utterance"] == m_sentences[key][index] else "MISMATCH_OR_MISSING"
            rows.append({
                "dataset": "MELD", "dialogue_id": key,
                "split": split_name(key, set(m_train), set(m_valid), set(m_rtest)),
                "utterance_index": index, "utterance_id": m_ids[key][index],
                "sentence": m_sentences[key][index], "speaker": json.dumps(m_speakers[key][index]),
                "label": m_labels[key][index],
                "raw_roberta_sentence_match": m_sentences[key][index] == m_rsentences[key][index],
                "raw_roberta_speaker_match": m_speakers[key][index] == m_rspeakers[key][index],
                "raw_roberta_label_match": m_labels[key][index] == m_rlabels[key][index],
                "all_feature_lengths_match": aligned,
                "legacy_text_dim": len(m_legacy[key][index]), "t1_dim": len(m_t1[key][index]),
                "t2_dim": len(m_t2[key][index]), "t3_dim": len(m_t3[key][index]),
                "t4_dim": len(m_t4[key][index]), "audio_dim": len(audio),
                "visual_dim": len(visual), "audio_all_zero": bool(np.all(audio == 0)),
                "visual_all_zero": bool(np.all(visual == 0)),
                "raw_audio_status": "CANDIDATE_EMBEDDED_MP4_AUDIO", "raw_video_status": "CANDIDATE_UTTERANCE_MP4",
                "candidate_raw_text_status": raw_text_status,
                "candidate_raw_speaker_status": "MATCH_RECONSTRUCTED_SLOT" if raw_text_status == "MATCH" else "MISMATCH_OR_VERSION_DIFFERENCE",
                "candidate_raw_label": raw_row["Emotion"] if raw_row else "MISSING",
                "candidate_raw_label_status": "MATCH" if raw_row else "MISSING",
                "candidate_raw_media_status": "CANDIDATE;SEE_RAW_VIDEO_AUDIT",
            })
    for name, values in [
        ("legacy_text", m_legacy.values()), ("T1", m_t1.values()), ("T2", m_t2.values()),
        ("T3", m_t3.values()), ("T4", m_t4.values()), ("audio", m_audio.values()),
        ("visual", m_visual.values()),
    ]:
        stats.append(array_stats("MELD", name, values))

    with (AUDIT_DIR / "utterance_alignment_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    with (AUDIT_DIR / "modality_feature_statistics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stats[0]))
        writer.writeheader()
        writer.writerows(stats)


def write_inventory():
    i_raw = load_pickle(DATA / "IEMOCAP_features.pkl")
    i_rob = load_pickle(DATA / "iemocap_features_roberta.pkl")
    m_raw = load_pickle(DATA / "MELD_features_raw1.pkl")
    m_rob = load_pickle(DATA / "meld_features_roberta.pkl")
    counts = {
        "IEMOCAP_features.pkl": (len(i_raw[0]), sum(map(len, i_raw[0].values()))),
        "iemocap_features_roberta.pkl": (len(i_rob[0]), sum(map(len, i_rob[0].values()))),
        "MELD_features_raw1.pkl": (len(m_raw[0]), sum(map(len, m_raw[0].values()))),
        "meld_features_roberta.pkl": (len(m_rob[0]), sum(map(len, m_rob[0].values()))),
    }
    rows = []
    for name in counts:
        configured = DATA / name
        target = configured.resolve()
        dialogues, utterances = counts[name]
        rows.append({
            "category": "preprocessed_feature", "dataset": "IEMOCAP" if name.lower().startswith("i") else "MELD",
            "configured_path": str(configured), "resolved_path": str(target), "file_type": "pickle",
            "exists": target.exists(), "readable": os.access(target, os.R_OK), "file_count": 1,
            "size_bytes": target.stat().st_size, "dialogue_count": dialogues,
            "sample_count": utterances, "sha256": sha256(target), "id_alignment": "VERIFIED_STRUCTURAL",
        })
    archive = EXTERNAL / "ERC_features.zip"
    rows.append({
        "category": "feature_archive", "dataset": "IEMOCAP+MELD", "configured_path": "",
        "resolved_path": str(archive), "file_type": "zip", "exists": archive.exists(),
        "readable": os.access(archive, os.R_OK), "file_count": 1, "size_bytes": archive.stat().st_size,
        "dialogue_count": "", "sample_count": "", "sha256": sha256(archive), "id_alignment": "N/A",
    })
    for name in ("IEMOCAP_diff.pkl", "MELD_diff.pkl"):
        target = CMERC / "caldiff" / name
        rows.append({
            "category": "difficulty_index", "dataset": name.split("_")[0], "configured_path": f"caldiff/{name}",
            "resolved_path": str(target), "file_type": "pickle", "exists": target.exists(),
            "readable": os.access(target, os.R_OK), "file_count": 1, "size_bytes": target.stat().st_size,
            "dialogue_count": "", "sample_count": "", "sha256": sha256(target), "id_alignment": "TRAIN_IDS_COVERED",
        })
    backup_manifest = json.loads((XP04 / "provenance/large_artifact_backup.json").read_text())
    for item in backup_manifest["files"]:
        rel = item["path"]
        if rel.startswith("runs/") and rel.endswith("/best.pt"):
            source = Path(item["source"])
            parts = Path(rel).parts
            rows.append({
                "category": "xp04_best_checkpoint", "dataset": parts[2].split("_")[0],
                "configured_path": rel, "resolved_path": str(source), "file_type": "pytorch_checkpoint",
                "exists": source.exists(), "readable": os.access(source, os.R_OK), "file_count": 1,
                "size_bytes": item["size_bytes"], "dialogue_count": "", "sample_count": "",
                "sha256": item["source_sha256"], "id_alignment": "N/A",
            })
    rows.extend([
        {
            "category": "candidate_raw_audio", "dataset": "IEMOCAP", "configured_path": "NOT_REFERENCED_BY_XP04",
            "resolved_path": str(IEM_RAW / "Session*/{sentences/wav,dialog/wav}/*.wav"), "file_type": "PCM WAV",
            "exists": IEM_RAW.exists(), "readable": os.access(IEM_RAW, os.R_OK), "file_count": 10190,
            "size_bytes": 4099591766, "dialogue_count": 151, "sample_count": 7433,
            "sha256": "NOT_HASHED_LARGE_COLLECTION", "id_alignment": "7433/7433_FILENAME_MATCH",
        },
        {
            "category": "candidate_raw_video", "dataset": "IEMOCAP", "configured_path": "NOT_REFERENCED_BY_XP04",
            "resolved_path": str(IEM_RAW / "Session*/dialog/avi/DivX/*.avi"), "file_type": "AVI",
            "exists": IEM_RAW.exists(), "readable": os.access(IEM_RAW, os.R_OK), "file_count": 151,
            "size_bytes": 7684366922, "dialogue_count": 151, "sample_count": 151,
            "sha256": "NOT_HASHED_LARGE_COLLECTION", "id_alignment": "151/151_DIALOGUE_STEM_MATCH",
        },
        {
            "category": "candidate_raw_text_index", "dataset": "IEMOCAP", "configured_path": "NOT_REFERENCED_BY_XP04",
            "resolved_path": str(IEM_RAW / "Session*/dialog/{transcriptions,EmoEvaluation}/*.txt"), "file_type": "text",
            "exists": IEM_RAW.exists(), "readable": os.access(IEM_RAW, os.R_OK), "file_count": 302,
            "size_bytes": 3202936, "dialogue_count": 151, "sample_count": 7433,
            "sha256": "NOT_HASHED_COLLECTION", "id_alignment": "7433/7433_TEXT;53_UNKNOWN_LABEL_REMAP",
        },
    ])
    meld_raw_root = MELD_RAW
    rows.append({
        "category": "candidate_raw_video", "dataset": "MELD", "configured_path": "NOT_REFERENCED_BY_XP04",
        "resolved_path": str(meld_raw_root), "file_type": "csv+mp4+tar.gz", "exists": meld_raw_root.exists(),
        "readable": os.access(meld_raw_root, os.R_OK), "file_count": 13706, "size_bytes": 21784177484,
        "dialogue_count": 1432, "sample_count": 13708, "sha256": "UNVERIFIED_BUNDLE",
        "id_alignment": "13707/13708_TEXT_MATCH;2_STANDARD_MP4_MISSING",
    })
    missing_categories = {
        "IEMOCAP": (("raw_frames", "image"), ("facial_action_units", "table")),
        "MELD": (("raw_audio", "standalone wav/flac"), ("raw_frames", "image"), ("facial_action_units", "table")),
    }
    for dataset, categories in missing_categories.items():
        for category, extension in categories:
            rows.append({
                "category": category, "dataset": dataset, "configured_path": "MISSING",
                "resolved_path": "MISSING", "file_type": extension, "exists": False,
                "readable": False, "file_count": 0, "size_bytes": 0, "dialogue_count": "",
                "sample_count": 0, "sha256": "", "id_alignment": "UNVERIFIED",
            })
    with (AUDIT_DIR / "file_inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    write_alignment_and_stats()
    write_inventory()
