#!/usr/bin/env python3
"""Candidate acoustic extraction, never an assertion of original provenance.

Only the supplied effective-Train manifest is accepted by default. Validation
requires the root experiment's explicit model-lock file. All rows are retained;
failed rows remain NaN, rather than being silently replaced by zero vectors.
"""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
import argparse
import concurrent.futures
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time
import numpy as np
import opensmile
from scipy.io import wavfile

OUT = Path(__file__).resolve().parents[1]
PREVIOUS = OUT.parent / "20261117T014409Z"
CANDIDATES = ("is10", "is10_compat", "emobase2010")
CONFIGS = OUT / "configs" / "iemocap"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def setup():
    source = PREVIOUS / "sources/opensmile/v3.0.1/config"
    copy = CONFIGS / "official_v3.0.1"
    if not copy.exists():
        shutil.copytree(source, copy)
    package = Path(opensmile.__file__).parent / "core/config/shared"
    for name in ("standard_external_wave_input.conf.inc", "standard_external_data_output_single.conf.inc"):
        target = copy / "shared" / name
        if not target.exists():
            shutil.copy2(package / name, target)
    wrapper_paths = {}
    for candidate, rel in (("is10", "is09-13/IS10_paraling.conf"),
                           ("is10_compat", "is09-13/IS10_paraling_compat.conf"),
                           ("emobase2010", "emobase/emobase2010.conf")):
        original = copy / rel
        text = original.read_bytes().decode("latin1")
        if candidate != "emobase2010":
            text = text.replace(r"\{../shared/standard_wave_input.conf.inc}", r"\{\cm[source{?}:external input]}")
            text = text.replace(r"\{../shared/standard_data_output.conf.inc}", r"\{\cm[sink{?}:external output]}")
            if candidate == "is10_compat":
                text = re.sub(r"^instance\[waveIn\]\.type=cWaveSource\s*$", "", text, flags=re.M)
                text = re.sub(r"\[waveIn:cWaveSource\].*?(?=\[fr40:cFramer\])", r"\\{\\cm[source{?}:external input]}\n\n", text, flags=re.S)
        else:
            text = re.sub(r"^instance\[waveIn\]\.type=cWaveSource\s*$", "", text, flags=re.M)
            text = re.sub(r"^instance\[arffsink\]\.type=cArffSink\s*$", "", text, flags=re.M)
            text = re.sub(r"\[waveIn:cWaveSource\].*?(?=\[fr40:cFramer\])", r"\\{\\cm[source{?}:external input]}\n\n", text, flags=re.S)
            text = text[:text.index("[arffsink:cArffSink]")] + """
[componentInstances:cComponentManager]
instance[funcconcat].type=cVectorConcat
[funcconcat:cVectorConcat]
reader.dmLevel=func;funcNz;functOnsets
writer.dmLevel=func_out
includeSingleElementFields=1
\\{\\cm[sink{?}:external output]}
"""
        destination = original.with_name(original.stem + ".python_external.conf")
        if not destination.exists():
            destination.write_bytes(text.encode("latin1"))
        elif destination.read_bytes() != text.encode("latin1"):
            raise RuntimeError("Refusing to replace divergent candidate config: " + str(destination))
        wrapper_paths[candidate] = str(destination)
    lib = Path(opensmile.__file__).parent / "core/bin/manylinux_2_17_x86_64/libSMILEapi.so"
    record = {
        "method_status": "candidate_only_not_original_production_configuration",
        "python_version": sys.version, "opensmile_python_version": opensmile.__version__,
        "engine_banner_version": "3.0.1", "library_path": str(lib), "library_sha256": sha(lib),
        "source_tag": "v3.0.1", "candidate_configs": wrapper_paths,
        "config_tree": [{"path": str(p.relative_to(CONFIGS)), "sha256": sha(p), "bytes": p.stat().st_size}
                        for p in sorted(copy.rglob("*")) if p.is_file()],
        "wrapper_modifications": {
            "is10_and_compat": "Replace only file source and standard sinks by Python external source/sink includes; compat embeds its cWaveSource instance/section, so remove those and insert external source.",
            "emobase2010": "Remove cWaveSource/cArffSink instance+sections; add external source and a VectorConcat(func;funcNz;functOnsets) into func_out plus external sink. DSP/functionals preserved."
        },
        "decoding": "scipy.io.wavfile.read; signed PCM divided by 2**(bits-1), unsigned8 centered; original float retained; float32 API signal",
        "sample_rate": 16000, "channel_rule": "require released single-channel utterance WAV (no resampling or downmix)",
        "amplitude_normalization": "PCM integer-to-unit scaling only; no per-utterance gain, no z-score",
        "intensity_threshold": None, "silence_trimming": False, "feature_normalization": None,
        "threads_per_process": 1, "maximum_worker_processes": 6,
    }
    provenance = OUT / "provenance/iemocap_extraction_configs.json"
    if not provenance.exists():
        write_json(provenance, record)
    return wrapper_paths


SMILES = {}
def initialize(config_paths):
    for candidate, config in config_paths.items():
        SMILES[candidate] = opensmile.Smile(feature_set=config,
            feature_level="func_out" if candidate == "emobase2010" else "func",
            sampling_rate=16000, channels=0, mixdown=False, resample=False,
            loglevel=0, num_workers=1)


def decode(path):
    rate, values = wavfile.read(path)
    if rate != 16000 or values.ndim != 1:
        raise ValueError("Candidate precondition failed: released 16k mono required; got " + str((rate, values.shape)))
    original_dtype = str(values.dtype)
    if np.issubdtype(values.dtype, np.signedinteger):
        signal = values.astype(np.float32) / float(2 ** (8 * values.dtype.itemsize - 1))
    elif values.dtype == np.uint8:
        signal = (values.astype(np.float32) - 128.0) / 128.0
    elif np.issubdtype(values.dtype, np.floating):
        signal = values.astype(np.float32)
    else:
        raise ValueError("Unsupported WAV sample type " + original_dtype)
    if not signal.size or not np.isfinite(signal).all():
        raise ValueError("Empty/nonfinite decoded audio")
    return signal, rate, original_dtype


def block_extract(task):
    block_number, rows, directory = task
    directory = Path(directory)
    vectors = {name: np.full((len(rows), 1582), np.nan, np.float32) for name in CANDIDATES}
    status = []
    start = time.monotonic()
    for j, row in enumerate(rows):
        item = {"row_index": int(row["row_index"]), "utterance_id": row["utterance_id"], "candidates": {}}
        try:
            signal, rate, dtype = decode(row["audio_path"])
            item.update(audio_sha256=sha(row["audio_path"]), sampling_rate=rate,
                        samples=int(signal.size), source_dtype=dtype,
                        decoded_sha256=hashlib.sha256(signal.tobytes()).hexdigest(),
                        peak=float(np.max(np.abs(signal))), rms=float(np.sqrt(np.mean(signal.astype(float)**2))))
            for name, extractor in SMILES.items():
                try:
                    frame = extractor.process_signal(signal, rate)
                    values = frame.to_numpy()
                    if values.shape != (1, 1582) or not np.isfinite(values).all():
                        raise ValueError("Unexpected shape or nonfinite output: " + str(values.shape))
                    vectors[name][j] = values[0]
                    item["candidates"][name] = {"status": "ok", "shape": [1582], "feature_sha256": hashlib.sha256(values[0].astype(np.float32).tobytes()).hexdigest()}
                except Exception as error:
                    item["candidates"][name] = {"status": "error", "error": repr(error)}
        except Exception as error:
            item["audio_error"] = repr(error)
            for name in CANDIDATES:
                item["candidates"][name] = {"status": "error", "error": repr(error)}
        status.append(item)
    stem = directory / ("block_%05d" % block_number)
    with open(str(stem) + ".npz.tmp", "wb") as handle:
        np.savez(handle, **vectors)
    Path(str(stem) + ".npz.tmp").replace(str(stem) + ".npz")
    write_json(Path(str(stem) + ".json"), {"rows": status, "elapsed_seconds": time.monotonic() - start})
    return {"block": block_number, "rows": len(rows), "seconds": time.monotonic()-start,
            "errors": sum(s["status"] != "ok" for r in status for s in r["candidates"].values())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--block-size", type=int, default=40)
    parser.add_argument("--model-lock", type=Path)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 6:
        parser.error("workers must be in [1,6]")
    if args.split == "validation" and (not args.model_lock or not args.model_lock.exists()):
        parser.error("Validation requires an explicit existing model-lock file from the root experiment")
    config_paths = setup()
    with open(args.manifest, newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row.setdefault("audio_path", row.get("media_path"))
        row.setdefault("utterance_id", row.get("raw_utterance_id"))
    if len(rows) != (5283 if args.split == "train" else 527):
        raise ValueError("Effective split row count mismatch")
    if any(row["split"].lower() != args.split for row in rows):
        raise ValueError("Manifest split mismatch")
    if args.limit:
        rows = rows[:args.limit]
    for i, row in enumerate(rows):
        row["row_index"] = str(i)
    run_name = args.split + ("_smoke_%d" % args.limit if args.limit else "")
    base = OUT / "local_only/IEMOCAP/candidates"
    blocks = base / "blocks" / run_name
    blocks.mkdir(parents=True, exist_ok=True)
    run = {"split": args.split, "rows": len(rows), "candidate_names": CANDIDATES,
           "manifest": str(args.manifest), "manifest_sha256": sha(args.manifest),
           "workers": args.workers, "block_size": args.block_size,
           "command": sys.argv, "pid": os.getpid(), "started_epoch": time.time(),
           "validation_model_lock": str(args.model_lock) if args.model_lock else None}
    write_json(OUT / ("provenance/iemocap_extraction_%s_run.json" % run_name), run)
    initialize(config_paths)
    for name, extractor in SMILES.items():
        names = list(extractor.feature_names)
        if len(names) != 1582 or len(set(names)) != 1582:
            raise ValueError("Invalid feature header " + name)
        (base / name).mkdir(parents=True, exist_ok=True)
        header = base / name / "header.json"
        if header.exists() and json.loads(header.read_text()) != names:
            raise ValueError("Candidate feature header changed")
        write_json(header, names)
    tasks = [(i//args.block_size, rows[i:i+args.block_size], str(blocks)) for i in range(0, len(rows), args.block_size)]
    pending = []
    for task in tasks:
        stem = blocks / ("block_%05d" % task[0])
        if Path(str(stem)+".json").exists() and Path(str(stem)+".npz").exists():
            old = json.loads(Path(str(stem)+".json").read_text())
            if [r["utterance_id"] for r in old["rows"]] != [r["utterance_id"] for r in task[1]]:
                raise ValueError("Resume block manifest mismatch")
        else:
            pending.append(task)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, initializer=initialize, initargs=(config_paths,)) as executor:
        futures = [executor.submit(block_extract, task) for task in pending]
        for future in concurrent.futures.as_completed(futures):
            print(json.dumps(future.result()), flush=True)
    all_status = []
    results = {name: np.full((len(rows),1582),np.nan,np.float32) for name in CANDIDATES}
    for number, task_rows, _ in tasks:
        stem = blocks / ("block_%05d" % number)
        record = json.loads(Path(str(stem)+".json").read_text())
        all_status.extend(record["rows"])
        with np.load(str(stem)+".npz") as arrays:
            begin = number*args.block_size
            for name in CANDIDATES:
                results[name][begin:begin+len(task_rows)] = arrays[name]
    outputs = {}
    for name in CANDIDATES:
        path = base / name / (run_name + ".npy")
        np.save(path, results[name])
        valid = np.isfinite(results[name]).all(axis=1)
        np.save(path.with_name(run_name+"_valid.npy"),valid)
        status_path = path.with_name(run_name+"_status.jsonl")
        status_path.write_text("".join(json.dumps({**{k:v for k,v in item.items() if k != "candidates"}, **item["candidates"][name]}) + "\n" for item in all_status))
        outputs[name] = {"path": str(path), "sha256": sha(path), "shape": list(results[name].shape),
                         "valid_rows": int(valid.sum()), "failed_rows": int((~valid).sum()),
                         "header_sha256": sha(path.with_name("header.json")), "status_sha256": sha(status_path)}
    run.update(completed_epoch=time.time(), elapsed_seconds=time.time()-run["started_epoch"], outputs=outputs)
    write_json(OUT / ("provenance/iemocap_extraction_%s_run.json" % run_name), run)
    print(json.dumps(run), flush=True)


if __name__ == "__main__":
    main()
