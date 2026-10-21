"""Compare real capacity retries and epoch order fingerprints, read-only."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    old = ROOT.parents[1]/"experiment_records/04_xp_mainline_validation"
    result = {"created_at_utc":datetime.now(timezone.utc).isoformat(),"xp04":{},"readme_completed_runs":{}}
    for dataset,seed in (("IEMOCAP",1475),("MELD",67137)):
        for variant in ("baseline","cmerc"):
            ident = f"{dataset}_{variant}_seed{seed}"
            source = old/"runs"/variant/f"{dataset}_validation_seed{seed}"/"metrics.jsonl"
            rows = [json.loads(line) for line in source.read_text().splitlines()]
            assert all("sampler_retries" in row for row in rows)
            result["xp04"][ident] = {"source":str(source),"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
                "epochs":len(rows),"total_sampler_retries":sum(row["sampler_retries"] for row in rows),
                "epochs_with_sampler_retries":sum(row["sampler_retries"]>0 for row in rows),
                "max_sampler_retries":max(row["sampler_retries"] for row in rows)}
            run = ROOT/"checkpoints"/ident
            if json.loads((run/"status.json").read_text())["status"]!="completed":
                continue
            source = run/"batches.jsonl"
            batches = [json.loads(line) for line in source.read_text().splitlines()]
            orders = []
            for epoch in range(1,len(rows)+1):
                ids = [key for batch in batches if batch["train"] and batch["epoch"]==epoch for key in batch["ids"]]
                orders.append({"epoch":epoch,"dialogues":len(ids),"order_sha256":hashlib.sha256(json.dumps(ids,separators=(",",":")).encode()).hexdigest()})
            full_count = max(row["dialogues"] for row in orders)
            full_orders = [row for row in orders if row["dialogues"]==full_count]
            result["readme_completed_runs"][ident] = {"source":str(source),"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
                "full_pool_epochs":len(full_orders),"distinct_full_pool_order_hashes":len({row["order_sha256"] for row in full_orders}),"orders":orders}
    path = ROOT/"provenance/batch_runtime_comparison.json"
    path.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"xp04_total_retries":{key:value["total_sampler_retries"] for key,value in result["xp04"].items()},
        "completed_readme_runs":len(result["readme_completed_runs"])}))


if __name__ == "__main__":
    main()
