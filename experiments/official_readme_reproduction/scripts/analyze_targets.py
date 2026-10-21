"""Read-only HSCL target diagnostics from fixed-checkpoint replay predictions."""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]


def main():
    result = {"scope":"descriptive fixed-checkpoint test diagnostics; no training, tuning or epoch selection",
        "definition":"official target1 iff full prediction correct and removal wrong, or both correct and full true-class log probability strictly higher",
        "limitations":["target balance is not independently verified counterfactual label quality",
            "target0 combines several different cases including both predictions wrong",
            "selected checkpoints depend on test selection; no unbiased generalization claim"],"runs":{}}
    classification = {"scope":result["scope"],"runs":{},"paired_checkpoint_changes":{}}
    predictions_by_run = {}
    for config in sorted((ROOT/"configs").glob("*.json")):
        cfg = json.loads(config.read_text())
        ident = cfg["run_id"]
        metrics = json.loads((ROOT/"results"/ident/"metrics.json").read_text())
        arrays = np.load(ROOT/"results"/ident/"predictions.npz")
        labels = arrays["labels"]
        full = arrays["full_log_probs"]
        correct = full.argmax(1)==labels
        predictions = full.argmax(1)
        predictions_by_run[ident] = (labels,predictions)
        class_ids = np.arange(full.shape[1])
        precision,recall,f1,support = precision_recall_fscore_support(labels,predictions,labels=class_ids,zero_division=0)
        classification["runs"][ident] = {"selected_epoch":metrics["selected_epoch_1based"],
            "confusion_matrix_true_rows_pred_columns":confusion_matrix(labels,predictions,labels=class_ids).tolist(),
            "classes":{str(c):{"precision_percent":float(precision[c]*100),"recall_percent":float(recall[c]*100),
                "f1_percent":float(f1[c]*100),"support":int(support[c])} for c in class_ids}}
        rows = np.arange(len(labels))
        records = {}
        for name in ("text","audio","visual","intra_context","inter_context"):
            removed = arrays[name]
            removed_correct = removed.argmax(1)==labels
            lower = full[rows,labels]>removed[rows,labels]
            target1 = correct & (~removed_correct | (removed_correct & lower))
            both_wrong = ~correct & ~removed_correct
            target0 = ~target1
            records[name] = {"target1_count":int(target1.sum()),"target0_count":int(target0.sum()),
                "target1_fraction":float(target1.mean()),
                "both_wrong_default_zero_count":int(both_wrong.sum()),
                "both_wrong_fraction_within_target0":float(both_wrong.sum()/target0.sum()) if target0.any() else None,
                "full_wrong_removed_correct_count":int((~correct & removed_correct).sum()),
                "full_correct_removed_wrong_count":int((correct & ~removed_correct).sum()),
                "positive_cc_fraction":float((removed[rows,labels]>full[rows,labels]).mean()),
                "by_true_class":{str(c):{"count":int((labels==c).sum()),"target1_fraction":float(target1[labels==c].mean())} for c in np.unique(labels)}}
        result["runs"][ident] = {"selected_epoch":metrics["selected_epoch_1based"],"full_accuracy_percent":float(correct.mean()*100),"removals":records}
    path = ROOT/"results/hscl_target_diagnostics.json"
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    for dataset,seed in (("IEMOCAP",1475),("MELD",67137)):
        labels,baseline = predictions_by_run[f"{dataset}_baseline_seed{seed}"]
        labels2,cmerc = predictions_by_run[f"{dataset}_cmerc_seed{seed}"]
        assert np.array_equal(labels,labels2)
        classification["paired_checkpoint_changes"][dataset] = {
            "both_correct":int(((baseline==labels)&(cmerc==labels)).sum()),
            "both_wrong":int(((baseline!=labels)&(cmerc!=labels)).sum()),
            "baseline_wrong_cmerc_correct":int(((baseline!=labels)&(cmerc==labels)).sum()),
            "baseline_correct_cmerc_wrong":int(((baseline==labels)&(cmerc!=labels)).sum()),
            "caveat":"separately test-selected checkpoints; descriptive paired predictions, not evidence of an unbiased treatment effect"}
    (ROOT/"results/classification_diagnostics.json").write_text(json.dumps(classification,indent=2,allow_nan=False)+"\n")
    print('Saved',path)


if __name__ == "__main__":
    main()
