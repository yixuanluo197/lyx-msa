# GitHub Delivery

This directory is the reviewable GitHub package for the completed CMERC Protocol B five-fold, five-seed training and calibration stage.

## Scientific status

- 10/10 fold-local Teachers completed or were verified as approved Pilot reuse.
- 10/10 fold-local Difficulty artifacts passed exact role and ID coverage checks.
- 50/50 CMERC models completed or were verified as approved Pilot reuse; all best checkpoints passed strict reload.
- 50/50 Calibration jobs completed and passed deterministic replay checks.
- Independent Evaluation remained sealed: no Evaluation loader, forward pass, labels, probabilities, or metrics were accessed.
- Calibration grouped OOF results are internal Calibration diagnostics and are not independent Evaluation results.

## Published scope

The GitHub package contains the frozen protocol and manifests, executable experiment code, tests, compact run and calibration tables, grouped OOF predictions, final reports, provenance hashes, and verification metadata. These files are sufficient to inspect the protocol, audit task completion, and reproduce the reported Calibration-stage summaries without presenting them as final Evaluation evidence.

## Server-only artifacts

Large and redundant runtime artifacts remain on the experiment server:

- Teacher and CMERC checkpoints, optimizer/RNG state, and training-role artifacts (`*.pt`);
- full probability arrays and six-view probability archives (`*.npz`);
- task attempts, caches, process state, and detailed runtime logs;
- duplicate copies of tables already published at the experiment root.

The excluded artifacts occupy approximately 13 GiB. Their identities and hashes are retained in `verification.json`, `provenance/source_hashes.json`, the run tables, and the final verification metadata. Model weights are intentionally not committed to Git.

## Publication target

- Repository: `https://github.com/yixuanluo197/lyx-msa`
- Branch: `main`
- Published path: `experiments/xp_caldist_behavioral_calibration/formal_fivefold_fiveseed/20261109T174024Z/`
