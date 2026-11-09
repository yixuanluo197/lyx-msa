# Real-model Batch-1 Removal Adapter Parity

The real-model gate passed in the XP04 environment (`torch 1.13.1+cu117`) on GPU 0. It used Fold 1 Training dialogue `Ses03F_impro03` with 35 utterances: 27 valid labels and 8 masked `xxx` utterances. All 35 nodes remained in the full, text-removed, audio-removed, visual-removed, intra-removed, and inter-removed forwards.

The reference adapter is extracted independently from the original `caldiff/cal_diff.py` constructor and forward statements. The generator adapter uses the frozen `model_cm.Model`, `Model_Intra`, and `Model_Inter`. Fresh seeded weights were strictly loaded into all eight real model instances; no historical checkpoint was used for this pre-training implementation gate.

All six raw-logit and log-probability tensors passed `atol=1e-6, rtol=1e-5`. The five valid-label gaps, speaker term, total gap, primary difficulty, and total-length control also passed. The largest final reported differences were `5.96e-7` for total gap and `2.06e-8` for final difficulty. Replacing all masked labels by `-100` left every reduction unchanged, proving that masked rows are excluded before true-class gather while remaining in neural context.

An initial XP04 rerun revealed that the comparison code had implemented absolute tolerance only. The reference and reducer were not changed; the comparator was corrected to the preregistered combined absolute/relative bound and regression-tested. Six parity tests pass. Full tensors and GPU snapshots are saved in `results/real_removal_parity.pt`, `results/real_removal_parity.json`, and `logs/real_removal_parity_*`.
