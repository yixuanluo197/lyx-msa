# Final execution status

Execution coverage: **complete**; missing evaluations: 0.

Validation is the primary analysis. Test is the prespecified replication analysis and is reported separately.
IEMOCAP seed 1937 test results were already observed in round 2 and were reused without retraining or reselection.
MELD seed 1937 is the technical pilot; its test may precede the remaining fixed matrix. Configurations do not change after that exposure.
Deltas are CMERC minus Baseline. Classification/calibration are percentages, with deltas in percentage points.
CEL is a sum of positive true-class log-probability differences under `prefix_mean_v1`, a prefix smoothing view, not complete modality removal.
Three seeds on a fixed split give uncertain estimates. Sample SD uses ddof=1; df=2 t intervals rely on normality and are descriptive.
The exact two-sided sign-flip test has minimum p=0.25 for three seeds. No robust statistical significance or proven innovation is claimed.

Frozen training commit: `5e5d7dd6cb7811f3c0d7c64c4afcbcb4aed9f69d`.
Protocol SHA256: `ba896759f510f63da99770508c19181bd49e6075b932bb1581d390e02e1e77a2`.
Analysis matrix SHA256: `7aa66f4be751569def805e969cb6420018ba1a4e42aa69c461105f84e9db9127`.
Statistics SHA256: `211bfd22d8503847230668ad08ca41df4fec93f10bb0ee07af8e8da6a22e528e`.

| Dataset | Seed | Execution status | Baseline epochs | CMERC epochs | Validation analyzed | Test analyzed |
| --- | --- | --- | --- | --- | --- | --- |
| IEMOCAP | 1937 | round 2 reuse | 80 | 80 | True | True |
| IEMOCAP | 2025 | completed | 80 | 80 | True | True |
| IEMOCAP | 3407 | completed | 80 | 80 | True | True |
| MELD | 1937 | completed | 40 | 40 | True | True |
| MELD | 2025 | completed | 40 | 40 | True | True |
| MELD | 3407 | completed | 40 | 40 | True | True |

## Complete available paired outcomes

| Dataset | Split | Segment | Metric | n | Baseline mean ± SD | CMERC mean ± SD | Paired delta mean ± SD | 95% t interval |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAP | test | overall | cel.total | 3 | 1526.717601 ± 290.630721 | 1008.262681 ± 68.989191 | -518.454919 ± 347.668872 | [-1382.112274, 345.202436] |
| IEMOCAP | test | overall | ece | 3 | 18.184590 ± 3.955896 | 15.504165 ± 3.370485 | -2.680425 ± 7.298827 | [-20.811717, 15.450868] |
| IEMOCAP | test | overall | weighted_f1 | 3 | 64.520884 ± 1.553512 | 65.174537 ± 0.583886 | 0.653653 ± 2.086690 | [-4.529971, 5.837278] |
| IEMOCAP | validation | overall | cel.total | 3 | 562.080434 ± 137.526493 | 364.565065 ± 39.907195 | -197.515369 ± 128.639294 | [-517.073090, 122.042351] |
| IEMOCAP | validation | overall | ece | 3 | 21.959176 ± 2.895528 | 16.713749 ± 1.363865 | -5.245427 ± 3.364377 | [-13.603002, 3.112148] |
| IEMOCAP | validation | overall | weighted_f1 | 3 | 61.760864 ± 2.089200 | 62.038097 ± 2.250578 | 0.277233 ± 0.199791 | [-0.219076, 0.773543] |
| MELD | test | overall | cel.total | 3 | 676.771066 ± 104.648242 | 703.628436 ± 44.199627 | 26.857370 ± 125.389971 | [-284.628586, 338.343326] |
| MELD | test | overall | ece | 3 | 21.563912 ± 2.988031 | 17.938954 ± 1.529862 | -3.624958 ± 4.479939 | [-14.753745, 7.503828] |
| MELD | test | overall | weighted_f1 | 3 | 64.597559 ± 0.814811 | 64.855516 ± 0.477477 | 0.257957 ± 0.821146 | [-1.781883, 2.297797] |
| MELD | validation | overall | cel.total | 3 | 262.024176 ± 37.738682 | 272.135501 ± 10.907525 | 10.111326 ± 46.135918 | [-104.496648, 124.719300] |
| MELD | validation | overall | ece | 3 | 22.116732 ± 2.507416 | 18.689310 ± 2.532525 | -3.427422 ± 4.926491 | [-15.665503, 8.810659] |
| MELD | validation | overall | weighted_f1 | 3 | 64.361201 ± 0.354548 | 64.628710 ± 0.927207 | 0.267509 ± 0.680183 | [-1.422161, 1.957178] |

Positive W-F1 deltas and negative ECE/CEL deltas indicate the prespecified favorable directions; opposite directions remain in the same tables.

## All recorded attempts, including failures and recovery

| Pair | Stage | Attempt | Status | Return code | Started UTC | Ended UTC | Wall seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAP/2025 | baseline_training | 1 | completed | 0 | 2026-10-18T15:28:04.168472+00:00 | 2026-10-18T15:32:44.466614+00:00 | 280.297354 |
| IEMOCAP/2025 | baseline_verification | 1 | completed | 0 | 2026-10-18T15:32:45.841744+00:00 | 2026-10-18T15:32:53.508696+00:00 | 7.525775 |
| IEMOCAP/2025 | difficulty_generation | 1 | completed | 0 | 2026-10-18T15:32:56.673467+00:00 | 2026-10-18T15:33:18.153960+00:00 | 21.074738 |
| IEMOCAP/2025 | cmerc_training | 1 | completed | 0 | 2026-10-18T15:33:20.912379+00:00 | 2026-10-18T15:57:23.974715+00:00 | 1443.061096 |
| IEMOCAP/2025 | cmerc_verification | 1 | completed | 0 | 2026-10-18T15:57:25.869499+00:00 | 2026-10-18T15:57:33.482260+00:00 | 7.552417 |
| IEMOCAP/2025 | validation_baseline_evaluation | 1 | completed | 0 | 2026-10-18T15:57:35.363003+00:00 | 2026-10-18T15:57:45.023098+00:00 | 9.659213 |
| IEMOCAP/2025 | validation_cmerc_evaluation | 1 | completed | 0 | 2026-10-18T15:57:46.530637+00:00 | 2026-10-18T15:57:56.196256+00:00 | 9.664626 |
| IEMOCAP/2025 | validation_comparison | 1 | completed | 0 | 2026-10-18T15:57:57.613094+00:00 | 2026-10-18T15:58:00.056016+00:00 | 2.441945 |
| IEMOCAP/2025 | test_baseline_evaluation | 1 | completed | 0 | 2026-10-18T16:14:32.679873+00:00 | 2026-10-18T16:14:41.118909+00:00 | 8.438228 |
| IEMOCAP/2025 | test_cmerc_evaluation | 1 | completed | 0 | 2026-10-18T16:14:47.082801+00:00 | 2026-10-18T16:14:55.613599+00:00 | 8.529854 |
| IEMOCAP/2025 | test_comparison | 1 | completed | 0 | 2026-10-18T16:14:59.220330+00:00 | 2026-10-18T16:15:01.504984+00:00 | 2.283752 |
| IEMOCAP/3407 | baseline_training | 1 | completed | 0 | 2026-10-18T15:28:39.215347+00:00 | 2026-10-18T15:33:25.517480+00:00 | 286.301408 |
| IEMOCAP/3407 | baseline_verification | 1 | completed | 0 | 2026-10-18T15:33:26.916166+00:00 | 2026-10-18T15:33:35.194034+00:00 | 7.955474 |
| IEMOCAP/3407 | difficulty_generation | 1 | completed | 0 | 2026-10-18T15:33:38.383238+00:00 | 2026-10-18T15:33:59.415298+00:00 | 20.926500 |
| IEMOCAP/3407 | cmerc_training | 1 | completed | 0 | 2026-10-18T15:34:01.464812+00:00 | 2026-10-18T15:59:29.958225+00:00 | 1528.492701 |
| IEMOCAP/3407 | cmerc_verification | 1 | completed | 0 | 2026-10-18T15:59:31.872103+00:00 | 2026-10-18T15:59:39.420117+00:00 | 7.547186 |
| IEMOCAP/3407 | validation_baseline_evaluation | 1 | completed | 0 | 2026-10-18T15:59:41.209259+00:00 | 2026-10-18T15:59:49.845525+00:00 | 8.635416 |
| IEMOCAP/3407 | validation_cmerc_evaluation | 1 | completed | 0 | 2026-10-18T15:59:51.216221+00:00 | 2026-10-18T15:59:59.860399+00:00 | 8.643255 |
| IEMOCAP/3407 | validation_comparison | 1 | completed | 0 | 2026-10-18T16:00:01.255263+00:00 | 2026-10-18T16:00:05.473654+00:00 | 3.975047 |
| IEMOCAP/3407 | test_baseline_evaluation | 1 | completed | 0 | 2026-10-18T16:15:08.739553+00:00 | 2026-10-18T16:15:17.140608+00:00 | 8.400166 |
| IEMOCAP/3407 | test_cmerc_evaluation | 1 | completed | 0 | 2026-10-18T16:15:23.071344+00:00 | 2026-10-18T16:15:31.600917+00:00 | 8.528676 |
| IEMOCAP/3407 | test_comparison | 1 | completed | 0 | 2026-10-18T16:15:35.175099+00:00 | 2026-10-18T16:15:37.454119+00:00 | 2.278100 |
| MELD/1937 | baseline_training | 1 | completed | 0 | 2026-10-18T14:57:28.055822+00:00 | 2026-10-18T15:02:14.802511+00:00 | 286.746147 |
| MELD/1937 | baseline_verification | 1 | completed | 0 | 2026-10-18T15:02:16.109993+00:00 | 2026-10-18T15:02:24.593002+00:00 | 8.482453 |
| MELD/1937 | difficulty_generation | 1 | completed | 0 | 2026-10-18T15:02:28.114017+00:00 | 2026-10-18T15:02:58.723546+00:00 | 30.608882 |
| MELD/1937 | cmerc_training | 1 | completed | 0 | 2026-10-18T15:03:02.050271+00:00 | 2026-10-18T15:26:23.623764+00:00 | 1401.572653 |
| MELD/1937 | cmerc_verification | 1 | completed | 0 | 2026-10-18T15:26:26.393829+00:00 | 2026-10-18T15:26:35.919324+00:00 | 9.524650 |
| MELD/1937 | validation_baseline_evaluation | 1 | completed | 0 | 2026-10-18T15:26:38.763422+00:00 | 2026-10-18T15:26:49.347323+00:00 | 10.583046 |
| MELD/1937 | validation_cmerc_evaluation | 1 | completed | 0 | 2026-10-18T15:26:50.657663+00:00 | 2026-10-18T15:27:00.241221+00:00 | 9.582714 |
| MELD/1937 | validation_comparison | 1 | completed | 0 | 2026-10-18T15:27:01.564107+00:00 | 2026-10-18T15:27:03.826635+00:00 | 2.261609 |
| MELD/1937 | test_baseline_evaluation | 1 | completed | 0 | 2026-10-18T15:27:34.397194+00:00 | 2026-10-18T15:27:44.026739+00:00 | 9.628644 |
| MELD/1937 | test_cmerc_evaluation | 1 | completed | 0 | 2026-10-18T15:27:47.227285+00:00 | 2026-10-18T15:27:56.834447+00:00 | 9.606083 |
| MELD/1937 | test_comparison | 1 | completed | 0 | 2026-10-18T15:27:59.081712+00:00 | 2026-10-18T15:28:01.445413+00:00 | 2.362758 |
| MELD/2025 | baseline_training | 1 | completed | 0 | 2026-10-18T15:29:14.911605+00:00 | 2026-10-18T15:35:15.819446+00:00 | 360.907280 |
| MELD/2025 | baseline_verification | 1 | completed | 0 | 2026-10-18T15:35:17.188735+00:00 | 2026-10-18T15:35:25.472275+00:00 | 8.282965 |
| MELD/2025 | difficulty_generation | 1 | completed | 0 | 2026-10-18T15:35:29.047284+00:00 | 2026-10-18T15:36:07.083501+00:00 | 37.713617 |
| MELD/2025 | cmerc_training | 1 | completed | 0 | 2026-10-18T15:36:10.693378+00:00 | 2026-10-18T16:10:25.042466+00:00 | 2054.348221 |
| MELD/2025 | cmerc_verification | 1 | completed | 0 | 2026-10-18T16:10:27.714065+00:00 | 2026-10-18T16:10:38.180861+00:00 | 10.465851 |
| MELD/2025 | validation_baseline_evaluation | 1 | completed | 0 | 2026-10-18T16:10:40.255154+00:00 | 2026-10-18T16:10:49.779579+00:00 | 9.523581 |
| MELD/2025 | validation_cmerc_evaluation | 1 | completed | 0 | 2026-10-18T16:10:51.044576+00:00 | 2026-10-18T16:11:00.595003+00:00 | 9.549487 |
| MELD/2025 | validation_comparison | 1 | completed | 0 | 2026-10-18T16:11:01.860689+00:00 | 2026-10-18T16:11:04.136918+00:00 | 2.275380 |
| MELD/2025 | test_baseline_evaluation | 1 | completed | 0 | 2026-10-18T16:15:44.724671+00:00 | 2026-10-18T16:15:54.281351+00:00 | 9.555842 |
| MELD/2025 | test_cmerc_evaluation | 1 | completed | 0 | 2026-10-18T16:16:00.215486+00:00 | 2026-10-18T16:16:09.769373+00:00 | 9.552963 |
| MELD/2025 | test_comparison | 1 | completed | 0 | 2026-10-18T16:16:13.372963+00:00 | 2026-10-18T16:16:15.659969+00:00 | 2.286056 |
| MELD/3407 | baseline_training | 1 | completed | 0 | 2026-10-18T15:29:49.940055+00:00 | 2026-10-18T15:35:58.384038+00:00 | 368.443521 |
| MELD/3407 | baseline_verification | 1 | completed | 0 | 2026-10-18T15:36:00.206480+00:00 | 2026-10-18T15:36:08.774666+00:00 | 8.567616 |
| MELD/3407 | difficulty_generation | 1 | completed | 0 | 2026-10-18T15:36:12.411893+00:00 | 2026-10-18T15:36:56.082616+00:00 | 43.669989 |
| MELD/3407 | cmerc_training | 1 | completed | 0 | 2026-10-18T15:36:59.601429+00:00 | 2026-10-18T16:13:44.064583+00:00 | 2204.462487 |
| MELD/3407 | cmerc_verification | 1 | completed | 0 | 2026-10-18T16:13:46.715896+00:00 | 2026-10-18T16:13:56.166789+00:00 | 9.450133 |
| MELD/3407 | validation_baseline_evaluation | 1 | completed | 0 | 2026-10-18T16:13:58.248361+00:00 | 2026-10-18T16:14:07.786237+00:00 | 9.537064 |
| MELD/3407 | validation_cmerc_evaluation | 1 | completed | 0 | 2026-10-18T16:14:09.042961+00:00 | 2026-10-18T16:14:18.585357+00:00 | 9.541428 |
| MELD/3407 | validation_comparison | 1 | completed | 0 | 2026-10-18T16:14:19.836360+00:00 | 2026-10-18T16:14:22.097377+00:00 | 2.260109 |
| MELD/3407 | test_baseline_evaluation | 1 | completed | 0 | 2026-10-18T16:16:23.000534+00:00 | 2026-10-18T16:16:32.639376+00:00 | 9.556092 |
| MELD/3407 | test_cmerc_evaluation | 1 | completed | 0 | 2026-10-18T16:16:38.571366+00:00 | 2026-10-18T16:16:48.131886+00:00 | 9.559611 |
| MELD/3407 | test_comparison | 1 | completed | 0 | 2026-10-18T16:16:51.752134+00:00 | 2026-10-18T16:16:54.358539+00:00 | 2.605480 |

Epoch counts for unadmitted/running runs are last complete log entries, not proof of checkpoint completeness. Exact state/metric snapshot hashes are retained in REPORT_EVIDENCE.json.


## Checkpoint and optimizer evidence

| Pair | Model | Best epoch | Best SHA256 | Latest SHA256 | Initial state SHA256 | Derived updates | Actual Adam updates |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAP/1937 | baseline | 57 | 1d9bbae19f0e4f8bd976bc3e4285f93c14f114cc54e6d28afcc76be8f90dd3bc | 43f074746530571c614f281c56217dbb622ca400acdbaad7f439d72fa9a93c47 | de943257bf89c6aa53abc205c794f22f17e79aa40ed0a1701a08b1cd6aa0752f | 560 | 560 |
| IEMOCAP/1937 | cmerc | 48 | d4e1501b1082f3f4cc951b65bbac5fc0b78779d3306ce671e3a4c613355650f4 | ebfa7644d4b67d5671d41b57d41dc82813fb0ab50273635dea044abaf263897b | de943257bf89c6aa53abc205c794f22f17e79aa40ed0a1701a08b1cd6aa0752f | 541 | 541 |
| IEMOCAP/2025 | baseline | 79 | ab50fdbadca965e2fe7e96c6e6f70fa5667d55cf75dfb56b01a8d6160e46f26e | 2cad48619b65750b6c3f6a2ee4ddf694e657da8c8243103dd639ad26cfb0c783 | cba58c1d9eda86d639ee41136ecd814c753d93269057c96d84ecb9bc99f0ef61 | 560 | 560 |
| IEMOCAP/2025 | cmerc | 51 | 919ff77488df350954bc38755cf8678c5fd09b30b5ba3d7d728e35ac2a7cf36f | a239fb69e466521aefb9f625f45ca1b2b665ba94477037c5d033e3778f01a70d | cba58c1d9eda86d639ee41136ecd814c753d93269057c96d84ecb9bc99f0ef61 | 541 | 541 |
| IEMOCAP/3407 | baseline | 49 | 476d940ffba04353f238f61013286d05494699248d486566324d70c2de880413 | dbd8b2f880b73648b9c0b836dcde30387e06d9a098f685750bf72ccd6b1db59e | 478056a63566540220344702bd8e397fd7f24411527386938629989a208b3ff1 | 560 | 560 |
| IEMOCAP/3407 | cmerc | 79 | 370eb5488e9e12dfec881855318702721a7677941694e52e2fdd629d7e335283 | 02125782570495e957910300dbac7e59d64f6e157dca31a38dfc59e3a686a842 | 478056a63566540220344702bd8e397fd7f24411527386938629989a208b3ff1 | 541 | 541 |
| MELD/1937 | baseline | 8 | 5b6d298b4efdef7ce4cf1f58254083f47865797f58e079ee0d5ab58e99429f3a | d7498d0ebe0e09249198cd355d04c72aea3599efdd2fc0cdbc67f4f72af73677 | dbe4ad864b0ed081436310a030b622d94a3b542b138ec02156bb6db95647266d | 2600 | 2600 |
| MELD/1937 | cmerc | 8 | 9bc08483758bab023d8e223d00e735f156dc073449f90371678d36a7379c8223 | 8ba3175cc935b6e7cd7658318153edd9b02dd65527097659a62ea4b8097a1a0d | dbe4ad864b0ed081436310a030b622d94a3b542b138ec02156bb6db95647266d | 2548 | 2548 |
| MELD/2025 | baseline | 8 | 0153de90e1fbaac822ef556b9c9c0c052e1c27657c871b7277f193dc037117cc | 503a3cbf5c9468ca56a7330d4d3d27155b223b6b313cb7076e0a74a6725e4ce3 | 5a2f2157512d44b03dee430fd9de205b2581493db3c9647676d9957890b1cea3 | 2600 | 2600 |
| MELD/2025 | cmerc | 8 | 2a0a2662f8742a48760e0a2b8b13cb826d6d153bb04981d40361827c1bf62434 | 1e006eede0023d0c222d5a2a4af1d823b2e555f7aba65afddae8ff8e5915ad2d | 5a2f2157512d44b03dee430fd9de205b2581493db3c9647676d9957890b1cea3 | 2548 | 2548 |
| MELD/3407 | baseline | 5 | d05e2d2bf0bfa5c45b5911d0921b826ad3bc325fbb4d6ad9fc13b7536ff3a272 | 6ebb1e0a71a4d51de7da60c84ba52948a5cdbed0c037c3db613f07a6056b382f | 0aa01a66b63b9979dde158ebd35ce408b9420c1968b03c58c96a11ec91eb261d | 2600 | 2600 |
| MELD/3407 | cmerc | 9 | 49678341f3abfaf29c4f327e5ebaaa13fd17ae1a51f4a56f7b5cbfbde2c24c30 | ab61ec060132c131e2d19069bbc47142faf45354e93180cf1f16105b2f347cad | 0aa01a66b63b9979dde158ebd35ce408b9420c1968b03c58c96a11ec91eb261d | 2548 | 2548 |

## Evidence boundary

Completed formal budgets and finite/update/provenance checks are distinct from performance improvement. Negative results are retained.
Missing runs/evaluations are not replaced with smoke tests. No formal performance conclusion is drawn for missing cells.
All model weights, features, dialogue IDs, predictions and large training logs remain on the experiment server. Public outputs contain aggregate values and hashes.
Pre-extracted feature temporal causality is the audited boundary; upstream feature extraction remains unaudited.
Synthetic causality tests were not repeated in this round. Three fixed-split seeds do not establish broad generalization.
