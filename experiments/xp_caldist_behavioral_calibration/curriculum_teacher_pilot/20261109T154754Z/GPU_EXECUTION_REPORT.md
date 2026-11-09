# GPU Execution Report

Preflight found four idle NVIDIA TITAN Xp GPUs and no compute processes. `/data1` had 75 GiB free and `/data2` had 134 GiB free. The formal schedule used only host GPU 0 for IEMOCAP and host GPU 1 for MELD; GPUs 2 and 3 remained unused. Each subprocess saw one device as `cuda:0`, with `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, and `num_workers=0`. DDP was not used.

The successful training jobs consumed 0.0942 and 0.0458 GPU-hours, respectively (0.1400 total GPU-hours). Parallel training wall time was governed by IEMOCAP at 339.09 seconds. GPU UUIDs, sampled memory/utilization, peak allocator values, PIDs, and exit codes are retained under `logs/` and checkpoint provenance.

The first launch was intentionally stopped after diagnosing an over-capacity legacy-shuffle batch. It completed 10 IEMOCAP and 25 MELD epochs before quarantine; neither partial checkpoint was used. The corrected jobs restarted from random initialization with XP04 capacity-safe batching and both exited with code 0. Difficulty inference then ran independently on GPUs 0 and 1 and also exited with code 0.
