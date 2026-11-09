# GPU Resource Check and Scheduling Plan

At preflight, four local NVIDIA TITAN Xp GPUs were visible. Each had 12,288 MiB total memory, 12,181 MiB free, 0% utilization, and no compute process. UUIDs and the captured `nvidia-smi` values are stored in [results/gpu_preflight.json](results/gpu_preflight.json). `/data1` had 75 GiB free and `/data2` had 135 GiB free.

The Pilot queue will bind IEMOCAP Fold 1 Seed 1475 and MELD Fold 1 Seed 1475 to separate host GPUs using `CUDA_VISIBLE_DEVICES`; each process then addresses its sole visible device as `cuda:0`. Before launch the scheduler must repeat `nvidia-smi` and exclude any GPU that has become occupied. It may use at most two GPUs for this two-job Pilot and at most four for the later 50-model phase.

Each task will use `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, and DataLoader `num_workers=0`. Output directories, logs, checkpoints and run states are isolated. A failed job may resume only under the preregistered compatible-checkpoint rule and cannot overwrite another task. Per-job GPU UUID, peak allocated/reserved memory, GPU-hours and wall time will be recorded.

No training job has been launched because the IEMOCAP label policy gate remains unresolved.
