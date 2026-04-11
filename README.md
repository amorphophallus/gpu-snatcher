# gpu-snatcher

Simple helper scripts for checking ZJU 4090 servers, starting single-card or multi-card training jobs automatically, preparing datasets, and pulling the latest evaluation checkpoint for local eval.

## Scripts

- `check_zju_4090.ps1` / `check_zju_4090.sh`
  Check whether each `zju_4090_*` server is reachable and which GPUs are free. A GPU is treated as available when memory usage is below 10%.

- `auto_train_single_card.ps1` / `auto_train_single_card.sh`
  Find one free GPU, replace `training.gpu_id=` in `TRAIN_COMMAND`, start the command in a remote `tmux` session, and return structured status.

- `auto_train_multi_card.ps1` / `auto_train_multi_card.sh`
  Find one server with enough free GPUs for a `torchrun` job, inject `CUDA_VISIBLE_DEVICES` and `--nproc_per_node=`, start the command in a remote `tmux` session, and return structured status.

- `cleanup_auto_train_sessions.ps1` / `cleanup_auto_train_sessions.sh`
  Remove the tmux sessions used by the auto-train scripts from all configured servers.

- `auto_eval.sh`
  Linux-only helper for locating the latest `outputs/{date}/{time}` run matching a `RUN_ID`, downloading the selected checkpoint into `LOCAL_PATH/checkpoints/bc/{TASK}/low/`, and launching `src.eval.evaluate_model` locally.

- `auto_data_preparation.sh`
  Linux-only helper for running rollout collection, batch processing pickles for multiple tasks, and uploading the whole `LOCAL_PATH/UPLOAD_RELATIVE_DIR` folder to `REMOTE_PATH/UPLOAD_RELATIVE_DIR` via `rsync` with progress display and resumable partial transfers.

## Usage

PowerShell:

```powershell
.\check_zju_4090.ps1
.\auto_train_single_card.ps1
.\auto_train_multi_card.ps1
.\cleanup_auto_train_sessions.ps1
```

Bash:

```bash
./check_zju_4090.sh
./auto_train_single_card.sh
./auto_train_multi_card.sh
./cleanup_auto_train_sessions.sh
./auto_eval.sh
./auto_data_preparation.sh
```

Before running `auto_train_single_card`, edit the globals at the top of the script such as `TRAIN_COMMAND`, `SSH_NAME` (optional), `GPU_ID` (optional single GPU id), and `DATA_DIR_PROCESSED` when needed.

Before running `auto_train_multi_card`, edit the globals at the top of the script such as `TRAIN_COMMAND`, `NUM_GPUs`, `SSH_NAME` (optional), `GPU_ID` (optional comma-separated preferred GPU list; only used when `SSH_NAME` is set), and `DATA_DIR_PROCESSED` when needed.

Before running `auto_eval.sh`, edit the globals at the top of the script such as `REMOTE_PATH`, `REMOTE_SSH_HOST` (optional, accepts `228` and expands it to `zju_4090_228`), `RUN_ID`, `LOCAL_PATH`, `TASK`, `PROJECT`, `EPOCH`, `N_ENVS`, `N_ROLLOUTS`, and `PARAMS`.

Before running `auto_data_preparation.sh`, edit the globals at the top of the script such as `STEPS`, `TASKS`, `TASK_CKPT`, `LOCAL_PATH`, `REMOTE_PATH`, `REMOTE_SSH_HOST`, `UPLOAD_RELATIVE_DIR`, `PROCESS_SUFFIX`, and `PROCESS_OUTPUT_SUFFIX`. You can comment out lines in `STEPS` to skip `collect_data`, `process_pickles`, or `upload`, and comment out lines in `TASKS` to limit which tasks run. `REMOTE_PATH` is the root path, `UPLOAD_RELATIVE_DIR` controls the upload subdirectory, and `REMOTE_SSH_HOST` is required for the upload step.

## Example

```powershell
$global:TRAIN_COMMAND = "python -m src.train.bc +experiment=rgbd/diff_unet training.gpu_id=0 wandb.project=test"
.\auto_train_single_card.ps1
```

Single-card example output:

```text
status: started
server: zju_4090_230
gpu_id: 1
tmux_name: comet
command_name: src.train.bc
wandb_run_name: revived-totem-7
```

```powershell
$global:TRAIN_COMMAND = "torchrun --standalone --nproc_per_node=2 -m src.train.bc_ddp +experiment=rgbd/diff_unet wandb.project=test"
$global:NUM_GPUs = 2
$global:SSH_NAME = "230"
$global:GPU_ID = "0,1,3"
.\auto_train_multi_card.ps1
```

Multi-card example output:

```text
status: started
server: zju_4090_230
num_gpus: 2
gpu_ids: 0,1
tmux_name: cedar
command_name: torchrun
wandb_run_name: bright-river-12
```
