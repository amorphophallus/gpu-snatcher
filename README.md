# gpu-snatcher

Simple helper scripts for checking ZJU 4090 servers and starting a single-card training job automatically.

## Scripts

- `check_zju_4090.ps1` / `check_zju_4090.sh`
  Check whether each `zju_4090_*` server is reachable and which GPUs are free. A GPU is treated as available when memory usage is below 20%.

- `auto_train_single_card.ps1` / `auto_train_single_card.sh`
  Find one free GPU, replace `training.gpu_id=` in `TRAIN_COMMAND`, start the command in a remote `tmux` session, and return structured status.

- `cleanup_auto_train_sessions.ps1` / `cleanup_auto_train_sessions.sh`
  Remove the tmux sessions used by the auto-train scripts from all configured servers.

## Usage

PowerShell:

```powershell
.\check_zju_4090.ps1
.\auto_train_single_card.ps1
.\cleanup_auto_train_sessions.ps1
```

Bash:

```bash
./check_zju_4090.sh
./auto_train_single_card.sh
./cleanup_auto_train_sessions.sh
```

Before running `auto_train_single_card`, edit the `TRAIN_COMMAND` variable at the top of the script you want to use.

## Example

```powershell
$global:TRAIN_COMMAND = "python -m src.train.bc +experiment=rgbd/diff_unet training.gpu_id=0 wandb.project=test"
.\auto_train_single_card.ps1
```

Example output:

```text
status: started
server: zju_4090_230
gpu_id: 1
tmux_name: comet
command_name: src.train.bc
wandb_run_name: revived-totem-7
```
