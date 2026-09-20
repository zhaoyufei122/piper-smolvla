# SmolVLA on Piper: collection, training and inference

This is the workflow used for the cube experiment. The implemented collection method is SpaceMouse teleoperation of the real arm, with the wrist camera attached to the gripper. Earlier alternatives are preserved in [the planning notes](docs/SMOLVLA_PLANNING.md).

## Environment

Use Linux for CAN, camera access and robot control. ROS is not required for these commands. The reference local inference environment, inspected with `PYTHONNOUSERSITE=1`, contains:

| Package | Version |
| --- | --- |
| Python | 3.12.14 |
| LeRobot | 0.6.1 |
| PyTorch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| Transformers | 5.5.4 |
| NumPy | 2.2.6 |

The robot tools also need `piper_sdk`, `python-can`, `numpy`, `opencv-python` and `pyrealsense2`. SpaceMouse collection uses `hidapi` or the Linux `evdev` fallback. Recording uses the `ffmpeg` executable; CAN bring-up uses `iproute2`, with `can-utils` / `ethtool` useful for diagnostics. This is a reference snapshot, not a fully locked installation manifest.

For a new machine, use the [LeRobot installation guide](https://huggingface.co/docs/lerobot/installation) and [PyTorch installation selector](https://pytorch.org/get-started/locally/) to select a build for its driver and GPU. Dependencies and model assets require downloads on first installation. Check the package resolver before letting it replace an existing working Torch installation.

For the existing environment:

```bash
conda activate lerobot
export PYTHONNOUSERSITE=1
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

`PYTHONNOUSERSITE=1` matters on the original laptop: a different CPU-only Torch exists in the user site directory. No reinstall is needed to avoid that shadowing.

## 1. Hardware preparation

Run from the repository root. Set up the adapter using the helper:

```bash
bash scripts/can_activate.sh can0 1000000
cd sdk_tests
python arm_tool.py status
```

The arm, Pika gripper and wrist camera must match the geometry and units in the scripts. Specify your camera serial instead of relying on the original bench's default. Only one process should command the arm; stop the ROS bridge and other controllers before using the SDK tools.

On the original bench, the observation pose was `-90 60 -45 0 30 0` degrees. Moving there with `joint_position_ctrl.py --target ...` is a real movement, so check the path and workspace first. See [hardware bring-up](docs/BRINGUP.md).

## 2. Collect demonstrations

From `sdk_tests/`, using an environment with the robot and SpaceMouse dependencies:

```bash
python teleop_spacemouse.py \
  --record "$HOME/piper_data/cubes" \
  --task "put the red cube in the black box" \
  --task "put the white cube in the black box" \
  --randomize \
  --camera-serial YOUR_D405_SERIAL
```

`R` starts/stops an episode, `X` discards it, and `T` changes the task between episodes. Space pauses/resumes teleoperation. SpaceMouse buttons open/close the gripper. Follow the displayed layout prompt and confirm that the task label matches the object you actually grasp. A randomized prompt records an intended layout; it does not verify the physical table arrangement.

Each episode contains `meta.json`, `data.jsonl` and `wrist.mp4`. State and action have seven values: six joints in radians, followed by gripper width in metres. RGB is recorded at 30 FPS. Keep the raw recordings unchanged when experimenting with training subsets or pause trimming.

## 3. Select and convert episodes

The initial collection has 56 directories, `episode_000` through `episode_055`. Episode `000` is practice and must be excluded. **The converter does not exclude it automatically.** Create a separate view of the retained episodes; this leaves the raw data untouched:

```bash
python - <<'SELECT_EPISODES'
from pathlib import Path
source = Path.home() / "piper_data/cubes"
selected = Path.home() / "piper_data/cubes_selected_v1"
selected.mkdir(exist_ok=False)
for episode in sorted(source.glob("episode_*")):
    if episode.name == "episode_000":
        continue
    if all((episode / name).exists() for name in ("meta.json", "data.jsonl", "wrist.mp4")):
        (selected / episode.name).symlink_to(episode.resolve(), target_is_directory=True)
print(selected)
SELECT_EPISODES
```

The selection above is for the current complete local collection. Review failures or unfinished recordings explicitly when adding future episodes. If you transfer this symlink view to a server, transfer the link targets too (for example with `rsync -L`); links into a laptop home directory will not work remotely.

With LeRobot available, still from `sdk_tests/`:

```bash
python to_lerobot.py "$HOME/piper_data/cubes_selected_v1" \
  --repo-id local/piper_cubes \
  --root "$HOME/piper_data/lerobot_cubes_v1" \
  --dry-run

python to_lerobot.py "$HOME/piper_data/cubes_selected_v1" \
  --repo-id local/piper_cubes \
  --root "$HOME/piper_data/lerobot_cubes_v1"
```

Use a new output directory for each export. The converter takes task labels from each episode's `meta.json`; it trims leading/trailing stillness and caps interior pauses at `--max-pause 0.5` seconds by default. This changes temporal spacing, so record that setting and inspect the exported sequences. The exact earlier server export is not bundled here.

## 4. Fine-tune on a remote GPU

Transfer the converted dataset to the server first. Run in a prepared LeRobot environment; set `PIPER_DATASET_ROOT` and `PIPER_TRAIN_ROOT` to directories on the server's data disk:

```bash
export PIPER_DATASET_ROOT=/path/on/data-disk/piper/lerobot
export PIPER_TRAIN_ROOT=/path/on/data-disk/piper/train

lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/piper_cubes \
  --dataset.root="$PIPER_DATASET_ROOT" \
  --output_dir="$PIPER_TRAIN_ROOT" \
  --rename_map='{"observation.images.wrist":"observation.images.camera1"}' \
  --batch_size=64 \
  --steps=20000 \
  --num_workers=16 \
  --policy.device=cuda \
  --wandb.enable=false \
  --save_freq=2000
```

These settings reproduce the reported run configuration: one RTX 4090, 20,000 steps, roughly 1 h 54 min. Hardware throughput can differ. The base model and tokenizer need to be downloaded or cached. The reported final training loss was approximately 0.013; this is not a measure of correct-colour grasp success. The saved configuration had no held-out evaluation split.

## 5. Bring the policy back

Copy the whole `train/checkpoints/020000/pretrained_model/` directory, not only `model.safetensors`. Keep `config.json`, both processor JSON files, the processor statistics and `train_config.json` together.

Example, on the laptop; replace the host, port and remote path:

```bash
mkdir -p "$HOME/piper_data/policies/piper_cubes_020000"
scp -P PORT -r USER@HOST:/path/on/data-disk/piper/train/checkpoints/020000/pretrained_model "$HOME/piper_data/policies/piper_cubes_020000/"
```

The original laptop's checkpoint is at `~/piper_data/policies/piper_cubes_020000/pretrained_model`. Weights and recordings are not distributed with this Git repository. Resuming training additionally requires the relevant training-state checkpoint; the inference directory alone is not a full training backup.

## 6. Run and record a trial

Read the [known control issues](docs/PROJECT_NOTES.md#control-code-status) before using the current experimental controller. From the repository root:

```bash
conda activate lerobot
export PYTHONNOUSERSITE=1
cd sdk_tests
python run_policy.py "$HOME/piper_data/policies/piper_cubes_020000/pretrained_model" \
  --task "put the red cube in the black box" \
  --camera-serial YOUR_D405_SERIAL \
  --trace "$HOME/piper_data/runs" \
  --dry-run
```

Dry-run uses real CAN feedback and the real camera, but suppresses movement commands. Remove the final flag only for a supervised hardware trial. Press space to start; space requests hold and `q` exits. Automatic return-home needs further stop-path validation and must not be assumed to obey every pause/fault path. The script enables the arm during startup in real mode, before space starts the policy.

For white, change only the task string to `put the white cube in the black box`; white selection is currently unreliable. The prompt is fixed for a process, so restart the script when changing it. When all Hugging Face assets are already cached, `HF_HOME="$HOME/hf_cache" HF_HUB_OFFLINE=1` can be added to the command environment.

Current defaults include `--smooth 0.25`, binary gripper targets of 20/70 mm, `--ensemble-every 0` (ensembling off), and a 120-second attempt timeout. Requested control/policy rates are 100/30 Hz; actual inference timing must be measured. The grip-width check detects a possible held object, not its colour or successful placement.

Trace paths without `.jsonl` are treated as directories and get numbered filenames. Inspect an actual trace afterwards:

```bash
python analyse_shake.py "$HOME/piper_data/runs/run_000_white.jsonl"
```

Use your actual filename. Stop the policy before starting another controller. Park using the bench's verified procedure before disabling: disabling an unsupported arm can let it fall.
