# UniVLA Baseline

## :video_game: Setup <a name="installation"></a>

1. (Optional) We use conda to manage the environment.

```bash
conda create -n univla python=3.10 -y
conda activate univla
```

2. Install dependencies.

```bash
# Clone our repo and pip install to download dependencies
git clone -b manipulation-challenge https://github.com/OpenDriveLab/AgiBot-World.git
cd UniVLA
pip install -r requirements.txt

git clone https://huggingface.co/spaces/LiheYoung/Depth-Anything
cd Depth-Anything/torchhub/facebookresearch_dinov2_main
pip install -e .

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation
```

## :fire: Running

### :one: Checkpoints Downloading
- Download the weight of latent action model from <td><a href="https://huggingface.co/qwbu/univla-latent-action-model">univla-latent-action-model</a></td>.

- Download the weight of visual backbone from <td><a href="https://huggingface.co/TRI-ML/prismatic-vlms/tree/main/prism-dinosiglip-224px%2B7b">TRI-ML/prismatic-vlms/prism-dinosiglip-224px+7b</a></td>.

- Download the weight of pretrained univla-7b from <td><a href="https://huggingface.co/qwbu/univla-7b">univla-7b</a></td>.

### :two: Training

```bash
# Start training with 8 GPUs
torchrun \
--standalone \
--nnodes 1 \
--nproc-per-node 8 \
scripts/finetune.py \
--vla_path univla-7b \
--lam_path univla-latent-action-model \
--data_root_dir dataset \
--meta_json_dir dataset \
--codebook_size 16 \
--batch_size 4 \
--grad_accumulation_steps 1 \
--max_steps 10000 \
--save_steps 1000 \
--decoder_n_layers 2 \
--decoder_hidden_dim 1024 \
--run_root_dir checkpoints/rundir \
--adapter_tmp_dir checkpoints/adapterdir \
--save_latest_checkpoint_only \
--with_proprio \
--use_lora \
--task_ids 0 \ # for task 1
# --task_ids 0 1 2 3 4 5 6 7 8 9 \ # for all 10 tasks
```

Once you finished training and get the action decoder and VLA backbone, you can simply start the evaluation with:

## :chart_with_upwards_trend: Evaluation
```bash
omni_python scripts/infer.py --task_name test_task_name
```
> In the inference process, we use ROS2 to achieve data communication between the model and the <td><a href="https://github.com/AgibotTech/genie_sim">Genie Sim Benchmark</a></td> simulation environment.
