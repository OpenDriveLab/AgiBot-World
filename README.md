# Manipulation Baseline
We adopt [UniVLA](https://github.com/OpenDriveLab/UniVLA) as the baseline model for the [AgiBot World Challenge @ IROS 2025](https://agibot-world.com/challenge) - Manipulation track.

This repo provides a minimal version of training code.

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
pip install -r requirements.txt

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation
```

## :fire: Training 

- With the pretrained generalist policy trained to plan over an embodiment-agnostic action space, we then add embodiment-specific action decoder heads for downstream deployment.
- Our action decoder is extremely lightweight with only around 12M parameters. Using parameter-efficient fine-tuning with LoRA rank 32, the total trainable parameters are around 123M.

### :one: Download the dataset

### :two: Download the checkpoints
- Download the checkpoint of latent action model from <td><a href="https://huggingface.co/qwbu/univla-latent-action-model">univla-latent-action-model</a></td>.

- Download the weight of <td><a href="https://huggingface.co/TRI-ML/prismatic-vlms/tree/main/prism-dinosiglip-224px%2B7b">TRI-ML/prismatic-vlms/prism-dinosiglip-224px+7b</a></td>.

- Download the univla-7b checkpoint from <td><a href="https://huggingface.co/qwbu/univla-7b">univla-7b</a></td>.

### :three: Modify your configs

1) You should first set the pretrained UniVLA and latent action model path in ```vla_path``` and ```lam_path``` of the [training config](https://github.com/OpenDriveLab/UniVLA/blob/b502b3eddc05fef9984d34932a41c96e5a9f21a3/vla-scripts/finetune_libero.py#L107).
2) Set your local dataset path in [```data_root_dir```](https://github.com/OpenDriveLab/UniVLA/blob/b502b3eddc05fef9984d34932a41c96e5a9f21a3/vla-scripts/finetune_libero.py#L110).

### :four: Running

```bash
# Start training with 8 GPUs
torchrun \
--standalone \
--nnodes 1 \
--nproc-per-node 8 \
scripts/finetune_genie.py \
--vla_path checkpoints/finetuned \
--lam_path checkpoints/lam-stage-2.ckpt \
--data_root_dir genie_dataset/dustbin\
--codebook_size 16 \
--batch_size 8 \
--grad_accumulation_steps 1 \
--max_steps 5000 \
--save_steps 1000 \
--run_root_dir output/dustbin \
--adapter_tmp_dir output/dustbin \
```

Once you finished training and get the action decoder and VLA backbone, you can simply start the evaluation with:

## Evaluation
```bash
omni_python scripts/infer.py
```
> In the inference process, we use ROS2 to achieve data communication between the model and the <td><a href="https://github.com/AgibotTech/genie_sim">Genie Sim Benchmark</a></td> simulation environment. The interface is to be updated.

## :pushpin: TODO list
-  [x] Training code and dataloader for challenge dataset.
-  [x] Evaluation code.
-  [ ] Finetuned checkpoints on challenge dataset.
-  [ ] Updated simulation environment.

## :pencil: Citation
If you find our code or models useful in your work, please cite [our paper](https://arxiv.org/pdf/2505.06111):

```bibtex
@article{bu2025univla,
  title={UniVLA: Learning to Act Anywhere with Task-centric Latent Actions}, 
  author={Qingwen Bu and Yanting Yang and Jisong Cai and Shenyuan Gao and Guanghui Ren and Maoqing Yao and Ping Luo and Hongyang Li},
  journal={arXiv preprint arXiv:2505.06111},
  year={2025}
}
```
