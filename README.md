# AgiBot-World Challenge IROS2025 - Manipulation Baseline
We adopt [UniVLA](https://github.com/OpenDriveLab/UniVLA) as the baseline model for the [AgiBot World Challenge @ IROS 2025](https://agibot-world.com/challenge) - Manipulation track.

This repo provides a minimal version of training codes.

## :video_game: Setup <a name="installation"></a>

1. (Optional) We use conda to manage the environment.

```bash
conda create -n univla python=3.10 -y
conda activate univla
```

2. Install dependencies.

```bash
# Install pytorch
# Look up https://pytorch.org/get-started/previous-versions/ with your cuda version for a correct command
# Our experiments are conducted with 'torch 2.2.0 + cuda 12.1'
pip install torch torchvision

# Clone our repo and pip install to download dependencies
git clone https://github.com/OpenDriveLab/UniVLA.git
cd univla
pip install -e .

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
torchrun --standalone --nnodes 1 --nproc-per-node 8 finetune_genie.py \
                                 --dataset_name "genie" \
                                 --run_root_dir "genie_log" \
```

Once you finished training and get the action decoder and VLA backbone, you can simply start the evaluation with:

## Evaluation
```bash
# Start evaluation on Genie Sim Benchmark
# [Optional] Install Genie Sim dependencies
pip install -r experiments/robot/genie/genie_requirements.txt

# By default, we test for 50 rollouts every task, totalling 500 independent trials.
python experiments/robot/genie/run_genie_eval_decoder.py \
    --task_suite_name dustbin \
    --action_decoder_path /path/to/your/action_decoder_path.pt \
    --pretrained_checkpoint /path/to/your/finetuned_univla \
    --save_video False    # Whether to save rollout videos \
    --seed 7
```

> To be updated.

## :pushpin: TODO list
-  [ ] Minimal version of training code for AgibotWorld dataset and pretrained weights.
-  [ ] Minimal version of training code for the challenge's dataset. (available once the challenge dataset is ready).
-  [ ] Evaluation script.
-  [ ] Submission instructions.

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
