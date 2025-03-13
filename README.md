<div id="top" align="center">

![agibot_world](https://github.com/user-attachments/assets/df64b543-db82-41ee-adda-799970e8a198)

<a href="https://opendrivelab.com/blog/agibot-world/" target="_blank">Research Blog on March 10</a> | <a href="https://arxiv.org/abs/2503.06669" target="_blank">Technical Report</a>

<a href="https://arxiv.org/abs/2503.06669"><img src="https://img.shields.io/badge/arXiv-Paper-<color>"></a> [![Static Badge](https://img.shields.io/badge/Download-grey?style=plastic&logo=huggingface&logoColor=yellow)](https://huggingface.co/agibot-world) [![Static Badge](https://img.shields.io/badge/Project%20Page-blue?style=plastic)](https://agibot-world.com) [![License](https://img.shields.io/badge/License-CC_%20_BY--NC--SA_4.0-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

</div>

## News📰 <a name="news"></a>

- **`[2025/03/10]`** 📄 <a href="https://opendrivelab.com/blog/agibot-world/" target="_blank">Research Blog</a> and <a href="https://arxiv.org/abs/2503.06669" target="_blank">Technical Report</a> released.
- **`[2025/02/26]`** Agibot World Beta released.
- **`[2025/01/03]`** <span style="color: #B91C1C; font-weight: bold;">Agibot World Alpha Sample Dataset released.</span>
- **`[2024/12/30]`** 🤖 Agibot World Alpha released.

## TODO List 📅 <a name="todolist"></a>

- [x] **AgiBot World Alpha**
- [x] **AgiBot World Beta** (expected Q1 2025)
  - [x] ~1,000,000 trajectories of high-quality robot data 
- [ ] **AgiBot World Foundation Model: GO-1** (expected Q2 2025)
  - [ ] Training & inference code
  - [ ] Pretrained model checkpoint
- [ ] **AgiBot World Colosseo** (expected 2025)
  - [ ] A comprehensive platform with toolkits including teleoperation, training and inference.
- [ ] **2025 AgiBot World Challenge** (expected 2025)

## Key Features 🔑 <a name="keyfeatures"></a>

- **1 million+** trajectories from 100 robots.
- **100+ 1:1 replicated real-life scenarios** across 5 target domains.
- **Cutting-edge hardware:** visual tactile sensors / 6-DoF Dexterous hand / mobile dual-arm robots
- **Wide-spectrum versatile challenging tasks**

<div style="max-width: 100%; overflow-x: auto; margin: 0 auto; !important;">
  <table style="border-collapse: collapse; border-spacing: 0; width: 100%; table-layout: fixed;">
    <tr style="border: none;">
      <td align="center" style="border: none; padding: 10px;">
        <img src="assets/Contact-rich_manipulation.gif" alt="Contact-rich Manipulation" width="230" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
        <p><b>Contact-rich Manipulation</b></p>
      </td>
      <td align="center" style="border: none; padding: 10px;">
        <img src="assets/Long-horizon_planning.gif" alt="Long-horizon Planning" width="230" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
        <p><b>Long-horizon Planning</b></p>
      </td>
      <td align="center" style="border: none; padding: 10px;">
        <img src="assets/Multi-robot_collaboration.gif" alt="Multi-robot Collaboration" width="230" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
        <p><b>Multi-robot Collaboration</b></p>
      </td>
    </tr>
  </table>
</div>



## Table of Contents

1. [Key Features](#keyfeatures)
2. [At a Quick Glance](#quickglance) 
3. [Getting Started](#installation)  
   - [Installation](#training)
   - [How to Get Started with Our AgiBot World Data](#preaparedata)
   - [Visualize Datasets](#visualizedatasets)
   - [Policy Learning Quickstart](#training)
4. [TODO List](#todolist)
5. [License and Citation](#liscenseandcitation)

## At a Quick Glance⬇️ <a name="quickglance"></a>

Follow the steps below to quickly explore and get an overview of AgiBot World with our [sample dataset](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha/blob/main/sample_dataset.tar) (~7GB).

```bash
# Installation
conda create -n agibotworld python=3.10 -y
conda activate agibotworld
pip install git+https://github.com/huggingface/lerobot@59e275743499c5811a9f651a8947e8f881c4058c
pip install matplotlib
git clone https://github.com/OpenDriveLab/AgiBot-World.git
cd AgiBot-World

# Download the sample dataset (~7GB) from Hugging Face. Replace <your_access_token> with your Hugging Face Access Token. You can generate an access token by following the instructions in the Hugging Face documentation from https://huggingface.co/docs/hub/security-tokens
mkdir data
cd data
curl -L -o sample_dataset.tar -H "Authorization: Bearer <your_access_token>" https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha/resolve/main/sample_dataset.tar
tar -xvf sample_dataset.tar

# Convert the sample dataset to LeRobot dataset format and visualize
cd ..
python scripts/convert_to_lerobot.py --src_path ./data/sample_dataset --task_id 390 --tgt_path ./data/sample_lerobot
python scripts/visualize_dataset.py --task-id 390 --dataset-path ./data/sample_lerobot
```

## Getting started 🔥 <a name="gettingstarted"></a>

#### Installation <a name="installation"></a>

Download our source code:
```bash
git clone https://github.com/OpenDriveLab/AgiBot-World.git
cd AgiBot-World
```

Our project is built upon the [lerobot library](https://github.com/huggingface/lerobot) (**dataset `v2.0`, commit 59e2757**), 
install lerobot through
```bash
pip install git+https://github.com/huggingface/lerobot@59e275743499c5811a9f651a8947e8f881c4058c
```

#### How to Get Started with Our AgiBot World Data <a name="preaparedata"></a>

- [OPTION 1] Download data from our [OpenDataLab](https://opendatalab.com/OpenDriveLab/AgiBot-World) page.

```bash
pip install openxlab # install CLI
openxlab dataset get --dataset-repo OpenDriveLab/AgiBot-World # dataset download
```

- [OPTION 2] Download data from our [Hugging Face](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha) page.

```bash
huggingface-cli download --resume-download --repo-type dataset agibot-world/AgiBotWorld-Alpha --local-dir ./AgiBotWorld-Alpha
```

Convert the data to **LeRobot Dataset** format.

```bash
python scripts/convert_to_lerobot.py --src_path /path/to/agibotworld/alpha --task_id 390 --tgt_path /path/to/save/lerobot
```

#### Visualize Datasets <a name="visualizedatasets"></a>

We adapt and extend the dataset visualization script from [LeRobot Project](https://github.com/huggingface/lerobot/blob/main/lerobot/scripts/visualize_dataset.py)

```bash
python scripts/visualize_dataset.py --task-id 390 --dataset-path /path/to/lerobot/format/dataset
```

It will open `rerun.io` and display the camera streams, robot states and actions, like this:
<div style="text-align: center;">
<img src="assets/dataset_visualization.gif" width="600">
</div>

#### Policy Training Quickstart <a name="training"></a>

Leveraging the simplicity of [LeRobot Dataset](https://github.com/huggingface/lerobot), we provide a user-friendly [Jupyter Notebook](https://github.com/OpenDriveLab/AgiBot-World/blob/main/AgibotWorld.ipynb) for training diffusion policy on AgiBot World Dataset.

<!-- <p align="right">(<a href="#top">back to top</a>)</p> -->



<!-- <p align="right">(<a href="#top">back to top</a>)</p> -->


## License and Citation📄   <a name="liscenseandcitation"></a>

All the data and code within this repo are under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). 

- Please consider citing our work either way below if it helps your research.
- For the full authorship and detailed contributions, please refer to [contributions](CONTRIBUTING.md).


```BibTeX
@misc{contributors2024agibotworldrepo,
  title={AgiBot World Colosseo},
  author={AgiBot World Colosseo contributors},
  howpublished={\url{https://github.com/OpenDriveLab/AgiBot-World}},
  year={2024}
}
```

- In alphabetical order by surname:
```BibTeX
@article{contributors2025agibotworld,
  title={{AgiBot World Colosseo: A Large-scale Manipulation Platform for Scalable and Intelligent Embodied Systems}},
  author={AgiBot-World-Contributors and Bu, Qingwen and Cai, Jisong and Chen, Li and Cui, Xiuqi and Ding, Yan and Feng, Siyuan and Gao, Shenyuan and He, Xindong and Huang, Xu and Jiang, Shu and Jiang, Yuxin and Jing, Cheng and Li, Hongyang and Li, Jialu and Liu, Chiming and Liu, Yi and Lu, Yuxiang and Luo, Jianlan and Luo, Ping and Mu, Yao and Niu, Yuehan and Pan, Yixuan and Pang, Jiangmiao and Qiao, Yu and Ren, Guanghui and Ruan, Cheng and Shan, Jiaqi and Shen, Yongjian and Shi, Chengshi and Shi, Mingkang and Shi, Modi and Sima, Chonghao and Song, Jianheng and Wang, Huijie and Wang, Wenhao and Wei, Dafeng and Xie, Chengen and Xu, Guo and Yan, Junchi and Yang, Cunbiao and Yang, Lei and Yang, Shukai and Yao, Maoqing and Zeng, Jia and Zhang, Chi and Zhang, Qinglin and Zhao, Bin and Zhao, Chengyue and Zhao, Jiaqi and Zhu, Jianchao},
  journal={arXiv preprint arXiv:2503.06669},
  year={2025}
}
```
