import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from PIL import Image
from dataclasses import dataclass
from typing import Union

import cv2
import numpy as np
import json
import draccus
from experiments.robot.geniesim.genie_model import WrappedGenieEvaluation, WrappedModel

import rclpy
import time, threading
from cv_bridge import CvBridge
import argparse

from genie_sim_ros import SimROSNode
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def resize_img(img, width, height):
    resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    resized_img = np.array(resized_img)
    return resized_img

def get_instruction(task_name):
    lang = "Pick up the yellow functional beverage can on the table with the left arm.;Threw the yellow functional beverage can into the trash can with the left arm.;Pick up the green carbonated beverage can on the table with the right arm.;Threw the green carbonated beverage can into the trash can with the right arm."
    if task_name == "iros_clear_the_countertop_waste":
        lang = "Pick up the yellow functional beverage can on the table with the left arm.;Threw the yellow functional beverage can into the trash can with the left arm.;Pick up the green carbonated beverage can on the table with the right arm.;Threw the green carbonated beverage can into the trash can with the right arm."
    elif task_name == "iros_restock_supermarket_items":
        lang = "Pick up the brown plum juice from the restock box with the right arm.;Place the brown plum juice on the shelf where the brown plum juice is located with the right arm."
    elif task_name == "iros_clear_table_in_the_restaurant":
        lang = "Pick up the bowl on the table near the right arm with the right arm.;Place the bowl on the plate on the table with the right arm."
    elif task_name == "iros_stamp_the_seal":
        lang = "Pick up the stamp from the ink pad on the table with the right arm.;Stamp the document on the table with the stamp in the right arm.;Place the stamp into the ink pad on the table with the right arm."
    elif task_name == "iros_pack_in_the_supermarket":
        lang = "Pick up the grape juice on the table with the right arm.;Put the grape juice into the felt bag on the table with the right arm."
    else:
        #To be implemented
        pass
    return lang

def infer(policy, cfg, args):

    rclpy.init()
    current_path = os.getcwd()
    sim_ros_node = SimROSNode()
    spin_thread = threading.Thread(
        target=rclpy.spin, args=(sim_ros_node,)
    )
    spin_thread.start()

    bridge = CvBridge()
    count = 0
    
    lang = get_instruction(args.task_name)

    while rclpy.ok():
        # 打开图像文件
        img_h_raw = sim_ros_node.get_img_head()
        img_l_raw = sim_ros_node.get_img_left_wrist()
        img_r_raw = sim_ros_node.get_img_right_wrist()
        act_raw = sim_ros_node.get_joint_state()

        if img_h_raw and img_l_raw and img_r_raw and act_raw and img_h_raw.header.stamp == img_l_raw.header.stamp == img_r_raw.header.stamp:
            # print("    img_h_raw", img_h_raw.header.stamp)
            # print("    img_l_raw", img_l_raw.header.stamp)
            # print("    img_r_raw", img_r_raw.header.stamp)
            # print("    LOOP", count)

            count = count + 1
            img_h = bridge.compressed_imgmsg_to_cv2(img_h_raw, desired_encoding="rgb8")
            img_l = bridge.compressed_imgmsg_to_cv2(img_l_raw, desired_encoding="rgb8")
            img_r = bridge.compressed_imgmsg_to_cv2(img_r_raw, desired_encoding="rgb8")

            # img_h_pil = Image.fromarray(img_h)
            # img_h_pil.save(f'frames/head_{count:05d}.png')
            # img_l_pil = Image.fromarray(img_l)
            # img_l_pil.save(f'frames/wrist_l_{count:05d}.png')
            # img_r_pil = Image.fromarray(img_r)
            # img_r_pil.save(f'frames/wrist_r_{count:05d}.png')

            state = np.array(act_raw.position)

            if cfg.with_proprio:
                action = policy.step(img_h, img_l, img_r, lang, state)
            else:
                action = policy.step(img_h, img_l, img_r, lang)

            sim_ros_node.publish_joint_command(action)
        else:
            print("skip")

        sim_ros_node.loop_rate.sleep()


@dataclass
class GenerateConfig:

    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = "checkpoints/finetuned"
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = True                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = False                        # Center crop? (if trained w/ random crop image aug)
    local_log_dir: str = "./experiments/eval_logs"   # Local directory for eval logs
    seed: int = 7  

    action_decoder_path:str = "rundir/action_decoder.pt"
    window_size: int = 30

    n_layers: int = 2
    hidden_dim: int = 1024
    balancing_factor: float = 0.1                     # larger for smoother

    with_proprio: bool = True
    debug: bool = False


@draccus.wrap()
def get_policy(cfg: GenerateConfig) -> None:

    wrapped_model = WrappedModel(cfg)
    wrapped_model.cuda()
    wrapped_model.eval()
    policy = WrappedGenieEvaluation(cfg, wrapped_model)

    return policy, cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", default="iros_stamp_the_seal", type=str, help="Selected task to run")
    # fmt: on
    args = parser.parse_args()
    policy, cfg = get_policy()
    infer(policy, cfg, args)