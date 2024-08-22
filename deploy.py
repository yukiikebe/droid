"""
deploy.py

Provide a lightweight server/client implementation for deploying OpenVLA models (through the HF AutoClass API) over a
REST API. This script implements *just* the server, with specific dependencies and instructions below.

Note that for the *client*, usage just requires numpy/json-numpy, and requests; example usage below!

Dependencies:
    => Server (runs OpenVLA model on GPU): `pip install uvicorn fastapi json-numpy`
    => Client: `pip install requests json-numpy`

Client (Standalone) Usage (assuming a server running on 0.0.0.0:8000):

```
import requests
import json_numpy
json_numpy.patch()
import numpy as np

action = requests.post(
    "http://0.0.0.3:8003/act",
    json={"image": np.zeros((256, 256, 3), dtype=np.uint8), "instruction": "do something"}
).json()

Note that if your server is not accessible on the open web, you can use ngrok, or forward ports to your client via ssh:
    => `ssh -L 8000:localhost:8000 ssh USER@<SERVER_IP>`
"""

import os.path

# ruff: noqa: E402
import json_numpy

json_numpy.patch()
import json
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import draccus
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from droid.data_processing.timestep_processing import TimestepProcesser
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
from droid.evaluation.policy_wrapper import PolicyWrapperRobomimic
import argparse
import numpy as np
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.file_utils as FileUtils

# === Utilities ===
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def get_openvla_prompt(instruction: str, droid_path: Union[str, Path]) -> str:
    if "v01" in droid_path:
        return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
    else:
        return f"In: What action should the robot take to {instruction.lower()}?\nOut:"

# === Server Interface ===
class DroidServer:
    def __init__(self, droid_path: Union[str, Path], attn_implementation: Optional[str] = "flash_attention_2") -> Path:
        """
        A simple server for OpenVLA models; exposes `/act` to predict an action for a given image + instruction.
            => Takes in {"image": np.ndarray, "instruction": str, "unnorm_key": Optional[str]}
            => Returns  {"action": np.ndarray}
        """
        
        self.droid_path, self.attn_implementation = droid_path, attn_implementation
        variant = dict(
            exp_name="policy_test",
            save_data=False,
            use_gpu=True,
            seed=0,
            policy_logdir="test",
            task="",
            layout_id=None,
            model_id=50,
            camera_kwargs=dict(),
            data_processing_kwargs=dict(
                timestep_filtering_kwargs=dict(),
                image_transform_kwargs=dict(),
            ),
            ckpt_path=self.droid_path,
        )
        
        torch.manual_seed(variant["seed"])
        np.random.seed(variant["seed"])

        # Set Compute Mode #
        use_gpu = variant.get("use_gpu", False)
        torch.device("cuda:0" if use_gpu else "cpu")

        self.ckpt_path = variant["ckpt_path"]

        self.device = TorchUtils.get_torch_device(try_to_use_cuda=True)
        ckpt_dict = FileUtils.maybe_dict_from_checkpoint(ckpt_path=self.ckpt_path)
        self.config = json.loads(ckpt_dict["config"])    
        
        # self.imsize = max(ckpt_dict["shape_metadata"]["all_shapes"]["camera/image/hand_camera_left_image"])  

        # Handle missing shape_metadata
        for obs_key in ckpt_dict["shape_metadata"]["all_shapes"].keys():
            if 'camera/image' in obs_key:
                imsize = max(ckpt_dict["shape_metadata"]["all_shapes"][obs_key])
                break

        ckpt_dict["config"] = json.dumps(self.config)
        
        self.policy, _ = FileUtils.policy_from_checkpoint(ckpt_dict=ckpt_dict, device=self.device, verbose=True)
        self.policy.goal_mode = self.config["train"]["goal_mode"]
        self.policy.eval_mode = True
        
        # Determine the action space (relative or absolute)
        action_keys = self.config.get("train", {}).get("action_keys", [])
        if "action/rel_pos" in action_keys:
            action_space = "cartesian_velocity"
            for k in action_keys:
                assert not k.startswith("action/abs_")
        elif "action/abs_pos" in action_keys:
            action_space = "cartesian_position"
            for k in action_keys:
                assert not k.startswith("action/rel_")
        else:
            action_space = "default"  # Default value or raise an error
            raise ValueError("No valid action space found in config.")

        # Determine the action space for the gripper
        if "action/gripper_velocity" in action_keys:
            gripper_action_space = "velocity"
        elif "action/gripper_position" in action_keys:
            gripper_action_space = "position"
        else:
            gripper_action_space = "default"  # Default value or raise an error
            raise ValueError("No valid gripper action space found in config.")

        # Prepare Policy Wrapper #
        data_processing_kwargs = dict(
            timestep_filtering_kwargs=dict(
                action_space=action_space,
                gripper_action_space=gripper_action_space,
                robot_state_keys=["cartesian_position", "gripper_position", "joint_positions"],
            ),
            image_transform_kwargs=dict(
                remove_alpha=True,
                bgr_to_rgb=True,
                to_tensor=True,
                augment=False,
            ),
        )
        timestep_filtering_kwargs = data_processing_kwargs.get("timestep_filtering_kwargs", {})
        image_transform_kwargs = data_processing_kwargs.get("image_transform_kwargs", {})

        policy_data_processing_kwargs = {}
        self.policy_timestep_filtering_kwargs = policy_data_processing_kwargs.get("timestep_filtering_kwargs", {})
        self.policy_image_transform_kwargs = policy_data_processing_kwargs.get("image_transform_kwargs", {})

        self.policy_timestep_filtering_kwargs.update(timestep_filtering_kwargs)
        self.policy_image_transform_kwargs.update(image_transform_kwargs)

        self.fs = self.config.get("train", {}).get("frame_stack", 1)  # Default to 1 if not present

        print("droid_path: ", self.droid_path)
        # Load VLA Model using HF AutoClasses
        # self.processor = AutoProcessor.from_pretrained(self.droid_path, trust_remote_code=True)
        self.wrapped_policy = PolicyWrapperRobomimic(
            policy=self.policy,
            timestep_filtering_kwargs=self.policy_timestep_filtering_kwargs,
            image_transform_kwargs=self.policy_image_transform_kwargs,
            frame_stack=self.fs,
            eval_mode=True,
        )
            

    def predict_action(self, payload: Dict[str, Any]) -> str:
        try:
            if double_encode := "encoded" in payload:
                # Support cases where `json_numpy` is hard to install, and numpy arrays are "double-encoded" as strings
                assert len(payload.keys() == 1), "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            # Parse payload components
            images, instruction, cartesian_position, gripper_position = payload["images"], payload["instruction"], payload["cartesian_position"], payload["gripper_position"]
            unnorm_key = payload.get("unnorm_key", None)

            # Run VLA Inference
            # prompt = get_openvla_prompt(instruction, self.droid_path)
            # inputs = self.processor(prompt, Image.fromarray(image).convert("RGB")).to(self.device, dtype=torch.bfloat16)
            
            observation = {
                "image": {
                    0: images[0],
                    1: images[1]
                    },
                "robot_state" :{
                    "cartesian_position" : cartesian_position,
                    "gripper_position": gripper_position
                },
                "instruction": instruction,
                "camera_type": {0:1, 1:1},
            }
            action = self.wrapped_policy.forward(observation)
            
            if double_encode:
                return JSONResponse(json_numpy.dumps(action))
            else:
                return JSONResponse(action)
        except Exception as e:  # noqa: E722
            traceback_log = traceback.format_exc()
            print(traceback_log)
            logging.error(traceback_log)
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'image': np.ndarray, 'instruction': str}\n"
                "You can optionally an `unnorm_key: str` to specific the dataset statistics you want to use for "
                "de-normalizing the output actions."
            )
            return json.dumps({"error": str(e), "traceback": traceback_log})

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI()
        self.app.post("/action")(self.predict_action)
        uvicorn.run(self.app, host=host, port=port)


@dataclass
class DeployConfig:
    # fmt: off
    droid_path: Union[str, Path] = "model_epoch_3000.pth"

    # Server Configuration
    host: str = "0.0.0.0"                                               # Host IP Address
    port: int = 8000                                                    # Host Port

    # fmt: on

@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    server = DroidServer(cfg.droid_path)
    server.run(cfg.host, port=cfg.port)

if __name__ == "__main__":
    deploy()