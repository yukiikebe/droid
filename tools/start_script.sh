#!/bin/bash
# start_script.sh

# Add your commands here
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY=$(cat /workspace/tools/wandb.api)
wandb login

cd /workspace
# Add more commands as needed

# Start an interactive bash shell
# exec /bin/bash "$@"
# exec python train.py --config-name=train_diffusion_unet_real_image_workspace task.dataset_path=data/demo_pusht_real
# exec python eval_real_robot.py -i data/outputs/2024.06.06/19.32.07_train_diffusion_unet_image_real_image/checkpoints/latest.ckpt -o data/eval_pusht_real --robot_ip 10.42.0.2

# python eval_real_robot.py -i modified_ckpt.ckpt -o data/eval_pusht_real --robot_ip 10.42.0.2

# Modified /home/yuki/research/diffusion_policy/diffusion_policy/common/cv2_util.py. Replaced cv2 to PIL resize

# pgrep -f 'pt_main_thread|python' | xargs -r kill -9