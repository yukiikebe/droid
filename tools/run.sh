#!/bin/bash
xhost local:root
# Set image name
IMAGE_NAME="yuki/droid:latest"

# Set Dockerfile path (you can modify this if your Dockerfile is in a different directory)
DOCKERFILE_PATH="docker/Dockerfile"

# Run the Docker container interactively
echo "Running Docker container from image ${IMAGE_NAME}..."
docker run -it --gpus all --rm \
    -v $PWD:/workspace \
    --shm-size=128g \
    -v /tmp/.X11-unix:/tmp/.X11-unix  -e DISPLAY \
    --device=/dev/video0 --device=/dev/video1 \
    --device=/dev/input/js0 \
    --privileged \
    --net host ${IMAGE_NAME} "/bin/bash"

    
    # --name Cont1

    # -c /workspace/tools/start_script.sh"

    # -v /home/yuki/UR10/diffusion_policy/data:/workspace/data \

    # docker run -it --name Cont2 -v "$PWD":/workspace 'yuki/droid:latest' bash