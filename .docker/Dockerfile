# Use NVIDIA's PyTorch image which already includes CUDA and cuDNN
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=US/Pacific

# Install system dependencies
RUN apt-get update && apt-get install -y --fix-missing \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglfw3 \
    patchelf \
    libspnav-dev \
    spacenavd \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    git \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    build-essential \
    libffi-dev \
    libssl-dev \
    pkg-config \
    libxcb-shm0-dev \
    libxcb-xfixes0-dev \
    libjpeg-dev \
    zlib1g-dev \
    libegl1-mesa-dev \
    libusb-1.0-0 \
    libusb-1.0-0-dev \
    cmake \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Update alternatives to use Python 3.8
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

# Install pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py && \
    pip install --upgrade 'pip<24.1' 'setuptools<58' 'wheel<0.37' && \
    rm get-pip.py

# Install numba version compatible with robosuite
RUN pip install numba==0.56.4

# Attempt to install egl_probe separately and ignore failure
RUN pip install egl_probe 

# Clone and install robosuite
RUN git clone https://github.com/cheng-chi/robosuite.git /tmp/robosuite && \
    cd /tmp/robosuite && \
    git checkout 277ab9588ad7a4f4b55cf75508b44aa67ec171f0 && \
    pip install . --no-deps

# Clone and install r3m
RUN git clone https://github.com/facebookresearch/r3m.git /tmp/r3m && \
    cd /tmp/r3m && \
    git checkout b2334e726887fa0206962d7984c69c5fb09cceab && \
    pip install .

# Install gym separately and handle errors
RUN pip install gym==0.21.0 --no-deps

# Install all Python packages except gym in one go
RUN pip install numpy==1.23.3 \
    scipy==1.9.1 \
    opencv-python==4.6.0.66 \
    cffi==1.15.1 \
    ipykernel==6.16.1 \
    matplotlib==3.6.1 \
    zarr==2.12.0 \
    numcodecs==0.10.2 \
    h5py==3.7.0 \
    hydra-core==1.2.0 \
    einops==0.4.1 \
    tqdm==4.64.1 \
    dill==0.3.5.1 \
    scikit-video==1.1.11 \
    scikit-image==0.19.3 \
    pymunk==6.2.1 \
    wandb==0.13.3 \
    threadpoolctl==3.1.0 \
    shapely==1.8.4 \
    cython==0.29.32 \
    imageio==2.22.0 \
    imageio-ffmpeg==0.4.7 \
    termcolor==2.0.1 \
    tensorboard==2.10.1 \
    tensorboardx==2.5.1 \
    psutil==5.9.2 \
    click==8.0.4 \
    boto3==1.24.96 \
    accelerate==0.13.2 \
    datasets==2.6.1 \
    diffusers==0.11.1 \
    av==10.0.0 \
    cmake==3.24.3 \
    imagecodecs==2022.9.26 \
    ray[default,tune]==2.2.0 \
    free-mujoco-py==2.1.6 \
    pygame==2.1.2 \
    robomimic==0.2.0 \
    pytorchvideo==0.1.5 \
    pyrealsense2 \
    ur-rtde==1.5.5 \
    atomics==1.0.2 \
    dm-control==1.0.9 \
    pynput==1.7.6 \
    uvicorn==0.30.6 \
    fastapi==0.112.0 \
    json_numpy==2.1.0 \
    draccus==0.8.0 \
    transformers==4.44.0 \
    opencv-contrib-python==4.10.0.84 \
    timm==1.0.8
     

# Set the working directory in the container
WORKDIR /workspace

# Copy the rest of the application code into the container
COPY . .

# Expose port for Jupyter notebook or other services
EXPOSE 8888

# Verify bash and spacenavd installation
RUN which bash || echo "bash not found"
RUN which spacenavd || echo "spacenavd not found"

# Set the entrypoint to activate the environment
ENTRYPOINT [ "/bin/bash", "-l", "-c" ]

# Command to start spacenavd service
CMD ["spacenavd", "--nodaemon"]
