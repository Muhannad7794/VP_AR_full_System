# ZED SDK 5.2 development image (CUDA 12.8, Ubuntu 22.04, Python 3.10).
# The SDK version matches the Jetson that records the SVO files: an SVO written
# by a newer SDK may not open with an older one.
# The host NVIDIA driver must support CUDA 12.8 or newer: the "CUDA Version"
# shown by `nvidia-smi` must be 12.8 or higher.
ARG ZED_IMAGE_TAG=5.2.3-devel-cuda12.8-ubuntu22.04
FROM stereolabs/zed:${ZED_IMAGE_TAG}

# Prevent interactive prompts during apt installations
ENV DEBIAN_FRONTEND=noninteractive
# Print progress output immediately instead of buffering it
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /VP_AR_full_System_dockerized

# Install system dependencies for OpenCV and Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install the ZED Python API (pyzed) matching the SDK version of the base image.
# The installer downloads the wheel from stereolabs.com, so the build needs internet access.
RUN python3 /usr/local/zed/get_python_api.py

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# pyzed 5.x is built against NumPy 2.x, so the pinned requirements are NumPy 2
# compatible. Any OpenCV package installed by the ZED installer is removed, so
# that only opencv-contrib-python from requirements.txt provides cv2.
RUN pip3 uninstall -y numpy opencv-python opencv-python-headless opencv-contrib-python || true
RUN pip3 install --no-cache-dir --force-reinstall -r requirements.txt

# Check at build time that NumPy, OpenCV and PyTorch import together and that
# the installed packages satisfy each other's declared requirements (pyzed's
# NumPy requirement included). pyzed itself cannot be imported here: it loads
# the GPU driver library (libcuda.so.1), which the NVIDIA container runtime
# only provides when a container runs, not during the build.
RUN python3 -c "import numpy, cv2, torch; \
print('numpy', numpy.__version__, '| opencv', cv2.__version__, '| torch', torch.__version__)" \
    && pip3 show pyzed | head -n 2 \
    && pip3 check

# Copy the rest of the application code
COPY . .

# Set default command
CMD ["/bin/bash"]