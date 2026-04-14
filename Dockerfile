FROM stereolabs/zed:4.2-devel-cuda12.1-ubuntu22.04

# Prevent interactive prompts during apt installations
ENV DEBIAN_FRONTEND=noninteractive

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

# Install the ZED Python API
RUN python3 /usr/local/zed/get_python_api.py

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Wipe out the ZED API's numpy version, then force a clean install of Python dependencies
RUN pip3 uninstall -y numpy && \
    pip3 install --no-cache-dir --force-reinstall -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set default command to run our extraction script
CMD ["/bin/bash"]
