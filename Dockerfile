FROM stereolabs/zed:4.2-gl-devel-cuda12.1-ubuntu22.04

# Prevent interactive prompts during apt installations
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory in the container
WORKDIR /usr/src/app

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

# (Optional) Install the ZED Python API
# The stereolabs base image usually requires you to run their python API script
RUN python3 /usr/local/zed/get_python_api.py

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of your offline pipeline code into the container
COPY . .

# Set default command
CMD ["python3", "processor.py"]