FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime

# Allow pip to install packages in externally-managed environments (safe for containers)
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Set the workdir
WORKDIR /app

# Minimal software stack
RUN apt-get update && \
    apt-get install -y nano less htop git libhdf5-serial-dev build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install core dependencies
#RUN pip install --no-cache-dir numpy>=1.24.0 && \
#    pip install --no-cache-dir pandas>=1.17.0 && \
#    pip install --no-cache-dir torch>=2.0.0 && \
#    pip install --no-cache-dir torch_geometric>=2.7.0 && \
#    pip install --no-cache-dir tqdm>=4.65.0 && \
#    pip install --no-cache-dir lightning>=2.0.0 && \
#    pip install --no-cache-dir awkward>=2.0.0 && \
#    pip install --no-cache-dir uproot>=5.0.0 && \
#    pip install --no-cache-dir pyarrow>=23.0.0

# Copy all application code
COPY . .

# Install the package with all dependencies (hepattn, hgcal-offline-reco, dev tools, and flash-attention)
RUN echo "=======================================" && \
    echo "Installing hgcal-offline-reco with dependencies" && \
    echo "Current time:" $(date) && \
    echo "=======================================" && \
    pip install --no-cache-dir .[dev,fa2] --verbose && \
    echo "Installation complete!"

