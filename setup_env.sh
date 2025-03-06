#!/bin/bash

#To run this file on ssp cloud:
#chmod u+x setup_env.sh && ./setup_env.sh

echo "Setting up SSP Cloud environment"

set -e

git clone https://github.com/marssoo/SAVs.git

cd SAVs/

git checkout ssp-cloud

#conda create -n savs python=3.10 -y

#conda activate savs
echo "Installing packages ..."

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

pip install -e .

pip install -U bitsandbytes

cd ../

echo "Downloading natural bench ..."

wget -O naturalbench.zip "https://huggingface.co/datasets/BaiqiL/naturalbench_pictures/resolve/main/raw_images.zip"

mkdir naturalbench

unzip naturalbench.zip -d naturalbench

rm naturalbench.zip

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd SAVs/

# Inform the user
echo "Setup completed successfully!"
