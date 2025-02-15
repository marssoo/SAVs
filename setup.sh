#!/bin/bash

echo "Setting up SSP Cloud environment"

set -e

git clone https://github.com/marssoo/SAVs.git

cd SAVs/

git checkout ssp-cloud

conda create -n savs python=3.10 -y

source activate savs

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

pip install -e .

cd ../

# Inform the user
echo "Setup completed successfully!"
