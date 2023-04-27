## Clone
```bash
git clone https://github.com/antonioglass/runpod-cog.git
```

## Install Cog

```bash
sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`

sudo chmod +x /usr/local/bin/cog
```
## Download Model

```bash
sudo chmod 777 scripts/download-weights

sudo cog run script/download-weights

sudo cog predict -i prompt="monkey scuba diving" -i negative_prompt=""
```

## Download LoRa

```bash
wget https://civitai.com/api/download/models/12345 --content-disposition
```
## Convert LoRa

```bash
pip install torch==1.13.1 safetensors diffusers==0.14.0 transformers==4.27.1 accelerate==0.17.1
```

In `format_convert.py` update the `model_id`, `safetensor_path` variable with the path to the safetensor file you downloaded, and the `bin_path` variable with the desired output path for the bin file.

```bash
python format_convert.py
```
## Run Model
```bash
sudo cog predict -i prompt="" \
-i negative_prompt="" \
-i width= \
-i height= \
-i num_outputs= \
-i num_inference_steps= \
-i guidance_scale= \
-i scheduler= \
-i lora="" \
-i lora_scale= \
-i seed=
```

## Cog Model Edits

Once Cog is installed and the base Cog model is cloned, the following edits need to be made within the cloned directory.

1. Update cog.yaml, add the latest version of [runpod](https://pypi.org/project/runpod/) to the requirements within the cog.yaml file.
2. Add a .env file with the required environment variables.
3. Add the worker file
4. chmod +x worker

Finally, test the worker locally with `cog run ./worker`

## Building Container

Once the worker is tested locally, the container can be built.

```BASH
cog build -t ai-api-{model-name}
docker tag ai-api-{model-name} runpod/ai-api-{model-name}:latest
docker push runpod/ai-api-{model-name}:latest
```

*Replacing `ai-api-{model-name}` and `runpod` with your own model name and dockerhub username.*

## Docker Quick Reference

Before a worker container can be started, Docker Engine is required to be on the host machine.

```BASH
sudo apt-get update

sudo apt-get install \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

sudo mkdir -p /etc/apt/keyrings

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update

sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -

curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

sudo
```
