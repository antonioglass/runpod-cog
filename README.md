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

sudo cog run scripts/download-weights

sudo cog predict -i prompt="monkey scuba diving" -i negative_prompt=""
```

## Convert Safetensors to Diffusers
```bash
pip install torch==1.13.1 safetensors diffusers==0.14.0 transformers==4.27.1 accelerate==0.17.1 omegaconf
```

```bash
python convert_original_stable_diffusion_to_diffusers.py \
--checkpoint_path ./path-to-model.safetensors \
--dump_path out-put-path \
--from_safetensors
```

## Convert VAE pt to Diffusers
```bash
pip install numexpr==2.7.3 pytorch_lightning
```

```bash
python convert_vae_pt_to_diffusers.py \
--vae_pt_path path-to-vae \
--dump_path vae
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
-i pose_image="path-to-image" \
-i lora="" \
-i lora_scale= \
-i seed=
```

## Push model to Huggingface
1. Create repository on Huggingface
2. Clone it
```bash
git clone https://huggingface.co/<your-username>/<your-model-name>
cd <your-model-name>
```
3. Move model into the directory
4. Download git-lfs
```bash
sudo apt update
sudo apt install git-lfs
```
5. Initialize git-lfs `git lfs install`
6. `huggingface-cli lfs-enable-largefiles .`
7. Push
```bash
# Create any files you like! Then...
git add .
git commit -m "First model version"  # You can choose any descriptive message
git push
```

## Cog Model Edits

Once Cog is installed and the base Cog model is cloned, the following edits need to be made within the cloned directory.

1. Update cog.yaml, add the latest version of [runpod](https://pypi.org/project/runpod/) to the requirements within the cog.yaml file.
2. Add a .env file with the required environment variables.
3. Add the worker file
4. chmod +x worker

Finally, test the worker locally with `cog run ./worker`

## Building Container and Pushing it to Docker Hub

Once the worker is tested locally, the container can be built.

```BASH
sudo cog build -t ai-api-{model-name}
sudo docker tag ai-api-{model-name} runpod/ai-api-{model-name}:latest
```

```BASH
sudo docker login
sudo docker push runpod/ai-api-{model-name}:latest
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

## Other
### Clip Skip 2
Set `num_hidden_layers` to 11 in `text_encoder/config.json`.
