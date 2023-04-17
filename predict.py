''' StableDiffusion-v1 Predict Module '''

import os
from typing import List

import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    # StableDiffusionInpaintPipeline,
    StableDiffusionInpaintPipelineLegacy,

    DDIMScheduler,
    DDPMScheduler,
    # DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    IPNDMScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    # KarrasVeScheduler,
    PNDMScheduler,
    # RePaintScheduler,
    # ScoreSdeVeScheduler,
    # ScoreSdeVpScheduler,
    # UnCLIPScheduler,
    # VQDiffusionScheduler,
    LMSDiscreteScheduler
)

from PIL import Image
from cog import BasePredictor, Input, Path
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
from compel import Compel

MODEL_CACHE = "diffusers-cache"


class Predictor(BasePredictor):
    '''Predictor class for StableDiffusion-v1'''

    def setup(self):
        '''
        Load the model into memory to make running multiple predictions efficient
        '''
        print("Loading pipeline...")

        self.txt2img_pipe = StableDiffusionPipeline.from_pretrained(
            "antonioglass/dlbrt",
            safety_checker=None,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        ).to("cuda")
        self.img2img_pipe = StableDiffusionImg2ImgPipeline(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            tokenizer=self.txt2img_pipe.tokenizer,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
            safety_checker=None,
            # safety_checker=self.txt2img_pipe.safety_checker,
            feature_extractor=self.txt2img_pipe.feature_extractor,
        ).to("cuda")
        self.inpaint_pipe = StableDiffusionInpaintPipelineLegacy(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            tokenizer=self.txt2img_pipe.tokenizer,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
            safety_checker=None,
            # safety_checker=self.txt2img_pipe.safety_checker,
            feature_extractor=self.txt2img_pipe.feature_extractor,
        ).to("cuda")
        
        # because lora is loaded for the entire model
        self.lora_loaded = False

        self.txt2img_pipe.enable_xformers_memory_efficient_attention()
        self.compel = Compel(tokenizer=self.txt2img_pipe.tokenizer, text_encoder=self.txt2img_pipe.text_encoder)
        self.img2img_pipe.enable_xformers_memory_efficient_attention()
        self.inpaint_pipe.enable_xformers_memory_efficient_attention()

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def predict(
        self,
        prompt: str = Input(description="Input prompt", default=""),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default=None,
        ),
        width: int = Input(
            description="Output image width; max 1024x768 or 768x1024 due to memory limits",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=512,
        ),
        height: int = Input(
            description="Output image height; max 1024x768 or 768x1024 due to memory limits",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=512,
        ),
        init_image: Path = Input(
            description="Initial image to generate variations of, resized to the specified WxH.",
            default=None,
        ),
        mask: Path = Input(
            description="""Black and white image to use as mask for inpainting over init_image.
                        Black pixels are inpainted and white pixels are preserved.
                        Tends to work better with prompt strength of 0.5-0.7""",
            default=None,
        ),
        prompt_strength: float = Input(
            description="Prompt strength init image. 1.0 full destruction of init image",
            default=0.8,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=10,
            default=1
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        scheduler: str = Input(
            default="K-LMS",
            choices=["DDIM", "DDPM", "DPM-M", "DPM-S", "EULER-A", "EULER-D",
                     "HEUN", "IPNDM", "KDPM2-A", "KDPM2-D", "PNDM",  "K-LMS"],
            description="Choose a scheduler. If you use an init image, PNDM will be used",
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        lora: str = Input(
            description="instantly download lora models and use them via runpod",
            default=None
        ),
        lora_scale : int = Input(
            description="what percentage of the lora model do you want applied?",
            default=1
        )
    ) -> List[Path]:
        '''
        Run a single prediction on the model
        '''        
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")

        if width * height > 786432:
            raise ValueError(
                "Maximum size is 1024x768 or 768x1024 pixels, because of memory limits."
            )

        extra_kwargs = {}
        if mask:
            if not init_image:
                raise ValueError("mask was provided without init_image")

            pipe = self.inpaint_pipe
            init_image = Image.open(init_image).convert("RGB")
            extra_kwargs = {
                "mask_image": Image.open(mask).convert("RGB").resize(init_image.size),
                "image": init_image,
                "strength": prompt_strength,
            }
        elif init_image:
            pipe = self.img2img_pipe
            extra_kwargs = {
                "init_image": Image.open(init_image).convert("RGB"),
                "strength": prompt_strength,
            }
        else:
            pipe = self.txt2img_pipe
            extra_kwargs = {
                "width": width,
                "height": height,
            }

        pipe.scheduler = make_scheduler(scheduler, pipe.scheduler.config)

        prompt=[prompt] * num_outputs if prompt is not None else None
        prompt_embeds = self.compel(prompt)
        negative_prompt=[negative_prompt] * num_outputs if negative_prompt is not None else None
        negative_prompt_embeds = self.compel(negative_prompt)
        # not sure if it's needed. see more here: https://github.com/damian0815/compel#0110---add-support-for-prompts-longer-than-the-models-max-token-length
        [prompt_embeds, negative_prompt_embeds] = self.compel.pad_conditioning_tensors_to_same_length([prompt_embeds, negative_prompt_embeds])
        
        if not (lora is None):
            print("loaded lora")
            pipe.unet.load_attn_procs(lora)
            self.lora_loaded = True
            extra_kwargs['cross_attention_kwargs'] = {"scale": lora_scale}
        
        # because lora is retained between requests
        if (lora is None) and self.lora_loaded:
            extra_kwargs['cross_attention_kwargs'] = {"scale": 0}

        generator = torch.Generator("cuda").manual_seed(seed)
        output = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            # width=width,
            # height=height,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
            **extra_kwargs,
        )

        output_paths = []
        for i, sample in enumerate(output.images):
            if output.nsfw_content_detected and output.nsfw_content_detected[i] and self.NSFW:
                continue

            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        if len(output_paths) == 0:
            raise Exception(
                "NSFW content detected. Try running it again, or try a different prompt."
            )

        return output_paths


def make_scheduler(name, config):
    '''
    Returns a scheduler from a name and config.
    '''
    return {
        "DDIM": DDIMScheduler.from_config(config),
        "DDPM": DDPMScheduler.from_config(config),
        # "DEIS": DEISMultistepScheduler.from_config(config),
        "DPM-M": DPMSolverMultistepScheduler.from_config(config),
        "DPM-S": DPMSolverSinglestepScheduler.from_config(config),
        "EULER-A": EulerAncestralDiscreteScheduler.from_config(config),
        "EULER-D": EulerDiscreteScheduler.from_config(config),
        "HEUN": HeunDiscreteScheduler.from_config(config),
        "IPNDM": IPNDMScheduler.from_config(config),
        "KDPM2-A": KDPM2AncestralDiscreteScheduler.from_config(config),
        "KDPM2-D": KDPM2DiscreteScheduler.from_config(config),
        # "KARRAS-VE": KarrasVeScheduler.from_config(config),
        "PNDM": PNDMScheduler.from_config(config),
        # "RE-PAINT": RePaintScheduler.from_config(config),
        # "SCORE-VE": ScoreSdeVeScheduler.from_config(config),
        # "SCORE-VP": ScoreSdeVpScheduler.from_config(config),
        # "UN-CLIPS": UnCLIPScheduler.from_config(config),
        # "VQD": VQDiffusionScheduler.from_config(config),
        "K-LMS": LMSDiscreteScheduler.from_config(config)
    }[name]
