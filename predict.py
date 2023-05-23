''' StableDiffusion-v1 Predict Module '''

import os
from typing import List

import torch
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
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
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import CrossFrameAttnProcessor

from PIL import Image
from cog import BasePredictor, Input, Path
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
from compel import Compel
from diffusers.utils import load_image
import imageio
import numpy as np

MODEL_CACHE = "diffusers-cache"


class Predictor(BasePredictor):
    '''Predictor class for StableDiffusion-v1'''

    def setup(self):
        '''
        Load the model into memory to make running multiple predictions efficient
        '''
        print("Loading pipeline...")

        self.controlnet_pose = ControlNetModel.from_pretrained(
             "./control_v11p_sd15_openpose",
             torch_dtype=torch.float16
        )
        self.txt2img_pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "./dlbrt",
            safety_checker=None,
            controlnet=self.controlnet_pose,
            torch_dtype=torch.float16
        ).to("cuda")
        
        self.txt2img_pipe.enable_xformers_memory_efficient_attention()
        self.compel = Compel(tokenizer=self.txt2img_pipe.tokenizer, text_encoder=self.txt2img_pipe.text_encoder)

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
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=22
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7
        ),
        scheduler: str = Input(
            default="EULER-A",
            choices=["DDIM", "DDPM", "DPM-M", "DPM-S", "EULER-A", "EULER-D",
                     "HEUN", "IPNDM", "KDPM2-A", "KDPM2-D", "PNDM",  "K-LMS"],
            description="Choose a scheduler. If you use an init image, PNDM will be used",
        ),
        video_path: str = Input(
            description="Path to processed mp4 video for ControlNet.",
            default="./dance1_corr.mp4",
        ),
        frame_count: int = Input(
            description="Frame count", default=8
        ),
        duration: int = Input(
            description="GIF duration / fps. duration = 1000 / fps", default=125
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        '''
        Run a single prediction on the model
        '''        
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if width * height > 786432:
            raise ValueError(
                "Maximum size is 1024x768 or 768x1024 pixels, because of memory limits."
            )

        pipe = self.txt2img_pipe
        pipe.scheduler = make_scheduler(scheduler, pipe.scheduler.config)
        
        # Video
        reader = imageio.get_reader(video_path, "ffmpeg")
        pose_images = [Image.fromarray(reader.get_data(i)) for i in range(frame_count)]

        prompt=[prompt] * len(pose_images)
        prompt_embeds = self.compel(prompt)
        negative_prompt=[negative_prompt] * len(pose_images)
        negative_prompt_embeds = self.compel(negative_prompt)
        # not sure if it's needed. see more here: https://github.com/damian0815/compel#0110---add-support-for-prompts-longer-than-the-models-max-token-length
        [prompt_embeds, negative_prompt_embeds] = self.compel.pad_conditioning_tensors_to_same_length([prompt_embeds, negative_prompt_embeds])

        #Video
        pipe.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))
        pipe.controlnet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))
        latents = torch.randn((1, 4, 64, 64), device="cuda", dtype=torch.float16).repeat(len(pose_images), 1, 1, 1)

        generator = torch.Generator("cuda").manual_seed(seed)
        output = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
            image=pose_images,
            latents=latents
        )
        
        result = output.images
        output_path = "/tmp/out-0.gif"
        imageio.mimsave(output_path, result, duration=duration, loop=0)
        output_path = [Path(output_path)]
        
        return output_path


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
