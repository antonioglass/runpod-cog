''' StableDiffusion-v1 Predict Module '''

import os
from typing import List

import torch
from diffusers import (
    TextToVideoZeroPipeline,
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    PNDMScheduler
)

from PIL import Image
from cog import BasePredictor, Input, Path
import imageio
import numpy as np


class Predictor(BasePredictor):
    '''Predictor class for StableDiffusion-v1'''

    def setup(self):
        '''
        Load the model into memory to make running multiple predictions efficient
        '''
        print("Loading pipeline...")

        self.txt2img_pipe = TextToVideoZeroPipeline.from_pretrained(
            "./anime",
            safety_checker=None,
            torch_dtype=torch.float16
        ).to("cuda")


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
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7
        ),
        scheduler: str = Input(
            default="DPM-M",
            choices=["DDIM", "DDPM", "DPM-M", "DPM-S", "PNDM"],
            description="Choose a scheduler. If you use an init image, PNDM will be used",
        ),
        duration: int = Input(
            description="GIF duration / fps. duration = 1000 / fps", default=250
        ),
        video_length: int = Input(
            description="The number of generated video frames",
            default=8
        ),
        motion_field_strength_x: float = Input(
            description="Strength of motion in generated video along x-axis.",
            default=12
        ),
        motion_field_strength_y: float = Input(
            description="Strength of motion in generated video along y-axis.",
            default=12
        ),
        t0: int = Input(
            description="Timestep t0. Should be in the range [0, num_inference_steps - 1]",
            default=44
        ),
        t1: int = Input(
            description="Timestep t1. Should be in the range [t0 + 1, num_inference_steps - 1]",
            default=47
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

        generator = torch.Generator("cuda").manual_seed(seed)
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
            motion_field_strength_x=motion_field_strength_x,
            motion_field_strength_y=motion_field_strength_y,
            t0=t0,
            t1=t1,
            video_length=video_length
            
        )
        
        result = output.images
        result = [(r * 255).astype("uint8") for r in result]
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
        "DPM-M": DPMSolverMultistepScheduler.from_config(config),
        "DPM-S": DPMSolverSinglestepScheduler.from_config(config),
        "PNDM": PNDMScheduler.from_config(config)
    }[name]
