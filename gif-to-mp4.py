# pip install diffusers transformers accelerate controlnet_aux mediapipe imageio==2.28.1 imageio[ffmpeg]

import torch
import os
from huggingface_hub import HfApi
from pathlib import Path
from PIL import Image
import numpy as np
from diffusers.utils import load_image
from controlnet_aux import OpenposeDetector
import imageio

# Load the GIF
gif = imageio.mimread('input.gif')

# Initialize the OpenposeDetector
processor = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')

# List to store the processed images
processed_images = []

for i, frame in enumerate(gif):
    # Convert imageio's image to PIL's
    image = Image.fromarray(frame)

    # Process image
    control_image = processor(image, hand_and_face=True)

    # Save the processed image to a numpy array
    processed_images.append(np.array(control_image))

# Create an mp4 video with the processed images
imageio.mimwrite('output.mp4', processed_images, fps=8)  # adjust fps as needed
