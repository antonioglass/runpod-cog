''' infer.py for runpod worker '''

import os
import predict

import runpod
from runpod.serverless.utils import rp_download, rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate


MODEL = predict.Predictor()
MODEL.setup()


INPUT_SCHEMA = {
    'prompt': {
        'type': str,
        'required': True
    },
    'negative_prompt': {
        'type': str,
        'required': False,
        'default': None
    },
    'width': {
        'type': int,
        'required': False,
        'default': 512,
        'constraints': lambda width: width in [128, 256, 384, 448, 512, 576, 640, 704, 768]
    },
    'height': {
        'type': int,
        'required': False,
        'default': 512,
        'constraints': lambda height: height in [128, 256, 384, 448, 512, 576, 640, 704, 768]
    },
    'num_inference_steps': {
        'type': int,
        'required': False,
        'default': 50,
        'constraints': lambda num_inference_steps: 0 < num_inference_steps < 500
    },
    'guidance_scale': {
        'type': float,
        'required': False,
        'default': 7,
        'constraints': lambda guidance_scale: 0 < guidance_scale < 20
    },
    'scheduler': {
        'type': str,
        'required': False,
        'default': 'DPM-M',
        'constraints': lambda scheduler: scheduler in ['DDIM', 'DDPM', 'DPM-M', 'DPM-S', 'PNDM']
    },
    'seed': {
        'type': int,
        'required': False,
        'default': None
    },
    'nsfw': {
        'type': bool,
        'required': False,
        'default': False
    },
    'duration': {
        'type': int,
        'required': False,
        'default': 250
    },
    'video_length': {
        'type': int,
        'required': False,
        'default': 8
    },
    'motion_field_strength_x': {
        'type': float,
        'required': False,
        'default': 12
    },
    'motion_field_strength_y': {
        'type': float,
        'required': False,
        'default': 12
    },
    't0': {
        'type': int,
        'required': False,
        'default': 44
    },
    't1': {
        'type': int,
        'required': False,
        'default': 47
    }
}


def run(job):
    '''
    Run inference on the model.
    Returns output path, width the seed used to generate the image.
    '''
    job_input = job['input']

    # Input validation
    validated_input = validate(job_input, INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    validated_input = validated_input['validated_input']

    MODEL.NSFW = job_input.get('nsfw', True)

    if job_input['seed'] is None:
        job_input['seed'] = int.from_bytes(os.urandom(2), "big")

    img_paths = MODEL.predict(
        prompt=job_input["prompt"],
        negative_prompt=job_input.get("negative_prompt", None),
        width=job_input.get('width', 512),
        height=job_input.get('height', 512),
        num_inference_steps=job_input.get('num_inference_steps', 50),
        guidance_scale=job_input['guidance_scale'],
        scheduler=job_input.get('scheduler', "DPM-M"),
        duration=job_input.get('duration', 250),
        video_length=job_input.get('video_length', 8),
        motion_field_strength_x=job_input.get('motion_field_strength_x', 12),
        motion_field_strength_y=job_input.get('motion_field_strength_y', 12),
        t0=job_input.get('t0', 44),
        t1=job_input.get('t1', 47),
        seed=job_input['seed']
    )

    job_output = []
    for index, img_path in enumerate(img_paths):
        image_url = rp_upload.upload_image(job['id'], img_path, index)

        job_output.append({
            "image": image_url,
            "seed": job_input['seed'] + index
        })

    # Remove downloaded input objects
    rp_cleanup.clean(['input_objects'])

    return job_output


runpod.serverless.start({"handler": run})
