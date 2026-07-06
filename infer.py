# Use the modified diffusers & peft library
import sys
import os
workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))

if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image

pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
pipe.to("cuda")

input_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")

image = pipe(
  image=input_image,
  prompt="Add a hat to the cat",
  guidance_scale=2.5
).images[0]

image.save("edit.jpg")