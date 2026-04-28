import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline
import torch

# load model and scheduler
model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(
    model_id, revision="fp16", torch_dtype=torch.float16
)
pipeline = pipeline.to("cuda")

# let's download an  image
# url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-upscale/low_res_cat.png"
# response = requests.get(url)
# low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
low_res_img = Image.open("C:\\dev\\resnet18-vae\\recon_cxr8.png").convert("RGB")
low_res_img = low_res_img.resize((128, 128))
# low_res_img.save("low_res_cat.png")
# prompt = "a white cat"
prompt = "a chest x-ray image with sharply defined anatomical boundaries"

upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]
upscaled_image.save("recon_cxr8_x4.png")
