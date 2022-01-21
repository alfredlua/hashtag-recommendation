from typing import Optional
from fastapi import FastAPI, File, UploadFile

import clip
import torch
from models import ClipCaptionModel
from caption import generate_beam, generate2
from transformers import GPT2Tokenizer

from PIL import Image
from io import BytesIO

app = FastAPI()

CPU = torch.device('cpu')

device = "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

prefix_length = 10

model = ClipCaptionModel(prefix_length)

model.load_state_dict(torch.load("./coco_weights.pt", map_location=CPU)) 

model = model.eval() 
device = "cpu"
model = model.to(device)

use_beam_search = True #@param {type:"boolean"}

def get_caption(pil_image):
    
    image = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        # if type(model) is ClipCaptionE2E:
        #     prefix_embed = model.forward_image(image)
        # else:
        prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
        prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)
    if use_beam_search:
        generated_text_prefix = generate_beam(model, tokenizer, embed=prefix_embed)[0]
    else:
        generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed)

    return generated_text_prefix

@app.get("/")
def read_root():
    return {"Hello World"}

@app.post("/infer/")
async def infer(image: UploadFile = File(...)):
    pil_image = Image.open(BytesIO(await image.read()))
    caption = get_caption(pil_image)
    return {"caption": caption}