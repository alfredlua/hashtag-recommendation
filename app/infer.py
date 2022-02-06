import clip
import torch
from models import ClipCaptionModel
from caption import generate_beam, generate2
from transformers import GPT2Tokenizer
import skimage.io as io
import PIL.Image

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

image = io.imread("./Images/museum.jpg")
pil_image = PIL.Image.fromarray(image)

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

print(generated_text_prefix)