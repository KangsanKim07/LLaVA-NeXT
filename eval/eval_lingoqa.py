from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates
from PIL import Image
import requests
import copy
import torch
import sys
import json
import warnings
from decord import VideoReader, cpu
import numpy as np
warnings.filterwarnings("ignore")
import pandas as pd
from PIL import Image
from transformers import pipeline
from tqdm import tqdm

def load_frames(image_paths):
    images = [np.array(Image.open(x)) for x in image_paths]
    images = np.stack(images, axis=0)
    return images

pretrained = "checkpoints/LLaVA-Video-7B-Qwen2"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map)  # Add any other thing you want to pass in llava_model_args
model.eval()
judge_name = 'wayveai/Lingo-Judge'
pipe = pipeline("text-classification", model=judge_name)
df = pd.read_parquet('/home/ubuntu/workspace/datasets/LingoQA/evaluation/val.parquet')

scores = []
pbar = tqdm(total=len(df))
for _, data in df.iterrows():
    image_paths = ["/home/ubuntu/workspace/datasets/LingoQA/evaluation/" + x for x in data['images']]
    video = load_frames(image_paths)
    video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
    video = [video]
    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
    question = DEFAULT_IMAGE_TOKEN + f"\n{data['question']}"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    cont = model.generate(
        input_ids,
        images=video,
        modalities=["video"],
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
    answer = data['answer']
    
    input = f"[CLS]\nQuestion: {data['question']}\nAnswer: {answer}\nStudent: {text_outputs}"
    result = pipe(input)
    score = result[0]['score']
    if score > 0.5:
        scores.append(1)
    else:
        scores.append(0)
    print(sum(scores)/len(scores))
    pbar.update(1)

pbar.close()
