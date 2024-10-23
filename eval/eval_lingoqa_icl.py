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
import concurrent.futures
from tqdm import tqdm
import os

def load_frames(image_paths):
    images = [np.array(Image.open(x)) for x in image_paths]
    images = np.stack(images, axis=0)
    return images

def load_and_preprocess_video(video_path, max_frames_num):
    video = load_frames(video_path)
    video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
    return video

def put_examples(conv, examples):
    videos = []
    video_paths = []
    for _, ex in examples.iterrows():
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + f"\n{ex['question']}")
        conv.append_message(conv.roles[1], ex['answer'])
        if os.path.exists("/home/ubuntu/workspace/datasets/LingoQA/action/" + ex['images'][0]):
            image_paths = ["/home/ubuntu/workspace/datasets/LingoQA/action/"  + x for x in ex['images']]
        else:
            image_paths = ["/home/ubuntu/workspace/datasets/LingoQA/scenery/" + x for x in ex['images']]
        video_paths.append(image_paths)
    max_frames_num = 32
    with concurrent.futures.ThreadPoolExecutor() as executor:
        videos = list(executor.map(lambda p: load_and_preprocess_video(p, max_frames_num), video_paths))
    return conv, videos

pretrained = "checkpoints/LLaVA-Video-7B-Qwen2"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map)  # Add any other thing you want to pass in llava_model_args
model.eval()
judge_name = 'wayveai/Lingo-Judge'
pipe = pipeline("text-classification", model=judge_name)
df = pd.read_parquet('/home/ubuntu/workspace/datasets/LingoQA/evaluation/val.parquet')
df1 = pd.read_parquet('/home/ubuntu/workspace/datasets/LingoQA/action/train.parquet')
df2 = pd.read_parquet('/home/ubuntu/workspace/datasets/LingoQA/scenery/train.parquet')
train_df = pd.concat([df1, df2], axis=0, ignore_index=True)
with open('/home/ubuntu/workspace/datasets/LingoQA/text_top2k_sim.json', 'r') as f:
    sim_jf = json.load(f)
    sim = {}
    for i in sim_jf:
        sim[i['test_id']] = i["train_ids"]

scores = []
pbar = tqdm(total=len(df))
for _, data in df.iterrows():
    example_ids = sim[data['question_id']][:16]
    examples = train_df[train_df['question_id'].isin(example_ids)]
    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
    question = DEFAULT_IMAGE_TOKEN + f"\n{data['question']}"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv, videos = put_examples(conv, examples)
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_paths = ["/home/ubuntu/workspace/datasets/LingoQA/evaluation/" + x for x in data['images']]
    video = load_frames(image_paths)
    video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
    videos.append(video)
    cont = model.generate(
        input_ids,
        images=videos,
        modalities=["video"]*len(videos),
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