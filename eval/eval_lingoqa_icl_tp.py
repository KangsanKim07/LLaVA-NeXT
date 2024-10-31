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
import torch.nn.functional as F
from llava.confidence.llava_with_confidence import LLavaWithConfidence

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

# model_path = "/home/ubuntu/workspace/LLaVA-NeXT/checkpoints/finetuned/llava-video_icl/checkpoint-300"
# model_base = "checkpoints/LLaVA-Video-7B-Qwen2"
# model_name = "lora_llava_qwen"
model_path = "checkpoints/LLaVA-Video-7B-Qwen2"
model_base = None
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(model_path, model_base, model_name, torch_dtype="bfloat16", device_map=device_map)  # Add any other thing you want to pass in llava_model_args
model.eval()
confidence_model = LLavaWithConfidence(model)

judge_name = 'wayveai/Lingo-Judge'
pipe = pipeline("text-classification", model=judge_name)
df = pd.read_json("/home/ubuntu/workspace/datasets/LingoQA/test_examples.jsonl", lines=True)
df1 = pd.read_parquet('/home/ubuntu/workspace/datasets/LingoQA/action/train.parquet')
df2 = pd.read_parquet('/home/ubuntu/workspace/datasets/LingoQA/scenery/train.parquet')
train_df = pd.concat([df1, df2], axis=0, ignore_index=True)
with open('/home/ubuntu/workspace/datasets/LingoQA/text_top2k_sim.json', 'r') as f:
    sim_jf = json.load(f)
    sim = {}
    for i in sim_jf:
        sim[i['test_id']] = i["train_ids"]

acc = []
correct, wrong = [], []
pbar = tqdm(total=len(df))
for _, data in df.iterrows():
    train_examples = [x['train_question_id'] for x in data['train_examples']]
    image_paths = ["/home/ubuntu/workspace/datasets/LingoQA/evaluation/" + x for x in data['images']]
    video = load_frames(image_paths)
    video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
    max_text_outputs = None
    max_confi = 0
    for num in range(4):
        example_ids = train_examples[:2]
        train_examples = train_examples[2:]
        examples = train_df[train_df['question_id'].isin(example_ids)]
        conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
        question = DEFAULT_IMAGE_TOKEN + f"\n{data['question']}"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv, videos = put_examples(conv, examples)
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        videos.append(video)
        cont, confidence = confidence_model.generate(
            input_ids,
            images=videos,
            modalities=["video"]*len(videos),
            do_sample=False,
            temperature=0,
            max_new_tokens=100
        )
        text_outputs = tokenizer.decode(cont.sequences[0], skip_special_tokens=True).strip()
        if confidence > 0.8:
            max_text_outputs = text_outputs
            max_confi = confidence
            break
        else:
            if confidence > max_confi:
                max_text_outputs = text_outputs
                max_confi = confidence
    answer = data['answer']    
    input = f"[CLS]\nQuestion: {data['question']}\nAnswer: {answer}\nStudent: {max_text_outputs}"
    result = pipe(input)
    score = result[0]['score']
    if score > 0.8:
        acc.append(1)
        correct.append(max_confi)
    else:
        acc.append(0)
        wrong.append(max_confi)
    print(round(sum(acc)/len(acc), 5), [False, True][acc[-1]], round(max_confi, 3), 'trial', num, 'shot', len(videos)-1, model_name)
    # print('correct prob', sum(correct)/(len(correct)+1e-5))
    # print('wrong prob', sum(wrong)/(len(wrong)+1e-5))
    pbar.update(1)

pbar.close()