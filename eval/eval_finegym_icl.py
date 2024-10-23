from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
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
import random
random.seed(1)
import concurrent.futures
from tqdm import tqdm

def load_video(video_path, max_frames_num,fps=1,force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()

    return spare_frames,frame_time,video_time

def load_and_preprocess_video(video_path, max_frames_num):
    video, frame_time, video_time = load_video(video_path, max_frames_num, 1, force_sample=False)
    video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
    return video

def put_examples(conv, examples):
    videos = []
    video_paths = []
    for ex in examples:
        question = DEFAULT_IMAGE_TOKEN + f"\n{ex['question']} Answer in short words or a sentence."
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], ex['answer'])
        video_path = "/home/ubuntu/workspace/datasets/" + ex['video']
        video_paths.append(video_path)
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

with open('/home/ubuntu/workspace/datasets/SportsQA/meta-data/test_finegym.json', 'r') as f:
    jf = json.load(f)
with open('/home/ubuntu/workspace/datasets/SportsQA/meta-data/train_finegym.json', 'r') as f:
    train_jf = json.load(f)

score = []
for data in tqdm(jf):
    conv_template = "qwen_1_5"
    conv = copy.deepcopy(conv_templates[conv_template])
    examples = [x for x in train_jf if x['fg_type'] == data['fg_type'] and x['question'] == data['question']]
    if len(examples) < 2:
        videos = []
    else:
        examples = random.sample(examples, 2)
        conv, videos = put_examples(conv, examples)
    question = DEFAULT_IMAGE_TOKEN + f"\n{data['question']} Answer in short words or a sentence."
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    video_path = "/home/ubuntu/workspace/datasets/" + data['video']
    max_frames_num = 32
    video, frame_time, video_time = load_video(video_path, max_frames_num, 1, force_sample=True)
    video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
    videos.append(video)
    cont = model.generate(
        input_ids,
        images=videos,
        modalities= ["video"] * len(videos),
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip().lower()
    answer = data['answer'].lower()
    print(text_outputs, '///', answer)

    if answer in text_outputs:
        score.append(1)
    else:
        score.append(0)
    
    print(sum(score)/len(score))

