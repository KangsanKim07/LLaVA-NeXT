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
from tqdm import tqdm
import concurrent
import torch.nn.functional as F
import time
torch.cuda.set_sync_debug_mode(1)
from llava.confidence.llava_with_confidence import LLavaWithConfidence

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
    max_frames_num = 32
    for ex in reversed(examples):
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + f"\n{ex['question']}")
        conv.append_message(conv.roles[1], f"The answer is ({ex['answer']})")
        video_path = f"/home/ubuntu/workspace/datasets/{ex['video']}"
        video_paths.append(video_path)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        videos = list(executor.map(lambda p: load_and_preprocess_video(p, max_frames_num), video_paths))
    return conv, videos

pretrained = "checkpoints/LLaVA-Video-7B-Qwen2"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"

tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map)  # Add any other thing you want to pass in llava_model_args
model.eval()
confidence_model = LLavaWithConfidence(model)

with open("/home/ubuntu/workspace/datasets/AnimalKingdom/annotation/val_mc.json", "r") as f:
    df = json.load(f)
with open('/home/ubuntu/workspace/datasets/AnimalKingdom/video_top500_sim.json', 'r') as f:
    simrank = json.load(f)
with open('/home/ubuntu/workspace/datasets/AnimalKingdom/annotation/train_mc.json', 'r') as f:
    train_jf = json.load(f)
    train_samples = {}
    for data in train_jf:
        train_samples[data['video'].split('/')[-1]] = data

acc = []
pbar = tqdm(total=len(df))
shot = 2
for idx, data in enumerate(df):
    video_path = "/home/ubuntu/workspace/datasets/" + data['video']
    max_frames_num = 32
    video,frame_time,video_time = load_video(video_path, max_frames_num, 1, force_sample=False)
    video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
    question =  DEFAULT_IMAGE_TOKEN + "\n" + data['question']
    max_text_outputs = None
    max_confi = 0
    try:
        candidates = simrank[data['video'].split('/')[-1]]['train_examples']
    except:
        candidates = [None] * 10
    for num in range(4):
        try:
            conv = copy.deepcopy(conv_templates[conv_template])
            examples = candidates[:shot]
            examples = [train_samples[x] for x in examples]
            candidates = candidates[shot:]
            conv, videos = put_examples(conv, examples)
        except:
            conv = copy.deepcopy(conv_templates[conv_template])
            videos = []
        videos.append(video)
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt() + "The answer is ("
        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        with torch.no_grad():
            cont, confidence = confidence_model.generate(
                input_ids,
                images=videos,
                modalities=["video"]*len(videos),
                do_sample=False,
                temperature=0,
                max_new_tokens=3,
            )
        del input_ids, videos
        torch.cuda.empty_cache()
        pred = tokenizer.batch_decode(cont.sequences, skip_special_tokens=True)[0].strip()
        if confidence > 0.7:
            max_text_outputs = pred
            max_confi = confidence
            break
        else:
            if confidence > max_confi:
                max_confi = confidence
                max_text_outputs = pred

    answer = data['answer']

    if max_text_outputs != "" and answer == max_text_outputs[0]:
        acc.append(1)
    else:
        acc.append(0)
    
    print('total acc', round(sum(acc)/len(acc), 5), ['Wrong', 'Correct'][acc[-1]], round(max_confi, 3), 'try', num, 'shot', shot)
    pbar.update(1)
    del cont, pred
    torch.cuda.empty_cache()

pbar.close()
