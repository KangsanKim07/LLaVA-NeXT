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

pretrained = "checkpoints/LLaVA-Video-7B-Qwen2"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"

tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map)  # Add any other thing you want to pass in llava_model_args
model.eval()

df = pd.read_json("/home/ubuntu/workspace/datasets/SUTD-TrafficQA/R2_test_examples.jsonl", lines=True)

score = []
pbar = tqdm(total=len(df))
for _, data in df.iterrows():
    video_path = "/home/ubuntu/workspace/datasets/SUTD-TrafficQA/raw_videos/compressed_videos/" + data['vid_filename']
    max_frames_num = 64
    video,frame_time,video_time = load_video(video_path, max_frames_num, 1, force_sample=False)
    video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
    video = [video]
    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
    q_body = data['q_body']
    options = [data['option0'], data['option1'], data['option2'], data['option3']]
    options = [x if x != "" else "Not sure" for x in options]
    
    question = DEFAULT_IMAGE_TOKEN + f"\n{q_body}\nOptions:\n(A) {options[0]}\n(B) {options[1]}\n(C) {options[2]}\n(D) {options[3]}\nAnswer with the option's letter from the given choices directly."
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt() + "("
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    cont = model.generate(
        input_ids,
        images=video,
        modalities= ["video"],
        do_sample=False,
        temperature=0,
        max_new_tokens=10,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
    answer = ['A', 'B', 'C', 'D'][data['answer']]

    if text_outputs != "" and answer in text_outputs[0]:
        score.append(1)
    else:
        score.append(0)
    
    print(sum(score)/len(score))
    pbar.update(1)

pbar.close()
