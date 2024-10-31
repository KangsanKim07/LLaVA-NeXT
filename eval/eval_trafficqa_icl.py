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
    for _, ex in examples.iterrows():
        video_path = "/home/ubuntu/workspace/datasets/SUTD-TrafficQA/raw_videos/compressed_videos/" + ex['vid_filename']
        video_paths.append(video_path)
        q_body = ex['q_body']
        options = [ex['option0'], ex['option1'], ex['option2'], ex['option3']]
        options = [x if x != "" else "Not sure" for x in options]
        answer = ['A', 'B', 'C', 'D'][ex['answer']]
        
        question = DEFAULT_IMAGE_TOKEN + f"\n{q_body}\nOptions:\n(A) {options[0]}\n(B) {options[1]}\n(C) {options[2]}\n(D) {options[3]}\nAnswer with the option's letter from the given choices directly."
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], f"The answer is ({answer}).")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        videos = list(executor.map(lambda p: load_and_preprocess_video(p, max_frames_num), video_paths))
    return conv, videos

pretrained = "checkpoints/LLaVA-Video-7B-Qwen2"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"

tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map) 
model.eval()

df = pd.read_json("/home/ubuntu/workspace/datasets/SUTD-TrafficQA/R2_test_examples.jsonl", lines=True)

file_path = '/home/ubuntu/workspace/datasets/SUTD-TrafficQA/annotations/R2_train.jsonl'
with open(file_path, 'r') as file:
    header_line = file.readline().strip()
    header = eval(header_line) 
train_df = pd.read_json(file_path, lines=True)
train_df = train_df[1:]
train_df.columns = header
acc = []

pbar = tqdm(total=len(df))
for q, data in df[::-1].iterrows():
    video_path = "/home/ubuntu/workspace/datasets/SUTD-TrafficQA/raw_videos/compressed_videos/" + data['vid_filename']
    max_frames_num = 32
    video,frame_time,video_time = load_video(video_path, max_frames_num, 1, force_sample=False)
    video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
    q_body = data['q_body']
    options = [data['option0'], data['option1'], data['option2'], data['option3']]
    options = [x if x != "" else "Not sure" for x in options]
    test_question = DEFAULT_IMAGE_TOKEN + f"\n{q_body}\nOptions:\n(A) {options[0]}\n(B) {options[1]}\n(C) {options[2]}\n(D) {options[3]}\nAnswer with the option's letter from the given choices directly."
    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
    train_examples = [x['train_record_id'] for x in data['train_examples']]
    answer = ['A', 'B', 'C', 'D'][data['answer']]
    max_text_outputs = None
    max_confi = 0
    for num in range(4):
        example_ids = train_examples[:4]
        train_examples = train_examples[4:]
        examples = train_df[train_df['record_id'].isin(example_ids)]
        conv = copy.deepcopy(conv_templates[conv_template])
        conv, videos = put_examples(conv, examples)
        conv.append_message(conv.roles[0], test_question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt() + "The answer is ("
        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        videos.append(video)
        with torch.no_grad():
            cont = model.generate(
                input_ids,
                images=videos,
                modalities= ["video"] * len(videos),
                do_sample=False,
                temperature=0,
                max_new_tokens=100,
                output_scores=True,
                return_dict_in_generate=True
            )
        del input_ids
        for x in videos:
            del x
        torch.cuda.empty_cache()

        generated_tokens = cont.sequences
        text_outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()
        # import pdb; pdb.set_trace()
        score = cont.scores[0]
        probs = F.softmax(score, dim=-1)
        prob = torch.max(probs[0]).item()
        if prob > 0.7:
            max_text_outputs = text_outputs
            max_confi = prob
            break
        else:
            if prob > max_confi:
                max_confi = prob
                max_text_outputs = text_outputs

    print(answer, max_text_outputs)
    if max_text_outputs != "" and answer in max_text_outputs[0]:
        acc.append(1)
    else:
        acc.append(0)
    
    print(round(sum(acc)/len(acc), 5), [False, True][acc[-1]], round(max_confi, 3), num, 'shot', len(videos)-1)
    pbar.update(1)
    del cont, probs, generated_tokens, score
    torch.cuda.empty_cache()

pbar.close()
