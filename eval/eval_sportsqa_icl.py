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
from tqdm import tqdm
import concurrent
import torch.nn.functional as F
import time
import deepspeed

torch.cuda.set_sync_debug_mode(1)
number_dict = {'none': '0', 'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6', 'seven': '7',
            'eight': '8', 'nine': '9', 'ten': '10', 'once': '1', 'twice': '2'}

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
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + f"\n{ex['question']} Answer in short words or a sentence.")
        conv.append_message(conv.roles[1], ex['answer'])
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

with open('/home/ubuntu/workspace/datasets/SportsQA/test_sim_all.json', 'r') as f:
    jf = json.load(f)
with open('/home/ubuntu/workspace/datasets/SportsQA/meta-data/train_all.json', 'r') as f:
    train_jf = json.load(f)
    train_dict = {}
    for i in train_jf:
        train_dict[i['qa_id']] = i

acc = []
jf = jf[7948:]
for q, data in enumerate(tqdm(jf)):
    video_path = "/home/ubuntu/workspace/datasets/" + data['video']
    max_frames_num = 32
    video,frame_time,video_time = load_video(video_path, max_frames_num, 1, force_sample=False)
    video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
    conv_template = "qwen_1_5"
    question = DEFAULT_IMAGE_TOKEN + f"\n{data['question']} Answer in short words or a sentence."
    max_text_outputs = None
    max_confi = 0
    candidates = data['train_examples']
    for num in range(4):
        conv = copy.deepcopy(conv_templates[conv_template])
        examples = candidates[:2]
        examples = [train_dict[x['train_question_id']] for x in examples]
        candidates = candidates[2:]
        conv, videos = put_examples(conv, examples)
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        videos.append(video)
        with torch.no_grad():
            cont = model.generate(
                input_ids,
                images=videos,
                modalities= ["video"] * len(videos),
                do_sample=False,
                temperature=0,
                max_new_tokens=50,
                output_scores=True,
                return_dict_in_generate=True
            )
        del input_ids
        torch.cuda.empty_cache()
        generated_tokens = cont.sequences
        text_outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()
        scores = cont.scores
        scores = torch.concat(list(scores), dim=0)
        probs = F.softmax(scores, dim=-1)
        prob = probs.max(dim=-1).values.min().item()
        if prob > 0.5:
            max_text_outputs = text_outputs
            max_confi = prob
            break
        else:
            if prob > max_confi:
                max_confi = prob
                max_text_outputs = text_outputs
    answer = data['answer'].lower()
    print(max_text_outputs, '///', answer)

    if answer in max_text_outputs.lower():
        acc.append(1)
    else:
        acc.append(0)
    
    print('total acc', round(sum(acc)/len(acc), 5), ['Wrong', 'Correct'][acc[-1]], round(max_confi, 3), 'try', num, 'shot', len(videos)-1, 'sportsqa')

    del cont, probs, generated_tokens, scores
    torch.cuda.empty_cache()
