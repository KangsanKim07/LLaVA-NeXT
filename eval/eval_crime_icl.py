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
import torch.nn.functional as F
import concurrent.futures
import random
random.seed(1)

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

def put_examples(conv, examples, question):
    videos = []
    video_paths = []
    max_frames_num = 32
    for ex in reversed(examples):
        ex_answer = ex.split('_')[0].split('0')[0]
        if ex_answer == 'Normal':
            ex_answer = "Normal Event"
        elif ex_answer == 'RoadAccidents':
            ex_answer = 'Road Accident'
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + f"\n{question}")
        conv.append_message(conv.roles[1], ex_answer)

        folder = ex.split('_')[0].split('0')[0]
        if folder == "Normal":
            folder = "Normal_Videos_event"
        video_path = f"/home/ubuntu/workspace/datasets/UCF_Crimes/videos/{folder}/{ex}"
        video_paths.append(video_path)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        videos = list(executor.map(lambda p: load_and_preprocess_video(p, max_frames_num), video_paths))
    return conv, videos


# model_path = "/home/ubuntu/workspace/LLaVA-NeXT/checkpoints/finetuned/llava-video_icl-f32nosf2shot/checkpoint-1200"
# model_base = "checkpoints/LLaVA-Video-7B-Qwen2"
# model_name = "lora_llava_qwen"
model_path = "checkpoints/LLaVA-Video-7B-Qwen2"
model_base = None
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"

tokenizer, model, image_processor, max_length = load_pretrained_model(model_path, model_base, model_name, torch_dtype="bfloat16", device_map=device_map)  # Add any other thing you want to pass in llava_model_args
model.eval()

test_vids = []
test_subset_id = 3
with open(f'/home/ubuntu/workspace/datasets/UCF_Crimes/Action_Regnition_splits/test_00{test_subset_id}.txt', 'r') as f:
    test_vids += f.readlines()
test_vids = [x.replace(' \n', '') for x in test_vids]

with open(f'/home/ubuntu/workspace/datasets/UCF_Crimes/Action_Regnition_splits/train_00{test_subset_id}.txt', 'r') as f:
    train_vids = f.readlines()
train_vids = [x.replace(' \n', '').split('/')[-1] for x in train_vids]

with open(f'/home/ubuntu/workspace/datasets/UCF_Crimes/video_top400_sim_subset{test_subset_id}.json') as f:
    jf = json.load(f)
    sim_rank = {}
    for i in jf:
        sim_rank[i['test_sample']] = i['train_examples'] 

question_sample = "Classify the following video into one of the following categories: Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting, Normal Event, Road Accident, Robbery, Shooting, Shoplifting, Stealing, or Vandalism. Just answer the name of the category."
acc = []
for path in tqdm(test_vids):
    video_path = "/home/ubuntu/workspace/datasets/UCF_Crimes/videos/" + path
    max_frames_num = 32
    video,frame_time,video_time = load_video(video_path, max_frames_num, 1, force_sample=False)
    video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
    question = DEFAULT_IMAGE_TOKEN + f"\n{question_sample}"
    max_text_outputs = None
    max_confi = 0
    candidates = sim_rank[path.split('/')[-1]]
    for num in range(1):
        conv = copy.deepcopy(conv_templates[conv_template])
        # examples = candidates[:2]
        # candidates = candidates[2:]
        examples = random.sample(train_vids, 2)
        conv, videos = put_examples(conv, examples, question_sample)
        videos.append(video)
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

        with torch.no_grad():
            cont = model.generate(
                input_ids,
                images=videos,
                modalities=["video"]*len(videos),
                do_sample=False,
                temperature=0,
                max_new_tokens=4096,
                output_scores=True,
                return_dict_in_generate=True
            )
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
    answer = path.split('/')[0]
    if answer == 'RoadAccidents':
        answer = 'Road Accident'
    elif answer == 'Normal_Videos_event':
        answer = 'Normal Event'
    print(max_text_outputs, '///', answer)

    if answer.lower() in max_text_outputs.lower():
        acc.append(1)
    else:
        acc.append(0)
    
    print('total acc', round(sum(acc)/len(acc), 5), [False, True][acc[-1]], round(max_confi, 3), 'try', num, 'shot', len(videos)-1)
    print("subset", test_subset_id)
