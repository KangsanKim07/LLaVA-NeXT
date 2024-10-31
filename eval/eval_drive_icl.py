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
import pandas as pd
import torch.nn.functional as F

pretrained = "checkpoints/LLaVA-Video-7B-Qwen2"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"

tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map)  # Add any other thing you want to pass in llava_model_args
model.eval()

part = 0
with open(f'/home/ubuntu/workspace/datasets/driveandact/midlevel_split{part}_test.json', 'r') as f:
    jf = json.load(f)
with open(f'/home/ubuntu/workspace/datasets/driveandact/midlevel_split{part}_train.json', 'r') as f:
    train_jf = json.load(f)
    train_data = {}
    for x in train_jf:
        train_data[f"{x['file_id']}@{x['frame_start']}@{x['frame_end']}@{x['activity']}"] = x

with open(f"/home/ubuntu/workspace/datasets/driveandact/video_top500_sim{part}.json", 'r') as f:
    simrank = json.load(f)

question_sample = "Classify the following video into one of the following categories: working_on_laptop, opening_door_inside, reading_newspaper, opening_bottle, looking_or_moving_around (e.g. searching), closing_door_outside, opening_laptop, unfastening_seat_belt, fastening_seat_belt, opening_backpack, taking_laptop_from_backpack, reading_magazine, entering_car, putting_on_sunglasses, putting_laptop_into_backpack, opening_door_outside, closing_door_inside, putting_on_jacket, pressing_automation_button, closing_laptop, sitting_still, interacting_with_phone, preparing_food, eating, drinking, closing_bottle, talking_on_phone, using_multimedia_display, taking_off_sunglasses, writing, placing_an_object, exiting_car, fetching_an_object, and taking_off_jacket. Just answer the name of the category."
acc = []
pbar = tqdm(total=len(jf))
current_video_path = None
current_video = None
for data in jf:
    video_path = f"/home/ubuntu/workspace/datasets/driveandact/kinect_ir/{data['file_id']}.mp4"
    if current_video_path != video_path:
        current_video = VideoReader(video_path, ctx=cpu(0),num_threads=1)
        current_video_path = video_path
        fps = round(current_video.get_avg_fps()/1)
    
    frame_start, frame_end = data['frame_start'], data['frame_end']
    frame_idx = [i for i in range(frame_start, frame_end, fps)]
    spare_frames = current_video.get_batch(frame_idx).asnumpy()
    video = image_processor.preprocess(spare_frames, return_tensors="pt")["pixel_values"].cuda().bfloat16()
    
    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
    question = DEFAULT_IMAGE_TOKEN + f"\n{question_sample}"

    test_id = f"{data['file_id']}@{data['frame_start']}@{data['frame_end']}@{data['activity']}"
    candidates = simrank[test_id]['train_examples']
    max_text_outputs = None
    max_confi = 0
    for num in range(4):
        example_ids = candidates[:2]
        examples = [train_data[x] for x in example_ids]
        candidates = candidates[2:]
        conv = copy.deepcopy(conv_templates[conv_template])
        videos = []
        for ex in examples:
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], ex['activity'])
            ex_video_path = f"/home/ubuntu/workspace/datasets/driveandact/kinect_ir/{ex['file_id']}.mp4"
            ex_video = VideoReader(video_path, ctx=cpu(0),num_threads=1)
            ex_frame_start, ex_frame_end = data['frame_start'], data['frame_end']
            ex_frame_idx = [i for i in range(ex_frame_start, ex_frame_end, fps)]
            ex_spare_frames = ex_video.get_batch(ex_frame_idx).asnumpy()
            ex_video_tensors = image_processor.preprocess(ex_spare_frames, return_tensors="pt")["pixel_values"].cuda().bfloat16()
            videos.append(ex_video_tensors)

        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

        with torch.no_grad():
            cont = model.generate(
                input_ids,
                images=videos,
                modalities=["video"] * len(videos),
                do_sample=False,
                temperature=0,
                max_new_tokens=10,
                output_scores=True,
                return_dict_in_generate=True
            )
        del input_ids, videos
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
    answer = data['activity']

    print(max_text_outputs, '///', answer)

    if answer.lower() in max_text_outputs.lower():
        acc.append(1)
    else:
        acc.append(0)
    
    print('total acc', round(sum(acc)/len(acc), 5), ['Wrong', 'Correct'][acc[-1]], round(max_confi, 3), 'try', num, 'shot', len(videos)-1, 'sportsqa')
    pbar.update(1)

    del cont, probs, generated_tokens, scores
    torch.cuda.empty_cache()

pbar.close()