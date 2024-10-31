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

pretrained = "checkpoints/LLaVA-Video-7B-Qwen2"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"

tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map)  # Add any other thing you want to pass in llava_model_args
model.eval()

with open('/home/ubuntu/workspace/datasets/driveandact/midlevel_split2_test.json', 'r') as f:
    jf = json.load(f)

question_sample = "Classify the following video into one of the following categories: working_on_laptop, opening_door_inside, reading_newspaper, opening_bottle, looking_or_moving_around (e.g. searching), closing_door_outside, opening_laptop, unfastening_seat_belt, fastening_seat_belt, opening_backpack, taking_laptop_from_backpack, reading_magazine, entering_car, putting_on_sunglasses, putting_laptop_into_backpack, opening_door_outside, closing_door_inside, putting_on_jacket, pressing_automation_button, closing_laptop, sitting_still, interacting_with_phone, preparing_food, eating, drinking, closing_bottle, talking_on_phone, using_multimedia_display, taking_off_sunglasses, writing, placing_an_object, exiting_car, fetching_an_object, and taking_off_jacket. Just answer the name of the category."
score = []
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
    try:
        spare_frames = current_video.get_batch(frame_idx).asnumpy()
    except:
        import pdb; pdb.set_trace()
    video = image_processor.preprocess(spare_frames, return_tensors="pt")["pixel_values"].cuda().bfloat16()
    video = [video]
    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
    question = DEFAULT_IMAGE_TOKEN + f"\n{question_sample}"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

    with torch.no_grad():
        cont = model.generate(
            input_ids,
            images=video,
            modalities="video",
            do_sample=False,
            temperature=0,
            max_new_tokens=100,
        )
    del input_ids, video
    torch.cuda.empty_cache()
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
    answer = data['activity']

    print(text_outputs, '///', answer)

    if answer.lower() in text_outputs.lower():
        score.append(1)
    else:
        score.append(0)
    
    print(sum(score)/len(score))
    pbar.update(1)

pbar.close()