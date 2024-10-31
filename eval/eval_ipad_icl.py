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
from tqdm import tqdm
import os
import random
random.seed(1)
import torch.nn.functional as F

def load_frames(image_paths):
    images = [np.array(Image.open(x)) for x in image_paths]
    images = np.stack(images, axis=0)
    return images

pretrained = "checkpoints/LLaVA-Video-7B-Qwen2"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map)  # Add any other thing you want to pass in llava_model_args
model.eval()

with open('/home/ubuntu/workspace/datasets/ipad/test_label.json', 'r') as f:
    jf = json.load(f)

criteria = {
    "R01": "The button switch should be red, and the body should be black. It should have six connection legs. The red switch should face right, with the legs facing left. The conveyor belt speed should be consistent.",
    "R02": "The object placed on the lifter should be positioned at the center of the lifter. The object should primarily be black or gray in color. The object should be rectangular in shape, with its longer side placed horizontally. The lifter should lift the object smoothly, without any stuttering. The object should be placed on the lifter and then removed afterward.",
    "R03": "All objects placed on the pallet must be white and should not be stacked on top of each other. The forklift must insert its forks as deeply as possible into the pallet before lifting. The pallet must be lifted. Objects should not fall off during the lifting process.",
    "R04": "The cutting process must be successful. The object being cut should be beige in color and have an intact rectangular shape. The object must be cut at a precise vertical angle; angled cuts are not allowed. After cutting, the blade should be raised back to its original height before cutting. The cut must align precisely with the designated cut line.",
    "S01": "The box on the conveyor should be beige in color. The box must have a cubic shape. The conveyor belt speed should remain constant. The box must be placed parallel to the conveyor belt; it should not be positioned at an angle.",
    "S02": "The object on the lifter must be placed at the exact center. The lifter should lift the object smoothly, without any stuttering. The object should be placed on the lifter, lifted, and then removed. The object must be a beige-colored cube. The longer side of the box should be placed parallel to the lifter.",
    "S03": "The object being lifted must be a single beige-colored cube. The forklift and the object should not be positioned too close to each other during lifting. The object must be lifted successfully. The object should not fall during the lifting process.",
    "S04": "The cutting process must be completed successfully. The object being cut should be beige in color and have an intact rectangular shape. The object must be cut at a precise vertical angle; angled cuts are not acceptable. After cutting, the blade should be raised back to its original height before the cut. The cut should align precisely with the designated cut line.",
    "S05": "The object on the conveyor should move at a constant speed. The object should move from the top right to the bottom left. The object on the conveyor must be a beige-colored cube-shaped box. The object should be positioned at the center of the conveyor.",
    "S06": "The object on the conveyor should move at a constant speed. The object should move from the top left to the bottom right. The object on the conveyor must be a beige-colored cube-shaped box. The object should be positioned at the center of the conveyor.",
    "S07": "The object being lifted should be a beige-colored cube-shaped box. The object must not fall in the middle of the process. The object should drop to the floor at the end; it should not get clogged or obstructed.",
    "S08": "The color of the placed box must match the color of the storage bin. The box must be placed completely inside the storage bin.",
    "S09": "The object must not fall in the middle of the process. The object should be lifted successfully. The object must be a beige-colored cube-shaped box. The orientation of the gripper and the box should be parallel.",
    "S10": "The object must not fall in the middle of the process. The crane should place the object down only after reaching the far right end; it should not be placed down midway. The object should be lifted parallel to the crane and should not be tilted. The object must be lifted successfully. The object should be a beige-colored cube-shaped box.",
    "S11": "The object must be cut successfully. The object should be cut precisely along the designated cut line. The cut pieces should be neatly stacked on top of each other.",
    "S12": "The object must be split into two parts. The object should be positioned deep onto the base plate. The drill should align precisely with the designated line. The drilling depth should not be too shallow."
}

acc = []
pbar = tqdm(total=len(jf))
for name, label in jf.items():
    max_text_outputs = None
    max_confi = 0
    category = name.split('/')[1]
    crit = criteria[category]
    question = f"You are an anomaly detection model. Watch the video and determine whether it contains any anomalies based on the given criteria.\nCriteria: {crit}\nIf an anomaly is detected, respond with 'yes'; if not, respond with 'no'."
    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
    full_question = DEFAULT_IMAGE_TOKEN + f"\n{question}"
    for num in range(8):
        conv = copy.deepcopy(conv_templates[conv_template])
        videos = []
        for i in range(2):
            conv.append_message(conv.roles[0], full_question)
            conv.append_message(conv.roles[1], 'no')
            train_path = f'/home/ubuntu/workspace/datasets/ipad/{name}'
            folder = str(random.randint(1, 20))
            if len(folder) == 1:
                folder = '0' + folder
            train_path = train_path.replace('testing', 'training')[:-2] + folder
            image_paths = [f'{train_path}/' + x for x in os.listdir(train_path)]
            image_paths = [image_paths[i] for i in range(0, len(image_paths), 25)]
            video = load_frames(image_paths)
            video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
            videos.append(video)

        conv.append_message(conv.roles[0], full_question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        image_paths = [f'/home/ubuntu/workspace/datasets/ipad/{name}/' + x for x in os.listdir(f'/home/ubuntu/workspace/datasets/ipad/{name}')]
        image_paths = [image_paths[i] for i in range(0, len(image_paths), 25)]
        video = load_frames(image_paths)
        video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
        videos.append(video)
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
        generated_tokens = cont.sequences
        text_outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip().lower()
        score = cont.scores[0]
        probs = F.softmax(score, dim=-1)
        prob = torch.max(probs[0]).item()
        if prob > 0.9:
            max_text_outputs = text_outputs
            max_confi = prob
            break
        else:
            if prob > max_confi:
                max_confi = prob
                max_text_outputs = text_outputs

    answer = 'yes' if label == 'anomaly' else 'no'
    
    if answer in max_text_outputs:
        acc.append(1)
    else:
        acc.append(0)
    print(answer, max_text_outputs)
    print('total acc', round(sum(acc)/len(acc), 5), [False, True][acc[-1]], round(max_confi, 3), 'try', num, 'shot', len(videos)-1)
    
    pbar.update(1)
    del cont, probs, generated_tokens, score, input_ids
    torch.cuda.empty_cache()

pbar.close()
