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

number_dict = {'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6', 'seven': '7',
            'eight': '8', 'nine': '9', 'ten': '10', 'once': '1', 'twice': '2'}

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

with open(f'/home/ubuntu/workspace/datasets/PitVQA/pitvqa_video_test.json', 'r') as f:
    jf = json.load(f)

score = []
for data in tqdm(jf):
    video_path = data['video']
    video = load_frames(video_path)
    video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
    video = [video]

    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
    question = DEFAULT_IMAGE_TOKEN + "\n" + data['question'] + " Answer in simple words or one sentence."
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    cont = model.generate(
        input_ids,
        images=video,
        modalities=["video"],
        do_sample=False,
        temperature=0,
        max_new_tokens=100,
    )
    output_text = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip().lower()
    answer = data['answer'].lower()

    for k, v in number_dict.items():
        if k in output_text:
            output_text = output_text.replace(k, v)
        if k in answer:
            answer = answer.replace(k, v)

    if answer in output_text:
        score.append(1)
    else:
        score.append(0)
    print(answer, '@@@', output_text)
    print(round(sum(score)/len(score), 5), "llava video 7B baseline")