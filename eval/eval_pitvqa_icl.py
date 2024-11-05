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
import random
random.seed(1)

number_dict = {'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6', 'seven': '7',
            'eight': '8', 'nine': '9', 'ten': '10', 'once': '1', 'twice': '2'}

def load_frames(image_paths):
    images = [np.array(Image.open(x)) for x in image_paths]
    images = np.stack(images, axis=0)
    return images

def put_examples(conv, examples):
    videos = []
    for ex in examples:
        question = DEFAULT_IMAGE_TOKEN + "\n" + ex['question'] + " Answer in simple words or one sentence."
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], ex['answer'])
        video_path = ex['video']
        video = load_frames(video_path)
        video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
        videos.append(video)
    return conv, videos
        

pretrained = "checkpoints/LLaVA-Video-7B-Qwen2"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"

tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map)  # Add any other thing you want to pass in llava_model_args
model.eval()

with open(f'/home/ubuntu/workspace/datasets/PitVQA/pitvqa_video_test_sim.json', 'r') as f:
    jf = json.load(f)
with open(f'/home/ubuntu/workspace/datasets/PitVQA/pitvqa_video_train.json', 'r') as f:
    train_jf = json.load(f)
    train_data = {}
    for d in train_jf:
        train_data[d['ann_id']] = d

acc = []
shot = 4
cnt = 0
for data in tqdm(jf):
    if cnt < 1150:
        cnt += 1
        continue
    cnt += 1
    video_path = data['video']
    video = load_frames(video_path)
    video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
    candidates = data['train_examples']
    max_text_outputs = None
    max_confi = 0
    for num in range(4): 
        examples = candidates[:shot]
        examples = [train_data[x['train_question_id']] for x in examples]
        candidates = candidates[shot:]
        # examples = random.sample(train_jf, 2)
        conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
        question = DEFAULT_IMAGE_TOKEN + "\n" + data['question'] + " Answer in simple words or one sentence."
        conv = copy.deepcopy(conv_templates[conv_template])
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
                modalities=["video"] * len(videos),
                do_sample=False,
                temperature=0,
                max_new_tokens=100,
                output_scores=True,
                return_dict_in_generate=True
            )
        del input_ids, videos
        torch.cuda.empty_cache()
        text_outputs = tokenizer.batch_decode(cont.sequences, skip_special_tokens=True)[0].strip().lower()
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
        del cont, probs, scores
        torch.cuda.empty_cache()

    answer = data['answer'].lower()

    for k, v in number_dict.items():
        if k in max_text_outputs:
            max_text_outputs = max_text_outputs.replace(k, v)
        if k in answer:
            answer = answer.replace(k, v)

    if answer in max_text_outputs:
        acc.append(1)
    else:
        acc.append(0)
    print(answer, '@@@', max_text_outputs)
    print('total acc', round(sum(acc)/len(acc), 5), [False, True][acc[-1]], round(max_confi, 3), 'try', num, 'shot', shot, 'llava pitvqa')