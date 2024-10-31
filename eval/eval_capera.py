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
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider

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

test_vids = []
with open(f'/home/ubuntu/workspace/datasets/CapERA/CapERA_DATASET_test.json', 'r') as f:
    test_jf = json.load(f)['ERA_caption']

question_sample = "Provide a concise depiction of this video."
history = []
for data in tqdm(test_jf):
    video_id = data['video_id']
    folder = video_id.split('_')[0]
    video_path = f"/home/ubuntu/workspace/datasets/CapERA/Videos/Test/{folder}/{video_id}"
    max_frames_num = 32
    video,frame_time,video_time = load_video(video_path, max_frames_num, 1, force_sample=False)
    video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
    video = [video]
    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
    question = DEFAULT_IMAGE_TOKEN + f"\n{question_sample}"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

    cont = model.generate(
        input_ids,
        images=video,
        modalities="video",
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
    )
    pred = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
    gold_captions = data['annotation']['English_caption']

    references = gold_captions
    candidate = pred

    elem = {
        "video_id": video_id,
        "prediction": pred,
        "ground_truth": gold_captions,
    }
    history.append(elem)

with open('capera_baseline.json', 'w') as f:
    json.dump(history, f)

    # references_tokenized = [ref.split() for ref in references]
    # smoothie = SmoothingFunction().method4
    # bleu1 = sentence_bleu(references_tokenized, candidate.split(), weights=(1, 0, 0, 0), smoothing_function=smoothie)
    # bleu2 = sentence_bleu(references_tokenized, candidate.split(), weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
    # bleu3 = sentence_bleu(references_tokenized, candidate.split(), weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
    # bleu4 = sentence_bleu(references_tokenized, candidate.split(), weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

    # meteor_scorer = Meteor()
    # _, meteor_score = meteor_scorer.compute_score({0: references}, {0: [candidate]})
    # rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    # rouge_score = rouge_scorer.score(' '.join(references), candidate)
    # rougeL = rouge_score['rougeL'].fmeasure


    # cider_scorer = Cider()
    # _, cider_score = cider_scorer.compute_score({0: references}, {0: [candidate]})

    # print(f"BLEU-1: {bleu1}")
    # print(f"BLEU-2: {bleu2}")
    # print(f"BLEU-3: {bleu3}")
    # print(f"BLEU-4: {bleu4}")
    # print(f"METEOR: {meteor_score}")
    # print(f"ROUGE-L: {rougeL}")
    # print(f"CIDEr: {cider_score}")
    # elem = {
    #     "prediction": pred,
    #     "ground_truth": gold_captions,
    #     "BLEU-1": bleu1,
    #     "BLEU-2": bleu2,
    #     "BLEU-3": bleu3,
    #     "BLEU-4": bleu4,
    #     "METEOR": meteor_score[0],
    #     "ROUGE-L": rougeL,
    #     "CIDEr": cider_score[0]
    # }