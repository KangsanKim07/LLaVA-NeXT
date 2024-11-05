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

def put_examples(conv, examples):
    videos = []
    for ex in examples:
        video_path = f"/home/ubuntu/workspace/datasets/CapERA/Videos/{ex}"
        try:
            video,frame_time,video_time = load_video(video_path, max_frames_num, 1, force_sample=False)
        except:
            print(video_path)
            continue
        video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
        videos.append(video)
        gt_cap = train_captions[ex.split('/')[-1]]
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + f"\nProvide a concise depiction of this video.")
        conv.append_message(conv.roles[1], gt_cap)
    
    return conv, videos


pretrained = "checkpoints/LLaVA-Video-7B-Qwen2"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"

tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map)  # Add any other thing you want to pass in llava_model_args
model.eval()

test_vids = []
with open(f'/home/ubuntu/workspace/datasets/CapERA/CapERA_DATASET_test.json', 'r') as f:
    test_jf = json.load(f)['ERA_caption']
with open(f'/home/ubuntu/workspace/datasets/CapERA/CapERA_DATASET_train.json', 'r') as f:
    train_jf = json.load(f)['ERA_caption']
    train_captions = {}
    for i in train_jf:
        train_captions[i['video_id']] = i['annotation']['English_caption'][0]

with open(f"/home/ubuntu/workspace/datasets/CapERA/video_top400_sim_subset.json", 'r') as f:
    jf = json.load(f)
    vid_sim = {}
    for i in jf:
        vid_sim[i['test_sample']] = i['train_examples']

question_sample = "Provide a concise depiction of this video."
history = []
max_frames_num = 32
shot = 8
for data in tqdm(test_jf):
    ground_truth = data['annotation']['English_caption']
    video_id = data['video_id']
    folder = video_id.split('_')[0]
    video_path = f"/home/ubuntu/workspace/datasets/CapERA/Videos/Test/{folder}/{video_id}"
    video,frame_time,video_time = load_video(video_path, max_frames_num, 1, force_sample=False)
    video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
    examples = vid_sim[f"Test/{folder}/{video_id}"]
    max_text_outputs = None
    max_confi = 0
    for num in range(4):
        conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
        question = DEFAULT_IMAGE_TOKEN + f"\n{question_sample}"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv, videos = put_examples(conv, examples[:shot])
        examples = examples[shot:]
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
                max_new_tokens=1024,
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
        del cont, generated_tokens, scores
        torch.cuda.empty_cache()
    gold_captions = data['annotation']['English_caption']

    references = gold_captions
    candidate = max_text_outputs

    print('num', num, 'shot', shot)
    print('pred:', max_text_outputs)
    print('gold:', gold_captions[0])

    elem = {
        "video_id": data['video_id'],
        "prediction": max_text_outputs,
        "ground_truth": gold_captions,
    }
    history.append(elem)

with open(f'/home/ubuntu/workspace/datasets/CapERA/capera_{shot}shoticl_tokenprob.json', 'w') as f:
    json.dump(history, f)