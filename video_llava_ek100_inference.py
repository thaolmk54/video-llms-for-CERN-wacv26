from transformers import AutoProcessor, BitsAndBytesConfig, VideoLlavaForConditionalGeneration

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import os
import av
import numpy as np

import json

MAX_LENGTH = 256
MODEL_ID = "LanguageBind/Video-LLaVA-7B-hf"
REPO_ID = "thaolmk54/VideoLLava-ek100" # Change to your hf-hub repo

class VideoLlavaDataset(Dataset):
     """
     PyTorch Dataset for VideoLlavaDataset. This class 
     """
     def __init__(self, dataset, mode="train"):
          super().__init__()
          self.dataset = dataset 
          self.mode = mode
     
     def __len__(self):
          return len(self.dataset)
     
     def __getitem__(self, idx: int):
        instance = self.dataset[idx]
        
        video_id = instance["video_id"]
        video_id_parts = video_id.split("_")
        if self.mode == "train":
            video_path = f"/weka/Projects/visual_reasoning/datasets/epic100_raw_download/epic55/videos/train/{video_id_parts[0]}/{video_id_parts[0]}_{video_id_parts[1]}.MP4"
            if not os.path.exists(video_path):
                video_path = f"/weka/Projects/visual_reasoning/datasets/epic100_raw_download/epic55/videos/test/{video_id_parts[0]}/{video_id_parts[0]}_{video_id_parts[1]}.MP4"
            if not os.path.exists(video_path):
                video_path = f"/weka/Projects/visual_reasoning/datasets/epic100_raw_download/extended_epic100/{video_id_parts[0]}/{video_id_parts[0]}_{video_id_parts[1]}.MP4"
        elif self.mode == "test":
            video_path = f"/weka/Projects/visual_reasoning/datasets/epic100_raw_download/epic55/videos/test/{video_id_parts[0]}/{video_id_parts[0]}_{video_id_parts[1]}.MP4"
        else:
            raise ValueError("Invalid mode. Choose either 'train' or 'test'")
        
        target_event_id = int(instance["target_event"].split("_")[-1])
        explaination_event_id = int(instance["explanation_event"].split("_")[-1])
        
        start_frame = int(instance["event_list"][0][1])
        end_frame = int(instance["event_list"][target_event_id][2])
        
        indices = np.arange(start_frame, end_frame, end_frame / 8).astype(int)
        container = av.open(video_path)
        clip = read_video_pyav(container, indices)
        
        event_list = [f"{event[-1]} during ({event[1]}, {event[2]})" for event in instance["event_list"][:target_event_id+1]]
        # event_list = ", ".join(event_list)
        
        target_event = f"{instance["event_list"][target_event_id][-1]} during ({instance["event_list"][target_event_id][1]}, {instance["event_list"][target_event_id][2]})"
        explaination_event = f"{instance["event_list"][explaination_event_id][-1]} during ({instance["event_list"][explaination_event_id][1]}, {instance["event_list"][explaination_event_id][2]})"
        explaination_event_for_val = f"{instance["event_list"][explaination_event_id][-1]}"

        mult_choice = ""
        for i, choice in enumerate(event_list):
            mult_choice += f"{choice}; "

        # Prepare a prompt template, can be changed depeding on the dataset and use-cases
        prompt = f"USER: <video>\nAnswer the following multiple choice question based on the video. " \
                f"Question: what is the trigger event of the {target_event}\n {mult_choice}\n ASSISTANT: Answer: {explaination_event}"

        return prompt, clip
   
processor = AutoProcessor.from_pretrained(MODEL_ID)
# Define quantization config
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Load the base model with adapters on top
model = VideoLlavaForConditionalGeneration.from_pretrained(
    REPO_ID,
    torch_dtype=torch.float16,
    # quantization_config=quantization_config,
    device_map="auto",
    local_files_only = True
)

with open("abductiveEk100/post_processed_test_data_instances.json") as f:
        test_data = json.load(f)
        
test_dataset = VideoLlavaDataset(test_data, mode="test")

prompt, clip = test_dataset[1]

answer = prompt[-1]
prompt = prompt[:-2]

inputs = processor(text=prompt, videos=clip, return_tensors="pt").to(model.device)
for k,v in inputs.items():
    print(k,v.shape)
    
# Generate token IDs
generated_ids = model.generate(**inputs, max_new_tokens=MAX_LENGTH)

# Decode back into text
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)