from PIL import Image
import requests
import numpy as np
import av
from huggingface_hub import hf_hub_download
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration

import json
import re

from tqdm import tqdm

def extract_answer(text):
    # Regular expression to match the part after "is" until the end of the sentence.
    match = re.search(r'ASSISTANT:.*?is (.+?)(?=\.\s|$)', text)

    if match:
        answer = match.group(1).strip(".").lower()
        # print(answer)
    else:
        print("Answer not found.")
        answer = ""
        
    return answer

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.

    Args:
        container (av.container.input.InputContainer): PyAV container.
        indices (List[int]): List of frame indices to decode.

    Returns:
        np.ndarray: np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")
processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")

# video_path = hf_hub_download(repo_id="raushan-testing-hf/videos-test", filename="sample_demo_1.mp4", repo_type="dataset")
# Change to EK100 video paths and run some test

# read data from abductiveEk100/post_processed_test_data_instances.json and extract the video path
correct_answers = 0
with open("abductiveEk100/post_processed_test_data_instances.json") as f:
    data = json.load(f)
    for instance in tqdm(data, total=len(data)):
        print("-"*50)
        video_id = instance["video_id"]
        video_id_parts = video_id.split("_")
        video_path = f"/home-old/Projects/visual_reasoning/datasets/cv-focus-datasets/epic100_raw_download/epic55/videos/test/{video_id_parts[0]}/{video_id_parts[0]}_{video_id_parts[1]}.MP4"
        
        target_event_id = int(instance["target_event"].split("_")[-1])
        explaination_event_id = int(instance["explanation_event"].split("_")[-1])
        print("event_list:", instance["event_list"])
        print("target_event id:", target_event_id)
        print("explaination_event id:", explaination_event_id)
        
        frame_list = instance["frame_list"]
        start_frame = int(instance["event_list"][0][1])
        end_frame = int(instance["event_list"][target_event_id][2])
        
        event_list = [f"{event[-1]} during ({event[1]}, {event[2]})" for event in instance["event_list"][:target_event_id+1]]
        event_list = ", ".join(event_list)
        
        target_event = f"{instance["event_list"][target_event_id][-1]} during ({instance["event_list"][target_event_id][1]}, {instance["event_list"][target_event_id][2]})"
        explaination_event = f"{instance["event_list"][explaination_event_id][-1]} during ({instance["event_list"][explaination_event_id][1]}, {instance["event_list"][explaination_event_id][2]})"
        explaination_event_for_val = f"{instance["event_list"][explaination_event_id][-1]}"

        print("video_path:", video_path)
        print("event_list:", event_list)
        print("start_frame:", start_frame)
        print("end_frame:", end_frame)
        print("target_event:", target_event)
        print("explaination_event:", explaination_event)
        container = av.open(video_path)

        prompt = f"USER: <video>Given a sequence of premise events include {event_list}, what is the cause event of the event {target_event}? Choose a correct answer within the premise events. ASSISTANT:"
        # sample uniformly 8 frames from the video
        # total_frames = container.streams.video[0].frames
        indices = np.arange(start_frame, end_frame, end_frame / 8).astype(int)
        print("indices:", indices)
        clip = read_video_pyav(container, indices)

        inputs = processor(text=prompt, videos=clip, return_tensors="pt")

        # Generate
        generate_ids = model.generate(**inputs, max_length=512)
        answer = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print(answer)
        answer_for_val = extract_answer(answer)
        print("answer_for_val:", answer_for_val)
        # check if the answer is correct
        if explaination_event_for_val in answer_for_val:
            print(f"{answer_for_val} is correct")
            correct_answers += 1
        
            
print("Accuracy:", correct_answers/len(data))
### Accuracy: 0.04263565891472868