from PIL import Image
import requests
import numpy as np
import av
from huggingface_hub import hf_hub_download
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration

import json
import re
import os

from tqdm import tqdm
# from modelscope import AutoModelForCausalLM, AutoTokenizer
# from modelscope import GenerationConfig
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

def extract_answer(text):
    # Regular expression to match the part after "is" until the end of the sentence.
    match = re.findall(r'ASSISTANT:\s*(.+?)(?=\.\s|$)', text)

    if match:
        answer = match[-1].strip(".").lower()
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

# Initialize Qwen2.5-VL model and processor
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

correct_answers = 0
with open("../abductiveEk100/post_processed_test_data_instances.json") as f:
    data = json.load(f)
    for instance in tqdm(data, total=len(data)):
        print("-"*50)
        # create in-context prompt:
        in_context_messages = []
        with open("../abductiveEk100/post_processed_train_data_instances.json") as f_ict:
            data_train = json.load(f_ict)
            picked_instances = np.random.choice(data_train, 2, replace=False)
            for instance_ict in picked_instances:
                print("Incontext prompt")
                
                target_event_id = int(instance_ict["target_event"].split("_")[-1])
                explaination_event_id = int(instance_ict["explanation_event"].split("_")[-1])
                print("event_list:", instance_ict["event_list"])
                print("target_event id:", target_event_id)
                print("explaination_event id:", explaination_event_id)
                
                frame_list = instance_ict["frame_list"]
                start_frame = int(instance_ict["event_list"][0][1])
                end_frame = int(instance_ict["event_list"][target_event_id][2])
                
                event_list = [f"{event[-1]} during ({event[1]}, {event[2]})" for event in instance_ict["event_list"][:target_event_id+1]]
                event_list = ", ".join(event_list)
                
                target_event = f"{instance_ict["event_list"][target_event_id][-1]} during ({instance_ict["event_list"][target_event_id][1]}, {instance_ict["event_list"][target_event_id][2]})"
                explaination_event = f"{instance_ict["event_list"][explaination_event_id][-1]} during ({instance_ict["event_list"][explaination_event_id][1]}, {instance_ict["event_list"][explaination_event_id][2]})"
                
                print("event_list:", event_list)
                print("start_frame:", start_frame)
                print("end_frame:", end_frame)
                print("target_event:", target_event)
                print("explaination_event:", explaination_event)

                # Add user question and assistant answer as separate messages
                in_context_messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Given a sequence of premise events include {event_list}, what is the cause event of the event {target_event}? Choose a correct answer within the premise events."}
                    ]
                })
                in_context_messages.append({
                    "role": "assistant", 
                    "content": [
                        {"type": "text", "text": explaination_event}
                    ]
                })

        video_id = instance["video_id"]
        video_id_parts = video_id.split("_")
        video_path = f"../epic100_raw_download/epic55/videos/test/{video_id_parts[0]}/{video_id_parts[0]}_{video_id_parts[1]}.MP4"
        
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
        container = av.open(video_path)
        
        # Create the complete messages list including in-context examples and the current question
        messages = in_context_messages.copy()
        
        # Add the current question with video
        current_question = f"Given a sequence of premise events include {event_list}, what is the cause event of the event {target_event}? Choose a correct answer within the premise events."
        
        indices = np.arange(start_frame, end_frame, end_frame / 8).astype(int)
        print("indices:", indices)
        clip = read_video_pyav(container, indices)
        
        # Add current user message with video frames
        messages.append({
            "role": "user", 
            "content": [
                {"type": "video", "video": [Image.fromarray(frame) for frame in clip]},
                {"type": "text", "text": current_question}
            ]
        })

        # Generate response using Qwen2.5-VL workflow
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        answer = output_text[0] if output_text else ""
        print("explaination_event_for_val:", explaination_event_for_val)
        answer_for_val = answer
        # answer_for_val = extract_answer(answer)
        print("answer_for_val:", answer_for_val)
        
        if explaination_event_for_val in answer_for_val:
            print(f"{answer_for_val} is correct")
            correct_answers += 1
            
print("Accuracy:", correct_answers/len(data))
