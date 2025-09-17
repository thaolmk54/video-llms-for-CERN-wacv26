from PIL import Image
import numpy as np
import sys
sys.path.append('./')
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init

import json
import re
import os
import torch

from tqdm import tqdm

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

# Initialize Video LLama2 
disable_torch_init()
model_path = 'DAMO-NLP-SG/VideoLLaMA2-7B'
model, processor, tokenizer = model_init(model_path)

correct_answers = 0
with open("../abductiveEk100/post_processed_test_data_instances.json") as f:
    data = json.load(f)
    for instance in tqdm(data, total=len(data)):
        print("-"*50)
        # create in-context prompt:
        in_context_examples = []
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

                # Build text-based in-context example
                example_text = f"Question: Given a sequence of premise events include {event_list}, what is the cause event of the event {target_event}? Choose a correct answer within the premise events.\nAnswer: {explaination_event}\n\n"
                in_context_examples.append(example_text)

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
        
        # Build the complete instruction with in-context examples
        in_context_text = "".join(in_context_examples)
        current_question = f"Given a sequence of premise events include {event_list}, what is the cause event of the event {target_event}? Choose a correct answer within the premise events."
        
        # Combine in-context examples with current question
        full_instruction = f"{in_context_text}Question: {current_question}\nAnswer: "
        
        print("Full instruction:", full_instruction)
        
        # Generate response using Video LLama2
        modal = 'video'
        output = mm_infer(
            processor[modal](video_path), 
            full_instruction, 
            model=model, 
            tokenizer=tokenizer, 
            do_sample=False, 
            modal=modal
        )
        
        answer = output if output else ""
        print("explaination_event_for_val:", explaination_event_for_val)
        answer_for_val = answer
        # answer_for_val = extract_answer(answer)
        print("answer_for_val:", answer_for_val)
        
        if explaination_event_for_val in answer_for_val:
            print(f"{answer_for_val} is correct")
            correct_answers += 1
            
print("Accuracy:", correct_answers/len(data))
