from PIL import Image
import numpy as np
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor

import json
import re
import os
import torch
import gc  # For memory management

from tqdm import tqdm

def extract_answer(text):
    # Improved answer extraction for the specific format we're using
    # Look for "The cause event is:" format first
    cause_match = re.search(r'The cause event is:\s*(.+?)(?=\.|$)', text, re.IGNORECASE)
    if cause_match:
        return cause_match.group(1).strip().lower()
    
    # Fallback to original ASSISTANT format
    assistant_match = re.findall(r'ASSISTANT:\s*(.+?)(?=\.\s|$)', text)
    if assistant_match:
        return assistant_match[-1].strip(".").lower()
    
    # Additional patterns for better extraction
    patterns = [
        r'(?:answer|response):\s*(.+?)(?=\.|$)',
        r'(?:^|\n)(.+?)(?:during\s*\([^)]+\))',
        r'(?:cause|caused by|reason).*?:\s*(.+?)(?=\.|$)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip().lower()
    
    # If no pattern matches, return the entire text (cleaned)
    return text.strip().lower()

def evaluate_answer(predicted_answer, correct_answer):
    """
    Improved evaluation function that checks for partial matches
    and handles different answer formats
    """
    predicted_lower = predicted_answer.lower()
    correct_lower = correct_answer.lower()
    
    # Direct substring match
    if correct_lower in predicted_lower:
        return True
    
    # Extract key action words from both answers
    predicted_words = set(re.findall(r'\b\w+\b', predicted_lower))
    correct_words = set(re.findall(r'\b\w+\b', correct_lower))
    
    # Check for significant overlap (at least 80% of correct answer words)
    if len(correct_words) > 0:
        overlap = len(predicted_words.intersection(correct_words))
        overlap_ratio = overlap / len(correct_words)
        if overlap_ratio >= 0.8:
            return True
    
    return False

# Initialize LLaVA-NeXT-Video model and processor with optimizations
model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    "llava-hf/LLaVA-NeXT-Video-7B-hf", 
    torch_dtype=torch.float16, 
    device_map="auto",
    attn_implementation="flash_attention_2",  # Use flash attention for better memory efficiency
    low_cpu_mem_usage=True  # Reduce CPU memory usage during loading
)
processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")

correct_answers = 0
with open("../abductiveEk100/post_processed_test_data_instances.json") as f:
    data = json.load(f)
    for instance in tqdm(data, total=len(data)):
        print("-"*50)
        # create in-context prompt:
        conversation = []
        
        # Add a single system message at the beginning
        conversation.append({
            "role": "system",
            "content": [
                {"type": "text", "text": "You are an expert in video analysis and causal reasoning. Your task is to identify the cause event of a target event by analyzing the sequence of events in the video. Always choose your answer from the given premise events."}
            ]
        })
        
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

                # Add user question and assistant answer as in-context examples
                conversation.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Based on the video, analyze these premise events: {event_list}. What is the cause event of '{target_event}'? Choose the correct answer from the premise events."}
                    ]
                })
                conversation.append({
                    "role": "assistant", 
                    "content": [
                        {"type": "text", "text": f"The cause event is: {explaination_event}"}
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
        
        # Add the current question with video
        current_question = f"Based on the video, analyze these premise events: {event_list}. What is the cause event of '{target_event}'? Choose the correct answer from the premise events and respond in the format: 'The cause event is: [your answer]'."
        
        # Add current user message with video path (LLaVA-NeXT-Video uses video path directly)
        conversation.append({
            "role": "user", 
            "content": [
                {"type": "text", "text": current_question},
                {"type": "video", "path": video_path}
            ]
        })

        # Generate response using LLaVA-NeXT-Video workflow with optimized parameters
        # Use more frames for better temporal understanding of causal relationships
        inputs = processor.apply_chat_template(conversation, num_frames=16, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        
        # Move inputs to model device
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Generate response with optimized parameters for better reasoning
        out = model.generate(
            **inputs, 
            max_new_tokens=256,  # Allow more tokens for detailed reasoning
            temperature=0.1,      # Lower temperature for more focused responses
            do_sample=True,       # Enable sampling for diversity
            top_p=0.9,           # Nucleus sampling for balanced creativity and accuracy
            repetition_penalty=1.1,  # Prevent repetitive responses
            pad_token_id=processor.tokenizer.eos_token_id  # Proper padding
        )
        
        # Decode the response
        output_text = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        # Extract the generated text (remove the input prompt part)
        if output_text:
            full_response = output_text[0]
            # Find the assistant's response part
            if "ASSISTANT:" in full_response:
                answer = full_response.split("ASSISTANT:")[-1].strip()
            else:
                answer = full_response
        else:
            answer = ""
            
        print("Full response:", full_response if output_text else "No response")
        print("Extracted answer:", answer)
        print("Expected answer:", explaination_event_for_val)
        
        # Use improved answer extraction and evaluation
        extracted_answer = extract_answer(answer)
        is_correct = evaluate_answer(extracted_answer, explaination_event_for_val)
        
        print("Processed answer:", extracted_answer)
        print("Evaluation result:", "CORRECT" if is_correct else "INCORRECT")
        
        if is_correct:
            correct_answers += 1
            
        # Clear GPU memory after each inference to prevent OOM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
            
print("Accuracy:", correct_answers/len(data))
