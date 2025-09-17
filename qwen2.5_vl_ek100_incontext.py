from PIL import Image
import numpy as np
import av

import json
import re
import os
import time
import psutil
import gc

from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

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

# Initialize Qwen2.5-VL model and processor with optimizations
print("Loading model and processor...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", 
    torch_dtype=torch.float16,  # Use float16 for better performance
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True  # Reduce CPU memory usage during loading
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", trust_remote_code=True)

# Load data once
print("Loading test data...")
with open("../abductiveEk100/post_processed_test_data_instances.json") as f:
    test_data = json.load(f)

print("Loading train data for in-context examples...")
with open("../abductiveEk100/post_processed_train_data_instances.json") as f:
    train_data = json.load(f)

# Pre-select in-context examples once instead of random selection each time
print("Pre-selecting in-context examples...")
np.random.seed(42)  # For reproducibility
selected_incontext_examples = np.random.choice(train_data, min(10, len(train_data)), replace=False)

# Performance monitoring setup
start_time = time.time()
process = psutil.Process(os.getpid()) if 'psutil' in globals() else None

correct_answers = 0
data = test_data
total_inference_time = 0
total_video_loading_time = 0
for instance in tqdm(test_data, total=len(test_data), desc="Processing instances"):
    instance_start_time = time.time()
    print("-"*50)
    
    # Create conversation with system message for better guidance
    conversation = []
    
    # Add a system message at the beginning for better task understanding
    conversation.append({
        "role": "system",
        "content": [
            {"type": "text", "text": "You are an expert in video analysis and causal reasoning. Your task is to identify the cause event of a target event by analyzing the sequence of events in the video. Always choose your answer from the given premise events."}
        ]
    })
    
    # Use pre-selected in-context examples (more efficient)
    picked_instances = np.random.choice(selected_incontext_examples, 2, replace=False)
    
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
    
    # Optimized video loading with error handling
    video_start_time = time.time()
    try:
        container = av.open(video_path)
    except Exception as e:
        print(f"Error opening video {video_path}: {e}")
        continue
    
    # Add the current question with video
    current_question = f"Based on the video, analyze these premise events: {event_list}. What is the cause event of '{target_event}'? Choose the correct answer from the premise events and respond in the format: 'The cause event is: [your answer]'."
    
    # Use more frames for better temporal understanding (increased from 8 to 16)
    num_frames = 16  # More frames for better causal relationship understanding
    indices = np.linspace(start_frame, end_frame-1, num_frames, dtype=int)
    print("indices:", indices)
    
    try:
        clip = read_video_pyav(container, indices)
        container.close()  # Close container to free memory
        video_loading_time = time.time() - video_start_time
        total_video_loading_time += video_loading_time
        print(f"Video loading time: {video_loading_time:.2f}s")
    except Exception as e:
        print(f"Error reading video frames: {e}")
        container.close()
        continue
    
    # Add current user message with video frames
    conversation.append({
        "role": "user", 
        "content": [
            {"type": "video", "video": [Image.fromarray(frame) for frame in clip]},
            {"type": "text", "text": current_question}
        ]
    })

    try:
        # Generate response using Qwen2.5-VL workflow
        inference_start_time = time.time()
        
        text = processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(conversation)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        # Inference: Generation with optimized parameters for better reasoning
        with torch.no_grad():  # Disable gradient computation for inference
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=256,      # Allow more tokens for detailed reasoning
                temperature=0.1,         # Lower temperature for more focused responses
                do_sample=True,          # Enable sampling for diversity
                top_p=0.9,              # Nucleus sampling for balanced creativity and accuracy
                repetition_penalty=1.1,  # Prevent repetitive responses
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        
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
        
        inference_time = time.time() - inference_start_time
        total_inference_time += inference_time
        print(f"Inference time: {inference_time:.2f}s")
        
        # Clear GPU cache to prevent memory accumulation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Explicit garbage collection
        del inputs, generated_ids, generated_ids_trimmed
        gc.collect()
            
    except Exception as e:
        print(f"Error during inference: {e}")
        answer = ""
        full_response = ""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        continue
    
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
    
    instance_time = time.time() - instance_start_time
    print(f"Total instance time: {instance_time:.2f}s")
    
    # Memory monitoring (if available)
    if process:
        memory_info = process.memory_info()
        print(f"Current memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

# Final performance summary
total_time = time.time() - start_time
print("\n" + "="*60)
print("PERFORMANCE SUMMARY")
print("="*60)
print(f"Total processing time: {total_time:.2f}s")
print(f"Average time per instance: {total_time/len(test_data):.2f}s")
print(f"Total video loading time: {total_video_loading_time:.2f}s")
print(f"Total inference time: {total_inference_time:.2f}s")
print(f"Average video loading time: {total_video_loading_time/len(test_data):.2f}s")
print(f"Average inference time: {total_inference_time/len(test_data):.2f}s")
print(f"Accuracy: {correct_answers/len(test_data):.4f}")
print(f"Processed {len(test_data)} instances")

if process:
    final_memory = process.memory_info()
    print(f"Final memory usage: {final_memory.rss / 1024 / 1024:.2f} MB")
print("="*60)
