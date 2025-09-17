import os
import av
import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, BitsAndBytesConfig, VideoLlavaForConditionalGeneration
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from torch.optim import AdamW

MAX_LENGTH = 256
MODEL_ID = "LanguageBind/Video-LLaVA-7B-hf"
REPO_ID = "thaolmk54/VideoLLava-ek100"
USE_LORA = False
USE_QLORA = True

def read_video_pyav(container, indices):
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

processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "right"

class VideoLlavaDataset(Dataset):
    def __init__(self, dataset, mode="train"):
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

def collate_fn(examples):
    texts, videos = zip(*examples)
    batch = processor(text=texts, videos=videos, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels
    return batch["input_ids"], batch["attention_mask"], batch["pixel_values_videos"], labels

def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for input_ids, attention_mask, pixel_values, labels in dataloader:
        input_ids, attention_mask, pixel_values, labels = input_ids.to(device), attention_mask.to(device), pixel_values.to(device), labels.to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values_videos=pixel_values, labels=labels)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate(model, dataloader, device):
    model.eval()
    total_correct = 0
    with torch.no_grad():
        for input_ids, attention_mask, pixel_values, answers in dataloader:
            input_ids, attention_mask, pixel_values = input_ids.to(device), attention_mask.to(device), pixel_values.to(device)
            generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, pixel_values_videos=pixel_values, max_new_tokens=MAX_LENGTH, do_sample=False)
            predictions = processor.batch_decode(generated_ids[:, input_ids.size(1):], skip_special_tokens=True)
            total_correct += sum([pred.strip().lower() == answer.lower() for pred, answer in zip(predictions, answers)])
    return total_correct / len(dataloader)

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector', 'vision_model']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open("abductiveEk100/post_processed_train_data_instances.json") as f:
        train_data = json.load(f)
    with open("abductiveEk100/post_processed_test_data_instances.json") as f:
        test_data = json.load(f)

    train_dataset = VideoLlavaDataset(train_data[:50], mode="train")
    test_dataset = VideoLlavaDataset(test_data, mode="test")
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=1)

    if USE_QLORA or USE_LORA:
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16) if USE_QLORA else None
        model = VideoLlavaForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float16, quantization_config=bnb_config, device_map="auto")
        lora_config = LoraConfig(r=8, lora_alpha=8, lora_dropout=0.1, target_modules=find_all_linear_names(model), init_lora_weights="gaussian")
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
    else:
        model = VideoLlavaForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    epochs = 2
    for epoch in range(epochs):
        train_loss = train(model, train_dataloader, optimizer, device)
        val_accuracy = validate(model, test_dataloader, device)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

if __name__ == "__main__":
    main()