import os
import av
import re
import bisect
import shutil
import numpy as np

from transformers import AutoProcessor
from transformers import BitsAndBytesConfig, VideoLlavaForConditionalGeneration
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# from huggingface_hub import snapshot_download
# from datasets import load_dataset, concatenate_datasets

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from lightning.pytorch.callbacks import Callback

import json

from pytorch_lightning.callbacks import ModelCheckpoint

from huggingface_hub import HfApi
api = HfApi()

# from huggingface_hub import create_repo
# create_repo("thaolmk54/VideoLLava-ek100")

MAX_LENGTH = 256
MODEL_ID = "LanguageBind/Video-LLaVA-7B-hf"
REPO_ID = "thaolmk54/VideoLLava-ek100" # Change to your hf-hub repo

USE_LORA = False
USE_QLORA = True

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

processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "right" # during training, one always uses padding on the right

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
   
def train_collate_fn(examples):
    videos = []
    texts = []
    texts, videos = list(zip(*examples))

    batch = processor(text=texts, videos=videos, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values_videos = batch["pixel_values_videos"]
    labels = batch["labels"]

    return input_ids, attention_mask, pixel_values_videos, labels


def eval_collate_fn(examples):
    # We only feed the prompt to the model
    videos = []
    texts = []
    texts, videos = list(zip(*examples))
    texts = [text[:-2] for text in texts]

    batch = processor(text=texts, videos=videos, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values_videos = batch["pixel_values_videos"]
    answer_choice = [texts[-1] for text in texts]

    return input_ids, attention_mask, pixel_values_videos, answer_choice

class VideoLlavaModelPLModule(L.LightningModule):
    def __init__(self, config, processor, model):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model

        self.batch_size = config.get("batch_size")

    def training_step(self, batch, batch_idx):

        input_ids, attention_mask, pixel_values_videos, labels = batch

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values_videos=pixel_values_videos,
            labels=labels
        )
        loss = outputs.loss

        self.log("train_loss", loss)
        print("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):

        input_ids, attention_mask, pixel_values_videos, answers = batch

        # autoregressively generate token IDs
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values_videos=pixel_values_videos,
            max_new_tokens=MAX_LENGTH,
            do_sample=False,
        )
        # turn them back into text, chopping of the prompt
        predictions = self.processor.batch_decode(generated_ids[:, input_ids.size(1):], skip_special_tokens=True)

        correct = 0
        for pred, answer in zip(predictions, answers):
            print("pred:", pred)
            print("answer:", answer)
            answer_for_val = extract_answer(pred)
            correct += (answer_for_val in answer.lower())
        self.log("val_accuracy", correct / len(answers))
        print("val_accuracy:", correct / len(answers))

        return correct

    def configure_optimizers(self):
        # you could also add a learning rate scheduler if you want
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.get("lr"))

        return optimizer

    def train_dataloader(self):
        return DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(test_dataset, collate_fn=eval_collate_fn, batch_size=self.batch_size, shuffle=False, num_workers=4)

class PushToHubCallback(Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            print(f"Pushing model to the hub, epoch {trainer.current_epoch}")
            pl_module.model.push_to_hub(REPO_ID,
                                        commit_message=f"Training in progress, epoch {trainer.current_epoch}")

        def on_train_end(self, trainer, pl_module):
            print(f"Pushing model to the hub after training")
            pl_module.processor.push_to_hub(REPO_ID,
                                        commit_message=f"Training done")
            pl_module.model.push_to_hub(REPO_ID,
                                        commit_message=f"Training done")
            
if __name__ == "__main__":
    with open("abductiveEk100/post_processed_train_data_instances.json") as f:
        train_data = json.load(f)
    with open("abductiveEk100/post_processed_test_data_instances.json") as f:
        test_data = json.load(f)
    # Split the dataset into train and test
    train_dataset = VideoLlavaDataset(train_data[:10], mode="train")
    test_dataset = VideoLlavaDataset(test_data, mode="test")

    ## Load model
    # Three options for training, from the lowest precision training to the highest precision training:
    # QLoRA: model uses 4-bit quantization, which helps in reducing memory usage while maintaining performance.
    # Standard LoRA:  model is loaded with standard LoRA adaptations.
    # Full Fine-Tuning: no memory optimization are done. In that case Flash Attention is used to speed up training, if hardware supports it.

    if USE_QLORA or USE_LORA:
        if USE_QLORA:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        model = VideoLlavaForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        # for full fine-tuning, we can speed up the model using Flash Attention
        # only available on certain devices, see https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features
        model = VideoLlavaForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            _attn_implementation="flash_attention_2",
            device_map="auto",
    )
        
    lora_config = LoraConfig(
    r=8,
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules=find_all_linear_names(model),
    init_lora_weights="gaussian",
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    
    config = {"max_epochs": 2,
          # "val_check_interval": 0.2, # how many times we want to validate during an epoch
          "check_val_every_n_epoch": 1,
          "gradient_clip_val": 1.0,
          "accumulate_grad_batches": 8,
          "lr": 1e-4,
          "batch_size": 1,
          "num_nodes": 1,
          "warmup_steps": 50,
    }

    model_module = VideoLlavaModelPLModule(config, processor, model)

    # early_stop_callback = EarlyStopping(monitor="val_edit_distance", patience=3, verbose=False, mode="min")
    early_stop_callback = EarlyStopping(monitor="val_accuracy", patience=3, verbose=False, mode="min")
    
    # checkpoint_callback = ModelCheckpoint(
    # monitor='val_loss',          # metric to monitor for saving
    # dirpath='./checkpoints',      # directory where checkpoints are saved
    # filename='my-model-{epoch:02d}-{val_loss:.2f}',  # name for each checkpoint
    # save_top_k=1,                 # save only the best model
    # mode='min'                    # minimize the val_loss
    # )
    
    trainer = L.Trainer(
        default_root_dir="video-llava/",
        accelerator="gpu",
        devices=[0],
        max_epochs=config.get("max_epochs"),
        accumulate_grad_batches=config.get("accumulate_grad_batches"),
        check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
        val_check_interval=config.get("val_check_interval"),
        gradient_clip_val=config.get("gradient_clip_val"),
        precision="16-mixed",
        limit_val_batches=5,
        num_sanity_val_steps=1,
        callbacks=[early_stop_callback],
    )
    
    trainer.fit(model_module)