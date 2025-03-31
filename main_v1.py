import os
import librosa
from tqdm import tqdm
import torch
from transformers import Trainer, TrainingArguments
import torch
from models.wav2vec_equalizer import Wav2VecEqualizer
from trainer import TrainerModelWrapper, compute_metrics
from utils import preprocess, peak_normalize
from dataset import EqualizerDataset
from models.demucs_equalizer import DemucsEqualizer, DoubleDemucsEqualizer, StyleTransform1, StyleTransform2, gen
from torch.utils.data import DataLoader
import numpy as np 
from metrics import Metric
from torch.utils.data import Subset
from torch.utils.data import random_split
import math
from datetime import datetime

BATCH_SIZE = 4 # 2 A100 stage 1 bs=8 
NUM_WORKERS = 8
SHUFFLE = True
SAMPLE_RATE = 44100
MONO = False
SEGMENT_LENGTH = 5
STRIDE_LENGTH = 0.5
GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_TYPE = "demucs" 
FREEZE = False

LEARNING_RATE = 5e-5 
EPOCHS = 15 
DATA_VERSION = 'v5' 
TEST_FILES = 5 

STAGE = 2 ######## ********** ########
NUMS = 10 ######## ********** ######## 10 15 20 30 40 (add 25 35)

N = 10
MODEL_VERSION = 'v1'  

MODE = 'TRAIN'  ######## ********** ######## 
DEVICES = ['y3','y4'] 

EM_POOL = False 
VLM_FILM = False 

TRAIN_WITHOUT_Y = '' 
TUNE_ONLY_Y = '' 

TUNE_RATIO = 0.05 
GEN = False 

FREEZE_DEMUC = False
POOL = 30

config = {
    "BATCH_SIZE": BATCH_SIZE,
    "NUM_WORKERS": NUM_WORKERS,
    "SHUFFLE": SHUFFLE,
    "SAMPLE_RATE": SAMPLE_RATE,
    "MONO": MONO,
    "SEGMENT_LENGTH": SEGMENT_LENGTH,
    "STRIDE_LENGTH": STRIDE_LENGTH,
    "GPU": GPU,
    "MODEL_TYPE": MODEL_TYPE,
    "FREEZE": FREEZE,
    "LEARNING_RATE": LEARNING_RATE,
    "EPOCHS": EPOCHS,
    "STAGE": STAGE,
    "DATA_VERSION": DATA_VERSION,
    "NUMS": NUMS,
    "N": N,
    "MODE": MODE,
    "DEVICES": DEVICES,
    "EM_POOL": EM_POOL,
    "VLM_FILM": VLM_FILM,
    "TRAIN_WITHOUT_Y": TRAIN_WITHOUT_Y,
    "TUNE_ONLY_Y": TUNE_ONLY_Y,
    "FREEZE_DEMUC": FREEZE_DEMUC,
    "POOL": POOL,
    "TUNE_RATIO": TUNE_RATIO,
}

print("****************************** CONFIGURATIONS ******************************")

for key, value in config.items():
    print(f"{key}: {value}")

print("Current date and time:", datetime.now())

print("****************************** STARTING EXPERIMENT ******************************")

    
if __name__ == "__main__":

    train_speakers, val_speakers, test_speakers = [], [], []
    train_digital_waveforms, val_digital_waveforms, test_digital_waveforms = [], [], []
    train_record_waveforms, val_record_waveforms, test_record_waveforms = [], [], []

    print(f"STAGE: {STAGE}, DEVICE: {DEVICES}, NUMS: {NUMS}, LEARNING_RATE: {LEARNING_RATE}" )

    (train_digital_low_waveforms, train_record_low_waveforms), \
    (val_digital_low_waveforms, val_record_low_waveforms), \
    (test_digital_low_waveforms, test_record_low_waveforms) = preprocess(f"data/{DATA_VERSION}/x", f"data/{DATA_VERSION}/{DEVICES[0]}", SEGMENT_LENGTH, STRIDE_LENGTH, SAMPLE_RATE, NUMS, MONO, TEST_FILES) # f"data/EN_y{STAGE}"

    # NOTE this is for Stage 2
    (train_digital_high_waveforms, train_record_high_waveforms), \
    (val_digital_high_waveforms, val_record_high_waveforms), \
    (test_digital_high_waveforms, test_record_high_waveforms) = preprocess(f"data/{DATA_VERSION}/x", f"data/{DATA_VERSION}/{DEVICES[1]}", SEGMENT_LENGTH, STRIDE_LENGTH, SAMPLE_RATE, 60, MONO, TEST_FILES) # f"data/EN_y{STAGE}"

    (train_digital_waveforms, val_digital_waveforms, test_digital_waveforms)  = (train_digital_low_waveforms, val_digital_low_waveforms, test_digital_low_waveforms) if STAGE == 1 else (train_digital_high_waveforms, val_digital_high_waveforms, test_digital_high_waveforms)
    (train_record_waveforms, val_record_waveforms, test_record_waveforms)  = (train_record_low_waveforms, val_record_low_waveforms, test_record_low_waveforms) if STAGE == 1 else (train_record_high_waveforms, val_record_high_waveforms, test_record_high_waveforms)
        
    print(len(train_digital_waveforms), len(val_digital_waveforms), len(test_digital_waveforms))
    print("NOT USING ANY CONDITIONS !!!")
    train_speakers, val_speakers = None, None

    train_dataset = EqualizerDataset(train_digital_waveforms, train_record_waveforms, train_speakers, return_dict=True, embedding_pool=EM_POOL, pool_size=POOL)
    val_dataset = EqualizerDataset(val_digital_waveforms, val_record_waveforms, val_speakers, return_dict=True, embedding_pool=EM_POOL, pool_size=POOL)
    
    save_path = f"assets/{DATA_VERSION}/curve/stage{STAGE}/{str(NUMS)}_{DEVICES}"
    print(save_path)
    training_args = TrainingArguments(
        output_dir=save_path,
        eval_strategy="epoch",
        learning_rate=LEARNING_RATE,
        save_strategy='epoch',
        logging_steps=50,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_dir="assets/logs",
        report_to="wandb",
        load_best_model_at_end=True,
        save_total_limit=2,
        save_safetensors=False,
    )
    
    if STAGE == 1:
        init_model = DemucsEqualizer()
        model = TrainerModelWrapper(init_model, version=MODEL_VERSION)
    elif STAGE == 2:
        stage1_ckpt = f"assets/{DATA_VERSION}/curve/stage1/{str(NUMS)}_{DEVICES}/pytorch_model.bin"
        model = TrainerModelWrapper(DoubleDemucsEqualizer(stage1_ckpt, model_name=MODEL_VERSION), version=MODEL_VERSION)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train(resume_from_checkpoint="assets/v5/curve/stage2/10_['y3', 'y4']/checkpoint-32505")
    
