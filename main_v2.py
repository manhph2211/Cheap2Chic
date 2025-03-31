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

NUM_WORKERS = 8
SHUFFLE = True
SAMPLE_RATE = 44100
MONO = False
SEGMENT_LENGTH = 5
STRIDE_LENGTH = 0.5
GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_TYPE = "demucs" 
FREEZE = False

EPOCHS = 15 
TEST_FILES = 5 
N = 10 
DATA_VERSION = 'v5' 
MODEL_VERSION = 'v2' 

STAGE = 1 ######## ********** ########
PAIR = ["y3", "y4"]######## ********** ########
MODE = 'TRAIN'  ######## ********** ######## 

BATCH_SIZE = 8 if STAGE == 1 else 4 
NUMS = 20 if STAGE == 1 else 60 
LEARNING_RATE = 5e-5 

DEVICES = "ALL" if STAGE == 1 else PAIR 
EM_POOL = True 
VLM_FILM = True 

TRAIN_WITHOUT_Y = PAIR ######## ********** ########

TUNE_ONLY_Y = PAIR[0] 
FREEZE_DEMUC = False
POOL = 30

TUNE_RATIO = 0.05 ######## ********** ########
GEN = False  ######## ********** ########
TUNE_GEN_RATIO = 0 if not GEN else 0.15 ######## ********** ########

"""
Step 1: Pretrain DeMT using mix of devices
Step 2: Finetune on unseen device
Step 3: Generate and continue finetune
Step 4: Train stage 2
"""

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
    if MODE == "TRAIN":    

        train_speakers, val_speakers, test_speakers = [], [], []
        train_digital_waveforms, val_digital_waveforms, test_digital_waveforms = [], [], []
        train_record_waveforms, val_record_waveforms, test_record_waveforms = [], [], []

        print(f"STAGE: {STAGE}, DEVICE: {DEVICES}, NUMS: {NUMS}, LEARNING_RATE: {LEARNING_RATE}" )

        if DEVICES != "ALL":
            (train_digital_low_waveforms, train_record_low_waveforms), \
            (val_digital_low_waveforms, val_record_low_waveforms), \
            (test_digital_low_waveforms, test_record_low_waveforms) = preprocess(f"data/{DATA_VERSION}/x", f"data/{DATA_VERSION}/{DEVICES[0]}", SEGMENT_LENGTH, STRIDE_LENGTH, SAMPLE_RATE, NUMS, MONO, TEST_FILES) # f"data/EN_y{STAGE}"

            (train_digital_high_waveforms, train_record_high_waveforms), \
            (val_digital_high_waveforms, val_record_high_waveforms), \
            (test_digital_high_waveforms, test_record_high_waveforms) = preprocess(f"data/{DATA_VERSION}/x", f"data/{DATA_VERSION}/{DEVICES[1]}", SEGMENT_LENGTH, STRIDE_LENGTH, SAMPLE_RATE, NUMS, MONO, TEST_FILES) # f"data/EN_y{STAGE}"

            (train_digital_waveforms, val_digital_waveforms, test_digital_waveforms)  = (train_digital_low_waveforms, val_digital_low_waveforms, test_digital_low_waveforms) if STAGE == 1 else (train_digital_high_waveforms, val_digital_high_waveforms, test_digital_high_waveforms)
            (train_record_waveforms, val_record_waveforms, test_record_waveforms)  = (train_record_low_waveforms, val_record_low_waveforms, test_record_low_waveforms) if STAGE == 1 else (train_record_high_waveforms, val_record_high_waveforms, test_record_high_waveforms)
            if STAGE == 2:
                train_speakers += [DEVICES[0]] * len(train_digital_waveforms) 
                val_speakers += [DEVICES[0]] * len(val_digital_waveforms) 

        else:
            for i in range(1, 1+N):
                if f"y{i}" in TRAIN_WITHOUT_Y:
                    print(f"IGNORING DEVICE {i} !!!")
                    continue 
                (train_digital_low_waveforms, train_record_low_waveforms), \
                (val_digital_low_waveforms, val_record_low_waveforms), \
                (test_digital_low_waveforms, test_record_low_waveforms) = preprocess(f"data/{DATA_VERSION}/x", f"data/{DATA_VERSION}/y{i}", SEGMENT_LENGTH, STRIDE_LENGTH, SAMPLE_RATE, NUMS, MONO, TEST_FILES) 

                train_digital_waveforms += train_digital_low_waveforms 
                val_digital_waveforms += val_digital_low_waveforms 
                test_digital_waveforms += test_digital_low_waveforms 

                train_record_waveforms += train_record_low_waveforms 
                val_record_waveforms += val_record_low_waveforms 
                test_record_waveforms += test_record_low_waveforms
                
                train_speakers += [f"y{i}"] * len(train_digital_low_waveforms) 
                val_speakers += [f"y{i}"] * len(val_digital_low_waveforms) 
                test_speakers += [f"y{i}"] * len(test_digital_low_waveforms) 
                    
            print("USING TEXT CONDITIONED MODEL !!!")
            print(f"Train dataset size: {len(train_digital_waveforms)}")
            print(f"Validation dataset size: {len(val_digital_waveforms)}")
            print(f"Test dataset size: {len(test_digital_waveforms)}")
            assert len(test_digital_waveforms) == len(test_speakers) == len(test_record_waveforms)
            
        print(len(train_digital_waveforms), len(val_digital_waveforms), len(test_digital_waveforms))
        
        if not VLM_FILM:
            print("NOT USING ANY CONDITIONS !!!")
            train_speakers, val_speakers = None, None

        train_dataset = EqualizerDataset(train_digital_waveforms, train_record_waveforms, train_speakers, return_dict=True, embedding_pool=EM_POOL, pool_size=POOL)
        val_dataset = EqualizerDataset(val_digital_waveforms, val_record_waveforms, val_speakers, return_dict=True, embedding_pool=EM_POOL, pool_size=POOL)
        
        step = "step1" if STAGE == 1 else "step4"
        save_path = f"assets/{DATA_VERSION}/main_no_high/{step}/{str(NUMS)}_stage{STAGE}_train_{DEVICES}_without_{TRAIN_WITHOUT_Y}"
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
        if DEVICES != "ALL":

            if MODEL_TYPE == "wav2vec":
                model = TrainerModelWrapper(Wav2VecEqualizer())
            elif MODEL_TYPE == "demucs":
                if STAGE == 1:
                    init_model = DemucsEqualizer()
                    model = TrainerModelWrapper(init_model, version=MODEL_VERSION)
                elif STAGE == 2:
                    model = TrainerModelWrapper(DoubleDemucsEqualizer("", model_name=MODEL_VERSION), version=MODEL_VERSION)
                else:
                    raise "Not Implement!"
            else:
                raise "Not Implement!"
        else:
            model = TrainerModelWrapper(StyleTransform2(), version=MODEL_VERSION)
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        trainer.train()
        
    elif MODE=='TEST':
        for DEVICES in ["XXX"]: ####
            print(f"Evaluating {DEVICES} ...")
            metric_tools = Metric()
            
            (train_digital_low_waveforms, train_record_low_waveforms), \
            (val_digital_low_waveforms, val_record_low_waveforms), \
            (test_digital_low_waveforms, test_record_low_waveforms) = preprocess(f"data/{DATA_VERSION}/x", f"data/{DATA_VERSION}/{DEVICES}", SEGMENT_LENGTH, STRIDE_LENGTH, SAMPLE_RATE, NUMS, MONO) # f"data/EN_y{STAGE}"
            
            (train_digital_high_waveforms, train_record_high_waveforms), \
            (val_digital_high_waveforms, val_record_high_waveforms), \
            (test_digital_high_waveforms, test_record_high_waveforms) = preprocess(f"data/{DATA_VERSION}/x", f"data/{DATA_VERSION}/{DEVICES}", SEGMENT_LENGTH, STRIDE_LENGTH, SAMPLE_RATE, NUMS, MONO) # f"data/EN_y{STAGE}"

            (train_digital_waveforms, val_digital_waveforms, test_digital_waveforms)  = (train_digital_low_waveforms, val_digital_low_waveforms, test_digital_low_waveforms) if STAGE == 1 else (train_digital_high_waveforms, val_digital_high_waveforms, test_digital_high_waveforms)
            (train_record_waveforms, val_record_waveforms, test_record_waveforms)  = (train_record_low_waveforms, val_record_low_waveforms, test_record_low_waveforms) if STAGE == 1 else (train_record_high_waveforms, val_record_high_waveforms, test_record_high_waveforms)
            
            test_speakers = [DEVICES] * len(test_record_low_waveforms) if VLM_FILM else []
            test_dataset = EqualizerDataset(test_digital_waveforms, test_record_waveforms, test_speakers, return_dict=False, embedding_pool=EM_POOL, mode=MODE)

            checkpoint_path = ""
            print(checkpoint_path, EM_POOL, DEVICES)
            if STAGE == 2:
                model = DoubleDemucsEqualizer("", device=DEVICES)
            else:
                model = StyleTransform2()
        
            state_dict = torch.load(checkpoint_path, map_location=GPU)
            state_dict = {k[6:]: v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            model.eval()  
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=True
            )

            model.to(GPU)
            model.eval()

            all_metrics = {k: [] for k in ["SDR", "MSE", "STOI", "SI-SNR", "SNR", "SIR"]}
            with torch.no_grad():
                for batch in tqdm(test_loader, desc="Evaluating on Test Set"):

                    inputs = batch[0].to(GPU)  
                    labels = batch[1].to(GPU) 
                    if VLM_FILM:
                        embs = batch[2].to(GPU) 
                    else:
                        embs = None

                    predictions = model(inputs, embs)

                    pred_audio = predictions.cpu().numpy()
                    gt_audio = labels.cpu().numpy()
                    batch_metrics = metric_tools.compute_all_metrics(gt_audio, pred_audio)

                    for key in all_metrics.keys():
                        all_metrics[key].extend(batch_metrics[key])

            final_metrics = {key: np.mean(values) for key, values in all_metrics.items()}

            print("\nEvaluation Results on Test Set:")
            for key, value in final_metrics.items():
                print(f"  - {key}: {value:.4f}")
    
    else:
        for TUNE_ONLY_Y in tqdm([PAIR[0]]):
            print(f"########## TUNING {TUNE_ONLY_Y} ##########")
            train_speakers, val_speakers, test_speakers = [], [], []
            train_digital_waveforms, val_digital_waveforms, test_digital_waveforms = [], [], []
            train_record_waveforms, val_record_waveforms, test_record_waveforms = [], [], []
            
            for i in range(1, 1+N):
                (train_digital_low_waveforms, train_record_low_waveforms), \
                (val_digital_low_waveforms, val_record_low_waveforms), \
                (test_digital_low_waveforms, test_record_low_waveforms) = preprocess(f"data/{DATA_VERSION}/x", f"data/{DATA_VERSION}/y{i}", SEGMENT_LENGTH, STRIDE_LENGTH, SAMPLE_RATE, NUMS, MONO) # f"data/EN_y{STAGE}"
                
                train_selected_nums = int(TUNE_RATIO*len(train_digital_low_waveforms))
                val_selected_nums = int(TUNE_RATIO*len(val_digital_low_waveforms))

                if GEN and f'y{i}'==TUNE_ONLY_Y:
                    if TUNE_GEN_RATIO == 'ALL':
                        (train_digital_low_waveforms, train_record_low_waveforms), \
                        (val_digital_low_waveforms, val_record_low_waveforms), \
                        (test_digital_low_waveforms, test_record_low_waveforms) = preprocess(f"data/{DATA_VERSION}/x", f"data/{DATA_VERSION}/y{i}", SEGMENT_LENGTH, STRIDE_LENGTH, SAMPLE_RATE, 40, MONO) # f"data/EN_y{STAGE}"
                        train_selected_nums = len(train_digital_low_waveforms)
                        val_selected_nums = len(val_digital_low_waveforms)
                        increase_nums = 1
                    else:
                        increase_nums = int(TUNE_GEN_RATIO * (NUMS-TEST_FILES))
                    print(f"Preparing add more {increase_nums} segments")
                    print(f"######### Before augmentation: training data size: {train_selected_nums}, each shaped: {train_record_low_waveforms[0].shape} #########")
                    add_train_record_waveforms = gen(train_digital_low_waveforms[:train_selected_nums*increase_nums], train_record_low_waveforms[:train_selected_nums*increase_nums], checkpoint_path="", speaker=TUNE_ONLY_Y)
                    print(f"######### After augmentation: training data size: {len(add_train_record_waveforms)}, each shaped: {train_record_low_waveforms[0].shape} #########")
                    # val_record_waveforms = gen(val_digital_waveforms)
                    # np.save(f"train_record_waveforms_y{i}.npy", add_train_record_waveforms)
                    add_train_digital_waveforms = train_digital_low_waveforms[:train_selected_nums*increase_nums]
                else:
                    add_train_digital_waveforms = train_digital_low_waveforms[:train_selected_nums] 
                    add_train_record_waveforms = train_record_low_waveforms[:train_selected_nums]
                
                add_val_digital_waveforms = val_digital_low_waveforms[:val_selected_nums]
                add_val_record_waveforms = val_record_low_waveforms[:val_selected_nums]
            
                train_digital_waveforms += add_train_digital_waveforms
                val_digital_waveforms += add_val_digital_waveforms

                train_record_waveforms += add_train_record_waveforms
                val_record_waveforms += add_val_record_waveforms
                
                train_speakers += [f"y{i}"] * len(add_train_digital_waveforms) 
                val_speakers += [f"y{i}"] * len(add_val_digital_waveforms)

            print("USING TEXT CONDITIONED MODEL !!!")
            print(f"Train dataset size: {len(train_digital_waveforms)}")
            print(f"Validation dataset size: {len(val_digital_waveforms)}")
            print(f"Test dataset size: {len(test_digital_waveforms)}")
            
            assert len(test_digital_waveforms) == len(test_speakers) == len(test_record_waveforms)
            
            print(len(train_digital_waveforms), len(val_digital_waveforms), len(test_digital_waveforms))

        
            train_dataset = EqualizerDataset(train_digital_waveforms, train_record_waveforms, train_speakers, return_dict=True, embedding_pool=EM_POOL, pool_size=POOL)
            val_dataset = EqualizerDataset(val_digital_waveforms, val_record_waveforms, val_speakers, return_dict=True, embedding_pool=EM_POOL, pool_size=POOL)

            print(len(train_dataset), len(val_dataset))


            checkpoint_path = ""

            model = StyleTransform2()
            state_dict = torch.load(checkpoint_path, map_location=GPU)
            state_dict = {k[6:]: v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)

            model.to(GPU)
            step = "step2" if not GEN else "step3"
            save_path = f"assets/{DATA_VERSION}/main_no_high/{step}/{str(NUMS)}_stage{STAGE}_train_{DEVICES}_without_{TRAIN_WITHOUT_Y}_and_tune_{TUNE_ONLY_Y}_with_ratio_{TUNE_RATIO}_gen_{TUNE_GEN_RATIO}"
            print(save_path)
            training_args = TrainingArguments(
                output_dir=save_path,
                eval_strategy="epoch",
                learning_rate=LEARNING_RATE,
                save_strategy='epoch',
                logging_steps=50,
                per_device_train_batch_size=BATCH_SIZE,
                per_device_eval_batch_size=BATCH_SIZE,
                num_train_epochs=10,
                weight_decay=0.01,
                logging_dir="assets/logs",
                report_to="wandb",
                load_best_model_at_end=True,
                save_total_limit=2,
                save_safetensors=False,
            )
            
            trainer = Trainer(
                model=TrainerModelWrapper(model),
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
            )

            trainer.train()
    
    print("****************************** ENDING EXPERIMENT ******************************")
