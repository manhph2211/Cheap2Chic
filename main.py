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
from models.demucs_equalizer import DemucsEqualizer, DoubleDemucsEqualizer, StyleTransform1, StyleTransform2
from torch.utils.data import DataLoader
import numpy as np 
from metrics import Metric
from torch.utils.data import Subset
from torch.utils.data import random_split


BATCH_SIZE = 8 # A100 stage 1 bs=8
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
STAGE = 1
DATA_VERSION = 'v3'
NUMS = 20
N = 6

MODE = 'TEST'
DEVICES = "ALL"
EM_POOL = True
VLM_FILM = True
TRAIN_WITHOUT_Y = '' 
TUNE_ONLY_Y = ''
FREEZE_DEMUC = False
POOL = 30
TUNE_RATIO = 0.1


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
            (test_digital_low_waveforms, test_record_low_waveforms) = preprocess(f"data/{DATA_VERSION}/x", f"data/{DATA_VERSION}/{DEVICES}", SEGMENT_LENGTH, STRIDE_LENGTH, SAMPLE_RATE, NUMS, MONO) # f"data/EN_y{STAGE}"

            # NOTE this is for Stage 2
            (train_digital_high_waveforms, train_record_high_waveforms), \
            (val_digital_high_waveforms, val_record_high_waveforms), \
            (test_digital_high_waveforms, test_record_high_waveforms) = preprocess(f"data/{DATA_VERSION}/x", f"data/{DATA_VERSION}/{DEVICES}", SEGMENT_LENGTH, STRIDE_LENGTH, SAMPLE_RATE, NUMS, MONO) # f"data/EN_y{STAGE}"

            (train_digital_waveforms, val_digital_waveforms, test_digital_waveforms)  = (train_digital_low_waveforms, val_digital_low_waveforms, test_digital_low_waveforms) if STAGE == 1 else (train_digital_high_waveforms, val_digital_high_waveforms, test_digital_high_waveforms)
            (train_record_waveforms, val_record_waveforms, test_record_waveforms)  = (train_record_low_waveforms, val_record_low_waveforms, test_record_low_waveforms) if STAGE == 1 else (train_record_high_waveforms, val_record_high_waveforms, test_record_high_waveforms)

        else:
            for i in range(1, 1+N):
                if TRAIN_WITHOUT_Y == f"y{i}":
                    print(f"IGNORING DEVICE {i} !!!")
                    continue 
                (train_digital_low_waveforms, train_record_low_waveforms), \
                (val_digital_low_waveforms, val_record_low_waveforms), \
                (test_digital_low_waveforms, test_record_low_waveforms) = preprocess(f"data/{DATA_VERSION}/x", f"data/{DATA_VERSION}/y{i}", SEGMENT_LENGTH, STRIDE_LENGTH, SAMPLE_RATE, NUMS, MONO) # f"data/EN_y{STAGE}"

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
        
        training_args = TrainingArguments(
            output_dir=f"assets/{DATA_VERSION}/{str(NUMS)}_stage{STAGE}_{DEVICES}_{EM_POOL}_{VLM_FILM}_{TRAIN_WITHOUT_Y}_{POOL}_{TUNE_ONLY_Y}",
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
            eval_accumulation_steps=1,

        )
        if DEVICES != "ALL":

            if MODEL_TYPE == "wav2vec":
                model = TrainerModelWrapper(Wav2VecEqualizer(freeze_encoder=FREEZE))
            elif MODEL_TYPE == "demucs":
                if STAGE == 1:
                    model = TrainerModelWrapper(DemucsEqualizer(freeze=FREEZE))
                elif STAGE == 2:
                    model = TrainerModelWrapper(DoubleDemucsEqualizer(f"assets/{DATA_VERSION}/{str(NUMS)}_stage1/pytorch_model.bin"))
                else:
                    raise "Not Implement!"
            else:
                raise "Not Implement!"
        else:
            model = TrainerModelWrapper(StyleTransform2(freeze=FREEZE))
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )

        trainer.train() #resume_from_checkpoint="assets/v3/20_stage1_ALL_True_True__30__no_harman/checkpoint-5320"
    elif MODE=='TEST':
        all_devices = ['y1', 'y2', 'y3', 'y4', 'y5', 'y6'] #if len(TRAIN_WITHOUT_Y) == 0 else [TRAIN_WITHOUT_Y]
        for DEVICES in all_devices:
            print(f"Evaluating {DEVICES} ...")
            metric_tools = Metric()
            
            (train_digital_low_waveforms, train_record_low_waveforms), \
            (val_digital_low_waveforms, val_record_low_waveforms), \
            (test_digital_low_waveforms, test_record_low_waveforms) = preprocess(f"data/{DATA_VERSION}/x", f"data/{DATA_VERSION}/{DEVICES}", SEGMENT_LENGTH, STRIDE_LENGTH, SAMPLE_RATE, NUMS, MONO) # f"data/EN_y{STAGE}"
            
            # NOTE this is for Stage 2
            (train_digital_high_waveforms, train_record_high_waveforms), \
            (val_digital_high_waveforms, val_record_high_waveforms), \
            (test_digital_high_waveforms, test_record_high_waveforms) = preprocess(f"data/{DATA_VERSION}/x", f"data/{DATA_VERSION}/{DEVICES}", SEGMENT_LENGTH, STRIDE_LENGTH, SAMPLE_RATE, NUMS, MONO) # f"data/EN_y{STAGE}"

            (train_digital_waveforms, val_digital_waveforms, test_digital_waveforms)  = (train_digital_low_waveforms, val_digital_low_waveforms, test_digital_low_waveforms) if STAGE == 1 else (train_digital_high_waveforms, val_digital_high_waveforms, test_digital_high_waveforms)
            (train_record_waveforms, val_record_waveforms, test_record_waveforms)  = (train_record_low_waveforms, val_record_low_waveforms, test_record_low_waveforms) if STAGE == 1 else (train_record_high_waveforms, val_record_high_waveforms, test_record_high_waveforms)
            
            test_speakers = [DEVICES] * len(test_record_low_waveforms) if VLM_FILM else []
            test_dataset = EqualizerDataset(test_digital_waveforms, test_record_waveforms, test_speakers, return_dict=False, embedding_pool=EM_POOL, mode=MODE)

            checkpoint_path = f"assets/v3/20_stage1_ALL_True_True__30_TUNE_{DEVICES}_0.05/pytorch_model.bin"
            print(checkpoint_path, EM_POOL, DEVICES)
            if STAGE == 2:
                model = DoubleDemucsEqualizer(f"assets/{DATA_VERSION}/{str(NUMS)}_stage1/pytorch_model.bin", device=DEVICES)
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
                    # if len(all_devices) == 1:
                    #     gt_audio = peak_normalize(gt_audio)
                    #     pred_audio = peak_normalize(pred_audio)
                    batch_metrics = metric_tools.compute_all_metrics(gt_audio, pred_audio)

                    for key in all_metrics.keys():
                        all_metrics[key].extend(batch_metrics[key])

            final_metrics = {key: np.mean(values) for key, values in all_metrics.items()}

            print("\nEvaluation Results on Test Set:")
            for key, value in final_metrics.items():
                print(f"  - {key}: {value:.4f}")
    else:
        for TUNE_ONLY_Y in tqdm(['y1', 'y2', 'y3', 'y4', 'y5','y6']):
            print(f"########## TUNING {TUNE_ONLY_Y} ##########")
            train_speakers, val_speakers, test_speakers = [], [], []
            train_digital_waveforms, val_digital_waveforms, test_digital_waveforms = [], [], []
            train_record_waveforms, val_record_waveforms, test_record_waveforms = [], [], []

            for i in range(1, 1+N):
                (train_digital_low_waveforms, train_record_low_waveforms), \
                (val_digital_low_waveforms, val_record_low_waveforms), \
                (test_digital_low_waveforms, test_record_low_waveforms) = preprocess(f"data/{DATA_VERSION}/x", f"data/{DATA_VERSION}/y{i}", SEGMENT_LENGTH, STRIDE_LENGTH, SAMPLE_RATE, NUMS, MONO) # f"data/EN_y{STAGE}"

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
        
        
            train_dataset = EqualizerDataset(train_digital_waveforms, train_record_waveforms, train_speakers, return_dict=True, embedding_pool=EM_POOL, pool_size=POOL)
            num_samples = len(train_dataset)
            num_select = max(1, int(TUNE_RATIO * num_samples))
            train_subset, _ = random_split(train_dataset, [num_select, num_samples - num_select])

            val_dataset = EqualizerDataset(val_digital_waveforms, val_record_waveforms, val_speakers, return_dict=True, embedding_pool=EM_POOL, pool_size=POOL)

            num_val_samples = len(val_dataset)
            num_val_select = max(1, int(TUNE_RATIO * num_val_samples))
            val_subset, _ = random_split(val_dataset, [num_val_select, num_val_samples - num_val_select])
            print(num_select, num_val_select)

            # test_dataset = EqualizerDataset(test_digital_waveforms, test_record_waveforms, test_speakers, return_dict=False, embedding_pool=True)

            checkpoint_path = f"assets/v3/20_stage1_ALL_True_True_{TUNE_ONLY_Y}_30_/pytorch_model.bin"

            model = StyleTransform2()
            state_dict = torch.load(checkpoint_path, map_location=GPU)
            state_dict = {k[6:]: v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            if FREEZE_DEMUC: 
                for param in model.model1.parameters():
                    param.requires_grad = False                    
                                    
            model.to(GPU)

            training_args = TrainingArguments(
                output_dir=f"assets/{DATA_VERSION}/{str(NUMS)}_stage{STAGE}_{DEVICES}_{EM_POOL}_{VLM_FILM}_{TRAIN_WITHOUT_Y}_{POOL}_TUNE_{TUNE_ONLY_Y}_{TUNE_RATIO}",
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
                eval_accumulation_steps=1,

            )
            
            trainer = Trainer(
                model=TrainerModelWrapper(model),
                args=training_args,
                train_dataset=train_subset,
                eval_dataset=val_subset,
                compute_metrics=compute_metrics
            )

            trainer.train() #resume_from_checkpoint
    print("****************************** ENDING EXPERIMENT ******************************")
