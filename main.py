import os
import librosa
from tqdm import tqdm
import torch
from transformers import Trainer, TrainingArguments
import torch
from models.wav2vec_equalizer import Wav2VecEqualizer
from trainer import TrainerModelWrapper, compute_metrics
from utils import preprocess
from dataset import EqualizerDataset
from models.demucs_equalizer import DemucsEqualizer, DoubleDemucsEqualizer, StyleTransform1, StyleTransform2
from torch.utils.data import DataLoader
import numpy as np 
# from metrics import Metric


BATCH_SIZE = 8 # A100 stage 1 bs=8
NUM_WORKERS = 8
SHUFFLE = True
SAMPLE_RATE = 44100  
MONO = False
SEGMENT_LENGTH = 5
STRIDE_LENGTH = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_TYPE = "demucs" 
FREEZE = False
LEARNING_RATE = 5e-5 
EPOCHS = 20

STAGE = 1
DATA_VERSION = 'v3'
NUMS = 20
N = 5
MODE = 'TRAIN'
DEVICE = "ALL"

    
if __name__ == "__main__":
    if MODE == "TRAIN":    

        train_speakers, val_speakers, test_speakers = [], [], []
        train_digital_waveforms, val_digital_waveforms, test_digital_waveforms = [], [], []
        train_record_waveforms, val_record_waveforms, test_record_waveforms = [], [], []

        print(f"STAGE: {STAGE}, DEVICE: {DEVICE}, NUMS: {NUMS}, LEARNING_RATE: {LEARNING_RATE}" )

        if DEVICE != "ALL":
            (train_digital_low_waveforms, train_record_low_waveforms), \
            (val_digital_low_waveforms, val_record_low_waveforms), \
            (test_digital_low_waveforms, test_record_low_waveforms) = preprocess(f"data/{DATA_VERSION}/x", f"data/{DATA_VERSION}/{DEVICE}", SEGMENT_LENGTH, STRIDE_LENGTH, SAMPLE_RATE, NUMS, MONO) # f"data/EN_y{STAGE}"

            # NOTE this is for Stage 2
            (train_digital_high_waveforms, train_record_high_waveforms), \
            (val_digital_high_waveforms, val_record_high_waveforms), \
            (test_digital_high_waveforms, test_record_high_waveforms) = preprocess(f"data/{DATA_VERSION}/x", f"data/{DATA_VERSION}/{DEVICE}", SEGMENT_LENGTH, STRIDE_LENGTH, SAMPLE_RATE, NUMS, MONO) # f"data/EN_y{STAGE}"

            (train_digital_waveforms, val_digital_waveforms, test_digital_waveforms)  = (train_digital_low_waveforms, val_digital_low_waveforms, test_digital_low_waveforms) if STAGE == 1 else (train_digital_high_waveforms, val_digital_high_waveforms, test_digital_high_waveforms)
            (train_record_waveforms, val_record_waveforms, test_record_waveforms)  = (train_record_low_waveforms, val_record_low_waveforms, test_record_low_waveforms) if STAGE == 1 else (train_record_high_waveforms, val_record_high_waveforms, test_record_high_waveforms)

        else:
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
        train_dataset = EqualizerDataset(train_digital_waveforms, train_record_waveforms, train_speakers, return_dict=True)
        val_dataset = EqualizerDataset(val_digital_waveforms, val_record_waveforms, val_speakers, return_dict=True)
        # test_dataset = EqualizerDataset(test_digital_waveforms, test_record_waveforms, test_speakers, return_dict=False)
        
        training_args = TrainingArguments(
            output_dir=f"assets/{DATA_VERSION}/{str(NUMS)}_stage{STAGE}_{DEVICE}",
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
        if DEVICE != "ALL":

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

        trainer.train()
    
    else:
        
        metric_tools = Metric()
        
        (train_digital_low_waveforms, train_record_low_waveforms), \
        (val_digital_low_waveforms, val_record_low_waveforms), \
        (test_digital_low_waveforms, test_record_low_waveforms) = preprocess(f"data/{DATA_VERSION}/x", f"data/{DATA_VERSION}/{DEVICE}", SEGMENT_LENGTH, STRIDE_LENGTH, SAMPLE_RATE, NUMS, MONO) # f"data/EN_y{STAGE}"
        
        # NOTE this is for Stage 2
        (train_digital_high_waveforms, train_record_high_waveforms), \
        (val_digital_high_waveforms, val_record_high_waveforms), \
        (test_digital_high_waveforms, test_record_high_waveforms) = preprocess(f"data/{DATA_VERSION}/x", f"data/{DATA_VERSION}/{DEVICE}", SEGMENT_LENGTH, STRIDE_LENGTH, SAMPLE_RATE, NUMS, MONO) # f"data/EN_y{STAGE}"

        (train_digital_waveforms, val_digital_waveforms, test_digital_waveforms)  = (train_digital_low_waveforms, val_digital_low_waveforms, test_digital_low_waveforms) if STAGE == 1 else (train_digital_high_waveforms, val_digital_high_waveforms, test_digital_high_waveforms)
        (train_record_waveforms, val_record_waveforms, test_record_waveforms)  = (train_record_low_waveforms, val_record_low_waveforms, test_record_low_waveforms) if STAGE == 1 else (train_record_high_waveforms, val_record_high_waveforms, test_record_high_waveforms)
        
        test_speakers = [DEVICE] * len(test_record_low_waveforms) 
        test_dataset = EqualizerDataset(test_digital_waveforms, test_record_waveforms, test_speakers, return_dict=False)

        checkpoint_path = f"assets/{DATA_VERSION}/{str(NUMS)}_stage{STAGE}/pytorch_model.bin"
        if STAGE == 2:
            model = DoubleDemucsEqualizer(f"assets/{DATA_VERSION}/{str(NUMS)}_stage1/pytorch_model.bin", device=DEVICE)
        else:
            model = StyleTransform2(device=DEVICE)
            
        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
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

        model.to(DEVICE)
        model.eval()

        all_metrics = {k: [] for k in ["Log-Magnitude MSE", "SDR", "PESQ", "Cosine Similarity", "SI-SNR"]}

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating on Test Set"):
                inputs = batch[0].to(DEVICE)  
                labels = batch[1].to(DEVICE) 
                embs = batch[2].to(DEVICE) 

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
