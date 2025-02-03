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
from torch.nn import L1Loss


BATCH_SIZE = 8 # A100 stage 1 bs=8
NUM_WORKERS = 8
SHUFFLE = True
SAMPLE_RATE = 44100  
SEGMENT_LENGTH = 5
STRIDE_LENGTH = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_TYPE = "demucs" 
FREEZE = False
LEARNING_RATE = 5e-5 
EPOCHS = 10

STAGE = 1
DATA_VERSION = 'v3'
NUMS = 20
MODE = 'TRAIN'
TEXT_CONDITIONED = True

    
if __name__ == "__main__":
    (train_digital_low_waveforms, train_record_low_waveforms), \
    (val_digital_low_waveforms, val_record_low_waveforms), \
    (test_digital_low_waveforms, test_record_low_waveforms) = preprocess(f"data/{DATA_VERSION}/x", f"data/{DATA_VERSION}/y1", SEGMENT_LENGTH, STRIDE_LENGTH, SAMPLE_RATE, NUMS) # f"data/EN_y{STAGE}"

    (train_digital_high_waveforms, train_record_high_waveforms), \
    (val_digital_high_waveforms, val_record_high_waveforms), \
    (test_digital_high_waveforms, test_record_high_waveforms) = preprocess(f"data/{DATA_VERSION}/x", f"data/{DATA_VERSION}/y2", SEGMENT_LENGTH, STRIDE_LENGTH, SAMPLE_RATE, NUMS) # f"data/EN_y{STAGE}"

    print(f"STAGE: {STAGE}, NUMS: {NUMS}, LEARNING_RATE: {LEARNING_RATE}" )

    if not TEXT_CONDITIONED:
        (train_digital_waveforms, val_digital_waveforms, test_digital_waveforms)  = (train_digital_low_waveforms, val_digital_low_waveforms, test_digital_low_waveforms) if STAGE == 1 else (train_digital_high_waveforms, val_digital_high_waveforms, test_digital_high_waveforms)
        (train_record_waveforms, val_record_waveforms, test_record_waveforms)  = (train_record_low_waveforms, val_record_low_waveforms, test_record_low_waveforms) if STAGE == 1 else (train_record_high_waveforms, val_record_high_waveforms, test_record_high_waveforms)
        (train_speakers, val_speakers, test_speakers) = (None, None, None)
    else:
        train_digital_waveforms = train_digital_low_waveforms + train_digital_high_waveforms
        val_digital_waveforms = val_digital_low_waveforms + val_digital_high_waveforms
        test_digital_waveforms = test_digital_low_waveforms + test_digital_high_waveforms

        train_record_waveforms = train_record_low_waveforms + train_record_high_waveforms
        val_record_waveforms = val_record_low_waveforms + val_record_high_waveforms
        test_record_waveforms = test_record_low_waveforms + test_record_high_waveforms
        
        train_speakers = ['y1']*len(train_digital_low_waveforms)  + ['y2']*len(train_digital_high_waveforms)
        val_speakers = ['y1']*len(val_digital_low_waveforms)  + ['y2']*len(val_digital_high_waveforms)
        test_speakers = ['y1']*len(test_digital_low_waveforms)  + ['y2']*len(test_digital_high_waveforms)
                
        print("USING TEXT CONDITIONED MODEL !!!")
        print(f"Train dataset size: {len(train_digital_waveforms)}")
        print(f"Validation dataset size: {len(val_digital_waveforms)}")
        print(f"Test dataset size: {len(test_digital_waveforms)}")
        
    print(len(train_digital_waveforms), len(val_digital_waveforms), len(test_digital_waveforms))
    train_dataset = EqualizerDataset(train_digital_waveforms, train_record_waveforms, train_speakers, return_dict=True)
    val_dataset = EqualizerDataset(val_digital_waveforms, val_record_waveforms, val_speakers, return_dict=True)
    test_dataset = EqualizerDataset(test_digital_waveforms, test_record_waveforms, test_speakers, return_dict=False)

    training_args = TrainingArguments(
        output_dir=f"assets/{DATA_VERSION}/{str(NUMS)}_stage{STAGE}_{TEXT_CONDITIONED}",
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
    if not TEXT_CONDITIONED:

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
    
    if MODE == "TRAIN":    
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )

        trainer.train()
    else:
        checkpoint_path = f"assets/{DATA_VERSION}/{str(NUMS)}_stage{STAGE}/pytorch_model.bin"
        if STAGE == 2:
            model = DoubleDemucsEqualizer(f"assets/{DATA_VERSION}/{str(NUMS)}_stage1/pytorch_model.bin", device=DEVICE)
        else:
            model = DemucsEqualizer(device=DEVICE)
            
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

        mae_loss = L1Loss()
        mae_list = []

        model.to(DEVICE)
        model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating on Test Set"):
                inputs = batch[0].to(DEVICE)  
                labels = batch[1].to(DEVICE) 
                predictions = model(inputs)
                mae = mae_loss(predictions, labels).item()
                mae_list.append(mae)

        mean_mae = sum(mae_list) / len(mae_list)
        print(f"Mean Absolute Error (MAE) on Test Set: {mean_mae}")
