import os
import librosa
from tqdm import tqdm
import torch
from transformers import Trainer, TrainingArguments
import torch
from models.wav2vec import Wav2VecEqualizer
from trainer import TrainerModelWrapper, compute_metrics
from utils import preprocess
from dataset import EqualizerDataset
from models.demucs import DemucsEqualizer, DoubleDemucsEqualizer

BATCH_SIZE = 32
NUM_WORKERS = 8
SHUFFLE = True
SAMPLE_RATE = 16000  
SEGMENT_LENGTH = 5
STRIDE_LENGTH = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_TYPE = "demucs" # wav2vec
FREEZE = False
LEARNING_RATE = 1e-4
EPOCHS = 50
STAGE = 2

    
if __name__ == "__main__":
    (train_digital_waveforms, train_record_low_waveforms), \
    (val_digital_waveforms, val_record_low_waveforms), \
    (test_digital_waveforms, test_record_low_waveforms) = preprocess("data/EN_x", f"data/EN_y{STAGE}", SEGMENT_LENGTH, STRIDE_LENGTH, SAMPLE_RATE)

    train_dataset = EqualizerDataset(train_digital_waveforms, train_record_low_waveforms, return_dict=True)
    val_dataset = EqualizerDataset(val_digital_waveforms, val_record_low_waveforms, return_dict=True)
    test_dataset = EqualizerDataset(test_digital_waveforms, test_record_low_waveforms, return_dict=True)
    
    training_args = TrainingArguments(
        output_dir="assets",
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
    if MODEL_TYPE == "wav2vec":
        model = TrainerModelWrapper(Wav2VecEqualizer(freeze_encoder=FREEZE))
    else:
        if STAGE == 1:
            model = TrainerModelWrapper(DemucsEqualizer(freeze=FREEZE))
        elif STAGE == 2:
            model = TrainerModelWrapper(DoubleDemucsEqualizer("assets/model1/pytorch_model.bin"))
        else:
            raise "Not Implement!"
            
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
