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


BATCH_SIZE = 8 # 2 A100 stage 1 bs=8 
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
STAGE = 1 ######## ********** ########
DATA_VERSION = 'v4' ######## ********** ########
NUMS = 25 ######## ********** ########
N = 6

MODE = 'TEST'
DEVICES = "y1" ######## ********** ########
EM_POOL = False
VLM_FILM = False
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


print("****************************** STARTING TESTING ******************************")

    
if __name__ == "__main__":

    all_devices = ['y1']
    for DEVICES in all_devices:
        print(f"Evaluating {DEVICES} ...")
        metric_tools = Metric()
        
        (train_digital_low_waveforms, train_record_low_waveforms), \
        (val_digital_low_waveforms, val_record_low_waveforms), \
        (test_digital_low_waveforms, test_record_low_waveforms) = preprocess(f"data/{DATA_VERSION}/x", f"data/{DATA_VERSION}/{DEVICES}", SEGMENT_LENGTH, STRIDE_LENGTH, SAMPLE_RATE, NUMS, MONO) # f"data/EN_y{STAGE}"
       
        (train_digital_waveforms, val_digital_waveforms, test_digital_waveforms)  = (train_digital_low_waveforms, val_digital_low_waveforms, test_digital_low_waveforms) 
        (train_record_waveforms, val_record_waveforms, test_record_waveforms)  = (train_record_low_waveforms, val_record_low_waveforms, test_record_low_waveforms) 
        
        test_speakers = None
        test_dataset = EqualizerDataset(test_digital_waveforms, test_record_waveforms, test_speakers, return_dict=False, embedding_pool=EM_POOL, mode=MODE)
        for size in [25]:#[20,25,30,40,50,60]:
            checkpoint_path = "assets/v4/30_stage1_y1_False_False__30__size_test_gen/checkpoint-6660/pytorch_model.bin"#f"assets/v4/{size}_stage1_{DEVICES}_False_False__30__size_test/pytorch_model.bin"
            print(checkpoint_path, EM_POOL, DEVICES)
            model = DemucsEqualizer(freeze=FREEZE)

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

                    predictions = model(inputs)

                    pred_audio = predictions.cpu().numpy()
                    gt_audio = labels.cpu().numpy()
                    batch_metrics = metric_tools.compute_all_metrics(gt_audio, pred_audio)

                    for key in all_metrics.keys():
                        all_metrics[key].extend(batch_metrics[key])

            final_metrics = {key: np.mean(values) for key, values in all_metrics.items()}

            print("\nEvaluation Results on Test Set:")
            for key, value in final_metrics.items():
                print(f"  - {key}: {value:.4f}")
