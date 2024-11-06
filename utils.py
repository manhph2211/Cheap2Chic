import librosa
import os
from tqdm import tqdm


def preprocess(digital_path, record_low_path, segment_length, stride_length, target_sampling_rate):
    total_files = 30
    train_end = int(0.8 * total_files)
    val_end = train_end + int(0.1 * total_files)

    train_digital_waveforms, val_digital_waveforms, test_digital_waveforms = [], [], []
    train_record_low_waveforms, val_record_low_waveforms, test_record_low_waveforms = [], [], []

    for i in tqdm(range(total_files)):
        if i < train_end:
            digital_waveforms = train_digital_waveforms
            record_low_waveforms = train_record_low_waveforms
        elif i < val_end:
            digital_waveforms = val_digital_waveforms
            record_low_waveforms = val_record_low_waveforms
        else:
            digital_waveforms = test_digital_waveforms
            record_low_waveforms = test_record_low_waveforms

        digital_file = os.path.join(digital_path, f"{i}.wav")
        record_low_file = os.path.join(record_low_path, f"y{record_low_path[-1]}_{i}.wav")

        digital_data, _ = librosa.load(digital_file, sr=target_sampling_rate)
        record_low_data, _ = librosa.load(record_low_file, sr=target_sampling_rate)

        segment_samples = int(segment_length * target_sampling_rate)
        stride_samples = int(stride_length * target_sampling_rate)

        for start in range(0, len(digital_data) - segment_samples + 1, stride_samples):
            digital_segment = digital_data[start:start + segment_samples]
            record_low_segment = record_low_data[start:start + segment_samples]

            digital_waveforms.append(digital_segment)
            record_low_waveforms.append(record_low_segment)

    return (train_digital_waveforms, train_record_low_waveforms), \
           (val_digital_waveforms, val_record_low_waveforms), \
           (test_digital_waveforms, test_record_low_waveforms)

