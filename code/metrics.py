import numpy as np
import librosa
from sklearn.metrics.pairwise import cosine_similarity
from pesq import pesq
import torch
from pystoi import stoi

class Metric:
    def __init__(self, sr=44100):
        self.sr = sr  # Sampling rate

    def mse(self, references, estimates):
        return np.mean((references - estimates) ** 2, axis=2) 
    
    def rmse(self, references, estimates):
        return np.sqrt(self.mse(references, estimates))

    def sdr(self, references, estimates):
        delta = 1e-8
        num = np.sum(np.square(references), axis=2)
        den = np.sum(np.square(references - estimates), axis=2)
        return 10 * np.log10((num + delta) / (den + delta))
    
    def snr(self, references, estimates):
        noise = references - estimates
        signal_power = np.sum(np.square(references), axis=2)
        noise_power = np.sum(np.square(noise), axis=2)
        return 10 * np.log10((signal_power + 1e-8) / (noise_power + 1e-8))

    # SIR Calculation: Signal-to-Interference Ratio
    def sir(self, references, estimates):
        interference = estimates - references  # Assuming reference is the true signal, interference is the difference
        signal_power = np.sum(np.square(references), axis=2)
        interference_power = np.sum(np.square(interference), axis=2)
        return 10 * np.log10((signal_power + 1e-8) / (interference_power + 1e-8))

    def cos_sim(self, references, estimates):
        references = references.reshape(references.shape[0], -1)
        estimates = estimates.reshape(estimates.shape[0], -1)
        return cosine_similarity(references, estimates)[:, 0].reshape(references.shape[0], -1)

    def sisnr(self, references, estimates):
        references = references - np.mean(references, axis=2, keepdims=True)
        estimates = estimates - np.mean(estimates, axis=2, keepdims=True)
        reference_energy = np.sum(references ** 2, axis=2, keepdims=True)
        projection = np.sum(references * estimates, axis=2, keepdims=True) * references / (reference_energy + 1e-8)
        noise = estimates - projection
        return 10 * np.log10(np.sum(projection ** 2, axis=2) / (np.sum(noise ** 2, axis=2) + 1e-8))

    def pesq_score(self, references, estimates, original_sr=44100, target_sr=16000):
        return np.array([
            pesq(target_sr, 
                librosa.resample(ref[0, :], orig_sr=original_sr, target_sr=target_sr), 
                librosa.resample(est[0, :], orig_sr=original_sr, target_sr=target_sr), 
                'wb'
            ) 
            for ref, est in zip(references, estimates)
        ]).reshape(references.shape[0], -1)
        
    def stoi_score(self, references, estimates):
        return np.array([stoi(ref[0, :], est[0, :], self.sr, extended=False) for ref, est in zip(references, estimates)]).reshape(references.shape[0], -1)

    # SNR Calculation: Signal-to-Noise Ratio
    def snr(self, references, estimates):
        noise = references - estimates
        signal_power = np.sum(np.square(references), axis=2)
        noise_power = np.sum(np.square(noise), axis=2)
        return 10 * np.log10((signal_power + 1e-8) / (noise_power + 1e-8))

    # SIR Calculation: Signal-to-Interference Ratio
    def sir(self, references, estimates):
        interference = estimates - references  # Assuming reference is the true signal, interference is the difference
        signal_power = np.sum(np.square(references), axis=2)
        interference_power = np.sum(np.square(interference), axis=2)
        return 10 * np.log10((signal_power + 1e-8) / (interference_power + 1e-8))

    def compute_all_metrics(self, references, estimates):
        return {
            "MSE": self.rmse(references, estimates),
            "SDR": self.sdr(references, estimates),
            # "PESQ": self.pesq_score(references, estimates),
            "STOI": self.stoi_score(references, estimates),
            # "Similarity": self.cos_sim(references, estimates),
            "SI-SNR": self.sisnr(references, estimates),
            "SNR": self.snr(references, estimates),
            "SIR": self.sir(references, estimates),
        }
