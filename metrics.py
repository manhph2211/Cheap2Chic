import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pesq import pesq


class Metric:
    def __init__(self, sr=44100):
        """
        Initialize the Metric class with a given sampling rate.
        """
        self.sr = sr  # Sampling rate of the audio

    def log_mse_time_domain(self, references, estimates):
        """
        Compute Log-MSE in the time domain.
        """
        references = (references - np.min(references, axis=1, keepdims=True)) / \
                     (np.max(references, axis=1, keepdims=True) - np.min(references, axis=1, keepdims=True))
        estimates = (estimates - np.min(estimates, axis=1, keepdims=True)) / \
                    (np.max(estimates, axis=1, keepdims=True) - np.min(estimates, axis=1, keepdims=True))
        return np.log(np.mean((references - estimates) ** 2, axis=1))

    def sdr(self, references, estimates):
        """
        Compute Signal-to-Distortion Ratio (SDR).
        """
        delta = 1e-7
        num = np.sum(np.square(references), axis=1)
        den = np.sum(np.square(references - estimates), axis=1)
        return 10 * np.log10((num + delta) / (den + delta))

    def cos_sim(self, references, estimates):
        """
        Compute cosine similarity between reference and estimated signals.
        """
        references = references.reshape(references.shape[0], -1)
        estimates = estimates.reshape(estimates.shape[0], -1)
        return np.array([cosine_similarity(ref.reshape(1, -1), est.reshape(1, -1))[0, 0] for ref, est in zip(references, estimates)])

    def sisnr(self, references, estimates):
        """
        Compute Scale-Invariant Signal-to-Noise Ratio (SI-SNR).
        """
        references = references - np.mean(references, axis=1, keepdims=True)
        estimates = estimates - np.mean(estimates, axis=1, keepdims=True)
        reference_energy = np.sum(references ** 2, axis=1, keepdims=True)
        projection = np.sum(references * estimates, axis=1, keepdims=True) * references / reference_energy
        noise = estimates - projection
        return 10 * np.log10(np.sum(projection ** 2, axis=1) / np.sum(noise ** 2, axis=1))

    def pesq_score(self, references, estimates):
        """
        Compute PESQ (Perceptual Evaluation of Audio Quality).
        Returns a score in the range [-0.5, 4.5], higher is better.
        """
        return np.array([pesq(self.sr, ref, est, 'wb') for ref, est in zip(references, estimates)])

    def compute_all_metrics(self, references, estimates):
        """
        Compute all metrics and return results in batch.
        """
        return {
            "Log-Magnitude MSE": self.log_mse_time_domain(references, estimates),
            "SDR": self.sdr(references, estimates),
            "PESQ": self.pesq_score(references, estimates),
            "Cosine Similarity": self.cos_sim(references, estimates),
            "SI-SNR": self.sisnr(references, estimates),
        }
