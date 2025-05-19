from math import log10
from numpy.linalg import norm
import numpy as np
from dictionaries import *
from extra_functions import *

# Назви словників
dicts_name = ["gauss", "meyer", "poly", "haar"]
patch_size = 512
n_atoms = 31
n_trials = 1000
noise_std = 0.5  # стандартне відхилення шуму

# Ініціалізація метрик для OMP і MP
metrics = {
    "OMP": {
        "mse": np.zeros((4,)),
        "mae": np.zeros((4,)),
        "rmse": np.zeros((4,)),
        "snr": np.zeros((4,)),
        "psnr": np.zeros((4,)),
        "cosine": np.zeros((4,)),
        "pearson": np.zeros((4,))
    },
    "MP": {
        "mse": np.zeros((4,)),
        "mae": np.zeros((4,)),
        "rmse": np.zeros((4,)),
        "snr": np.zeros((4,)),
        "psnr": np.zeros((4,)),
        "cosine": np.zeros((4,)),
        "pearson": np.zeros((4,))
    }
}

# Створення словників
D_gauss, _ = create_gauss_dictionary(patch_size, n_atoms)
D_meyer, _ = create_meyer_wavelet_dictionary(patch_size, n_atoms)
D_poly, _ = create_polynomial_dictionary(patch_size, n_atoms)
D_haar, _ = create_haar_wavelet_dictionary(patch_size, n_atoms)

dicts = [D_gauss, D_meyer, D_poly, D_haar]

for _ in range(n_trials):
    for i, gen_dict in enumerate(dicts):
        signal, _ = create_signal_from_dictionary(gen_dict)
        noise = np.random.normal(0, noise_std, size=signal.shape)
        noisy_signal = signal + noise

        for method in ["OMP", "MP"]:
            if method == "OMP":
                recon, _, _, _ = omp_1d(noisy_signal, gen_dict)
            else:
                recon, _, _, _ = mp_1d(noisy_signal, gen_dict)

            eps = 1e-12
            diff = signal - recon
            mse = np.mean(diff ** 2)
            mae = np.mean(np.abs(diff))
            rmse = np.sqrt(np.maximum(mse, eps))

            signal_power = np.mean(signal ** 2)
            noise_power = np.maximum(mse, eps)

            max_signal_sq = np.maximum(np.max(signal) ** 2, eps)
            psnr = 10 * np.log10(max_signal_sq / noise_power)
            snr = 10 * np.log10(np.maximum(signal_power, eps) / noise_power)

            cosine_sim = np.dot(signal, recon) / (norm(signal) * norm(recon) + eps)
            pearson_corr = np.corrcoef(signal, recon)[0, 1]

            metrics[method]["mse"][i] += mse
            metrics[method]["mae"][i] += mae
            metrics[method]["rmse"][i] += rmse
            metrics[method]["snr"][i] += snr
            metrics[method]["psnr"][i] += psnr
            metrics[method]["cosine"][i] += cosine_sim
            metrics[method]["pearson"][i] += pearson_corr

# Усереднення по спробах
for method in ["OMP", "MP"]:
    for key in metrics[method]:
        metrics[method][key] /= n_trials

# Вивід результатів
header = f"{'Метод':<6} {'Словник':<10} {'MSE':>10} {'MAE':>10} {'RMSE':>10} {'SNR(dB)':>10} {'PSNR(dB)':>10} {'CosSim':>10} {'Pearson':>10}"
print("\nОцінка якості реконструкції сигналів за словниками (з шумом)")
print(header)
print("-" * len(header))

for method in ["OMP", "MP"]:
    for i in range(4):
        row = f"{method:<6} {dicts_name[i]:<10} " \
              f"{metrics[method]['mse'][i]:10.4f} " \
              f"{metrics[method]['mae'][i]:10.4f} " \
              f"{metrics[method]['rmse'][i]:10.4f} " \
              f"{metrics[method]['snr'][i]:10.4f} " \
              f"{metrics[method]['psnr'][i]:10.4f} " \
              f"{metrics[method]['cosine'][i]:10.4f} " \
              f"{metrics[method]['pearson'][i]:10.4f}"
        print(row)