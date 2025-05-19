from math import log10
from numpy.linalg import norm
from dictionaries import *
from extra_functions import *

dicts_name = ["gauss", "meyer", "poly", "haar"]
patch_size = 512
n_atoms = 63
n_trials = 1000

# Initialize 3x3 matrices for each metric
mse_matrix = np.zeros((4, 4))
mae_matrix = np.zeros((4, 4))
rmse_matrix = np.zeros((4, 4))
snr_matrix = np.zeros((4, 4))
psnr_matrix = np.zeros((4, 4))
cosine_matrix = np.zeros((4, 4))
pearson_matrix = np.zeros((4, 4))

# Create all 4 dictionaries
D_gauss, _ = create_gauss_dictionary(patch_size, n_atoms)
D_meyer, _ = create_meyer_wavelet_dictionary(patch_size, n_atoms)
D_poly, _ = create_polynomial_dictionary(patch_size, n_atoms)
D_haar, _ = create_haar_wavelet_dictionary(patch_size, n_atoms)

dicts = [D_gauss, D_meyer, D_poly, D_haar]

for _ in range(n_trials):
    # Generate signals using each dictionary
    for i, gen_dict in enumerate(dicts):
        signal, _ = create_signal_from_dictionary(gen_dict)

        for j, recon_dict in enumerate(dicts):
            recon, _, _, _ = omp_1d(signal, recon_dict)

            eps = 1e-12
            diff = signal - recon
            mse = np.mean(diff ** 2)
            mae = np.mean(np.abs(diff))
            rmse = np.sqrt(mse)
            signal_power = np.mean(signal ** 2)
            noise_power = max(mse, eps)
            psnr = 10 * np.log10(max(np.max(signal) ** 2, eps) / noise_power)
            snr = 10 * np.log10((signal_power + eps) / noise_power)
            cosine_sim = np.dot(signal, recon) / (norm(signal) * norm(recon) + eps)
            pearson_corr = np.corrcoef(signal, recon)[0, 1]

            mse_matrix[i, j] += mse
            mae_matrix[i, j] += mae
            rmse_matrix[i, j] += rmse
            snr_matrix[i, j] += snr
            psnr_matrix[i, j] += psnr
            cosine_matrix[i, j] += cosine_sim
            pearson_matrix[i, j] += pearson_corr

# Average each matrix over all trials
mse_matrix /= n_trials
mae_matrix /= n_trials
rmse_matrix /= n_trials
snr_matrix /= n_trials
psnr_matrix /= n_trials
cosine_matrix /= n_trials
pearson_matrix /= n_trials

# Display results
def print_table(title, matrix, unit=""):
    print(f"\n{title} ({unit})" if unit else f"\n{title}")
    header = "".ljust(15) + " ".join([f"{name:>12}" for name in dicts_name])
    print(header)
    print("-" * len(header))
    for i, row in enumerate(matrix):
        row_str = f"{dicts_name[i]:<15}" + " ".join([f"{val:12.4f}" for val in row])
        print(row_str)
    print("")

# Виводимо всі метрики
print_table("MSE", mse_matrix)
print_table("MAE", mae_matrix)
print_table("RMSE", rmse_matrix)
print_table("SNR", snr_matrix, "dB")
print_table("PSNR", psnr_matrix, "dB")
print_table("Cosine Similarity", cosine_matrix)
print_table("Pearson Correlation", pearson_matrix)








"""
now please write  5.3. Ситуація “неправильного” словника
here wa used only omp, because  here i payed more attention to dictionaries, not to algorithms
"""