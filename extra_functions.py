from O_MP import *

O_MP_function = omp_1d
#O_MP_function = mp_1d


def pad_signal_to_patch_size(x, patch_size):
    """
    Доповнює сигнал нулями в кінці, якщо його довжина не кратна patch_size.

    Параметри:
    -----------
    x : ndarray
        Вхідний сигнал довільної довжини.
    patch_size : int
        Розмір патча, до якого сигнал має бути кратним.

    Повертає:
    -----------
    padded_x : ndarray
        Доповнений сигнал (або оригінал, якщо доповнення не потрібне).
    """
    length = len(x)
    remainder = length % patch_size
    if remainder == 0:
        return x.copy()  # Копіюємо, щоб не змінювати оригінал
    pad_needed = patch_size - remainder
    return np.pad(x, (0, pad_needed), mode='constant', constant_values=0)


def enlarge_signal_from_start(x, desired_length):
    """
    Розширює сигнал шляхом повторення початкових зразків, якщо його довжина менша за desired_length.
    Наприклад, x = [1, 2, 3], desired_length = 5 -> [1, 2, 3, 1, 2].

    Параметри:
    -----------
    x : ndarray
        Вхідний сигнал.
    desired_length : int
        Бажана довжина сигналу.

    Повертає:
    -----------
    enlarged_x : ndarray
        Розширений сигнал.
    """
    x = np.asarray(x)
    current_len = len(x)
    if current_len >= desired_length:
        return x[:desired_length].copy()  # Обрізаємо, якщо довший
    out = x.copy()
    while len(out) < desired_length:
        needed = desired_length - len(out)
        chunk = x[:min(needed, current_len)]  # Беремо потрібну кількість з початку
        out = np.concatenate([out, chunk])
    return out


def o_mp_denoise_1d(signal_noisy, D, patch_size=16, max_iter=10, tol=1e-3):
    """
    Видаляє шум із 1D-сигналу, розбиваючи його на патчі та застосовуючи O_MP_function.

    Параметри:
    -----------
    signal_noisy : ndarray, shape (N,)
        За шумлений сигнал.
    D : ndarray, shape (patch_size, K)
        Словник для апроксимації патчів.
    patch_size : int, optional (default=16)
        Розмір патча.
    max_iter : int, optional (default=10)
        Максимальна кількість ітерацій для O_MP_function.
    tol : float, optional (default=1e-3)
        Поріг норми залишку для зупинки.

    Повертає:
    -----------
    signal_denoised : ndarray, shape (N,)
        Відновлений (денойзений) сигнал.
    """
    N = len(signal_noisy)
    n_patches = N // patch_size
    signal_denoised = np.zeros(N, dtype=float)

    # Обробка повних патчів
    for i in range(n_patches):
        start = i * patch_size
        end = start + patch_size
        patch = signal_noisy[start:end]
        patch_approx, _, _, _ = O_MP_function(patch, D, max_iter=max_iter, tol=tol)
        signal_denoised[start:end] = patch_approx

    # Обробка залишку (хвостика)
    remainder = N % patch_size
    if remainder > 0:
        patch = signal_noisy[-remainder:]
        patch_approx, _, _, _ = O_MP_function(patch, D, max_iter=max_iter, tol=tol)
        signal_denoised[-remainder:] = patch_approx

    return signal_denoised


def o_mp_compress_1d(signal, D, patch_size=16, max_atoms=5):
    """
    Компресія 1D-сигналу з обмеженою кількістю атомів для кожного патча.

    Параметри:
    -----------
    signal : ndarray, shape (N,)
        Вхідний сигнал для компресії.
    D : ndarray, shape (patch_size, K)
        Словник для апроксимації патчів.
    patch_size : int, optional (default=16)
        Розмір патча.
    max_atoms : int, optional (default=5)
        Максимальна кількість атомів для компресії одного патча.

    Повертає:
    -----------
    signal_compressed : ndarray, shape (N,)
        Стиснений і відновлений сигнал.
    compression_data : list of tuples
        Список (coeffs, chosen_atoms) для кожного патча.
    """
    N = len(signal)
    n_patches = N // patch_size
    signal_compressed = np.zeros(N, dtype=float)
    compression_data = []

    # Обробка повних патчів
    for i in range(n_patches):
        start = i * patch_size
        end = start + patch_size
        patch = signal[start:end]
        residual = patch.copy()
        chosen_atoms = []
        coeffs = np.zeros(D.shape[1], dtype=float)

        for _ in range(max_atoms):
            corr = D.T @ residual
            i_best = np.argmax(np.abs(corr))
            if i_best not in chosen_atoms:
                chosen_atoms.append(i_best)
            D_sub = D[:, chosen_atoms]
            A = D_sub.T @ D_sub
            b = D_sub.T @ patch
            alpha_sub = np.linalg.inv(A) @ b
            coeffs[:] = 0
            for idx, c_atom in enumerate(chosen_atoms):
                coeffs[c_atom] = alpha_sub[idx]
            residual = patch - D_sub @ alpha_sub

        compression_data.append((coeffs.copy(), chosen_atoms.copy()))
        signal_compressed[start:end] = D @ coeffs

    # Обробка хвостика
    remainder = N % patch_size
    if remainder > 0:
        start = N - remainder
        patch = signal[start:]
        residual = patch.copy()
        chosen_atoms = []
        coeffs = np.zeros(D.shape[1], dtype=float)
        for _ in range(max_atoms):
            corr = D.T @ residual
            i_best = np.argmax(np.abs(corr))
            if i_best not in chosen_atoms:
                chosen_atoms.append(i_best)
            D_sub = D[:, chosen_atoms]
            A = D_sub.T @ D_sub
            b = D_sub.T @ patch
            alpha_sub = np.linalg.inv(A) @ b
            coeffs[:] = 0
            for idx, c_atom in enumerate(chosen_atoms):
                coeffs[c_atom] = alpha_sub[idx]
            residual = patch - D_sub @ alpha_sub
        compression_data.append((coeffs.copy(), chosen_atoms.copy()))
        signal_compressed[start:] = D @ coeffs

    return signal_compressed, compression_data


def o_mp_inpaint_1d(signal_damaged, mask, D, patch_size=16, max_iter=10, tol=1e-3):
    """
    Заповнення пропущених даних у 1D-сигналі (інпейнтинг) з урахуванням відомих зразків.

    Параметри:
    -----------
    signal_damaged : ndarray, shape (N,)
        Пошкоджений сигнал із пропущеними зразками.
    mask : ndarray, shape (N,)
        Маска: 1 — відомий зразок, 0 — пропущений.
    D : ndarray, shape (patch_size, K)
        Словник для апроксимації патчів.
    patch_size : int, optional (default=16)
        Розмір патча.
    max_iter : int, optional (default=10)
        Максимальна кількість ітерацій.
    tol : float, optional (default=1e-3)
        Поріг норми залишку.

    Повертає:
    -----------
    signal_inpainted : ndarray, shape (N,)
        Відновлений сигнал.
    """
    N = len(signal_damaged)
    n_patches = N // patch_size
    signal_inpainted = np.zeros(N, dtype=float)

    # Обробка повних патчів
    for i in range(n_patches):
        start = i * patch_size
        end = start + patch_size
        patch = signal_damaged[start:end]
        patch_mask = mask[start:end]
        known_idx = np.where(patch_mask == 1)[0]
        D_reduced = D[known_idx, :]
        patch_known = patch[known_idx]

        residual = patch_known.copy()
        chosen_atoms = []
        coeffs = np.zeros(D.shape[1], dtype=float)
        for _ in range(max_iter):
            corr = D_reduced.T @ residual
            i_best = np.argmax(np.abs(corr))
            if i_best not in chosen_atoms:
                chosen_atoms.append(i_best)
            D_sub = D_reduced[:, chosen_atoms]
            A = D_sub.T @ D_sub
            b = D_sub.T @ patch_known
            alpha_sub = np.linalg.inv(A) @ b
            coeffs[:] = 0
            for idx, c_atom in enumerate(chosen_atoms):
                coeffs[c_atom] = alpha_sub[idx]
            residual = patch_known - D_sub @ alpha_sub
            if np.linalg.norm(residual) < tol:
                break
        signal_inpainted[start:end] = D @ coeffs

    # Обробка хвостика
    remainder = N % patch_size
    if remainder > 0:
        start = N - remainder
        patch = signal_damaged[start:]
        patch_mask = mask[start:]
        known_idx = np.where(patch_mask == 1)[0]
        D_reduced = D[known_idx, :]
        patch_known = patch[known_idx]

        residual = patch_known.copy()
        chosen_atoms = []
        coeffs = np.zeros(D.shape[1], dtype=float)
        for _ in range(max_iter):
            corr = D_reduced.T @ residual
            i_best = np.argmax(np.abs(corr))
            if i_best not in chosen_atoms:
                chosen_atoms.append(i_best)
            D_sub = D_reduced[:, chosen_atoms]
            A = D_sub.T @ D_sub
            b = D_sub.T @ patch_known
            alpha_sub = np.linalg.inv(A) @ b
            coeffs[:] = 0
            for idx, c_atom in enumerate(chosen_atoms):
                coeffs[c_atom] = alpha_sub[idx]
            residual = patch_known - D_sub @ alpha_sub
            if np.linalg.norm(residual) < tol:
                break
        signal_inpainted[start:] = D @ coeffs

    return signal_inpainted


def o_mp_superres_1d(signal_lowres, scale_factor, D_highres, patch_size=16, max_iter=10, tol=1e-3):
    """
    Підвищення роздільної здатності сигналу з використанням zero-stuffing і O_MP_function.

    Параметри:
    -----------
    signal_lowres : ndarray, shape (N_low,)
        Сигнал із низькою роздільною здатністю.
    scale_factor : int
        Коефіцієнт підвищення роздільної здатності.
    D_highres : ndarray, shape (patch_size * scale_factor, K)
        Словник для високої роздільної здатності.
    patch_size : int, optional (default=16)
        Розмір патча у низькій роздільності.
    max_iter : int, optional (default=10)
        Максимальна кількість ітерацій.
    tol : float, optional (default=1e-3)
        Поріг норми залишку.

    Повертає:
    -----------
    signal_highres : ndarray, shape (N_high,)
        Сигнал із підвищеною роздільною здатністю.
    """
    N_low = len(signal_lowres)
    N_high = N_low * scale_factor
    n_patches = N_low // patch_size
    signal_highres = np.zeros(N_high, dtype=float)

    # Обробка повних патчів
    for i in range(n_patches):
        start_low = i * patch_size
        end_low = start_low + patch_size
        patch_low = signal_lowres[start_low:end_low]
        patch_high_len = patch_size * scale_factor
        patch_approx_init = np.zeros(patch_high_len)
        patch_approx_init[::scale_factor] = patch_low  # Zero-stuffing
        patch_high_approx, _, _, _ = O_MP_function(patch_approx_init, D_highres, max_iter=max_iter, tol=tol)
        start_high = start_low * scale_factor
        end_high = start_high + patch_high_len
        signal_highres[start_high:end_high] = patch_high_approx

    # Обробка хвостика
    remainder = N_low % patch_size
    if remainder > 0:
        start_low = N_low - remainder
        patch_low = signal_lowres[start_low:]
        patch_high_len = remainder * scale_factor
        patch_approx_init = np.zeros(patch_high_len)
        patch_approx_init[::scale_factor] = patch_low
        patch_high_approx, _, _, _ = O_MP_function(patch_approx_init, D_highres, max_iter=max_iter, tol=tol)
        start_high = (N_low - remainder) * scale_factor
        signal_highres[start_high:start_high + patch_high_len] = patch_high_approx

    return signal_highres



#випадково створює сигнал на основі відповідного словника
def create_signal_from_dictionary(D, n_atoms_to_pick=5, max_coef=3):


    N, K = D.shape        # Розміри словника: (N, K)

    # Ініціалізуємо вихідний сигнал нулями
    signal = np.zeros(N, dtype=float)

    # Будуть зберігатися параметри (atom_index, coefficient)
    params = [0]*K

    for _ in range(n_atoms_to_pick):
        # 1) Випадковий індекс атома
        atom_idx = np.random.randint(0, K)

        # 2) Випадковий цілий коефіцієнт (уникаємо 0)
        coef = 0
        while coef == 0:
            coef = np.random.randint(-max_coef, max_coef+1)

        # 3) Додаємо атом (довжиною N) у початок сигналу
        signal[:N] += coef * D[:, atom_idx]

        # Зберігаємо інформацію про вибраний атом
        params[atom_idx] += coef

    return signal, params

