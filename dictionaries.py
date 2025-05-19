import numpy as np

def shift(arr, n_atoms, desired_size):
    """
    Створює атоми шляхом зсуву материнської функції, гарантуючи правильне вікно.

    Параметри:
    -----------
    arr : ndarray
        Материнська функція.
    n_atoms : int
        Кількість атомів.
    desired_size : int
        Бажана довжина кожного атома.

    Повертає:
    -----------
    atoms : list
        Список атомів.
    """
    size = len(arr)
    atoms = []
    max_start = size - desired_size

    if n_atoms == 1:
        positions = [max_start // 2]
    else:
        positions = np.linspace(0, max_start, n_atoms, dtype=int)

    for start in positions:
        end = start + desired_size
        atom = arr[start:end]
        atoms.append(atom)
    return atoms


# Функція для створення материнського Мейєр-вейвлета
def create_meyer_mother_wavelet(patch_size, time_support=1):
    """
    Створює материнський Мейєр-вейвлет у часовій області.

    Параметри:
    -----------
    patch_size : int
        Кількість дискретних точок (N).
    time_support : float, optional (default=1)
        Ширина часової підтримки вейвлета, 1 для інтервалу [-0.5, 0.5].

    Повертає:
    -----------
    psi : ndarray
        Материнський Мейєр-вейвлет у часовій області.
    t : ndarray
        Часова вісь.
    """
    freq = np.fft.fftfreq(patch_size) * patch_size
    dt = time_support / patch_size
    w = 2 * np.pi * freq * dt
    psi_hat = np.zeros(patch_size, dtype=complex)
    nu = lambda x: np.where(x <= 0, 0, np.where(x >= 1, 1, x ** 4 * (35 - 84 * x + 70 * x ** 2 - 20 * x ** 3)))
    for i, omega in enumerate(w):
        abs_omega = np.abs(omega)
        if 2 * np.pi / 3 <= abs_omega <= 4 * np.pi / 3:
            psi_hat[i] = (1 / np.sqrt(2 * np.pi)) * np.exp(-1j * omega / 2) * np.sin(
                np.pi / 2 * nu(3 * abs_omega / (2 * np.pi) - 1))
        elif 4 * np.pi / 3 <= abs_omega <= 8 * np.pi / 3:
            psi_hat[i] = (1 / np.sqrt(2 * np.pi)) * np.exp(-1j * omega / 2) * np.cos(
                np.pi / 2 * nu(3 * abs_omega / (4 * np.pi) - 1))
    psi = np.fft.ifftshift(np.fft.ifft(psi_hat))
    psi = np.real(psi)
    psi /= np.linalg.norm(psi)
    t = np.linspace(-time_support / 2, time_support / 2, patch_size)
    return psi, t

# Мейєр-вейвлет словник із заданою кількістю атомів
def create_meyer_wavelet_dictionary(patch_size, n_atoms, time_support=1):
    """
    Створює словник Мейєр-вейвлетів із заданою кількістю атомів шляхом зсуву.

    Параметри:
    -----------
    patch_size : int
        Довжина патча сигналу (кількість дискретних точок, N).
    n_atoms : int
        Кількість атомів у словнику.
    time_support : float, optional (default=1)
        Ширина часової підтримки вейвлета.

    Повертає:
    -----------
    D : ndarray, shape (patch_size, n_atoms)
        Матриця словника.
    t : ndarray
        Часова вісь для візуалізації.
    """
    # Створюємо материнський вейвлет із розміром у 3 рази більшим
    big_patch_size = 3 * patch_size
    psi_big, _ = create_meyer_mother_wavelet(big_patch_size, time_support)
    t = np.linspace(-time_support / 2, time_support / 2, patch_size)

    # Створюємо атоми шляхом зсуву
    atoms = shift(psi_big, n_atoms, patch_size)
    D = np.zeros((patch_size, n_atoms), dtype=float)
    for i in range(n_atoms):
        D[:, i] = atoms[i]

    norms = np.linalg.norm(D, axis=0, keepdims=True)
    norms[norms == 0] = 1e-10
    D /= norms
    return D, t


# Гаусів словник із заданою кількістю атомів
def create_gauss_dictionary(patch_size, n_atoms, sigma=0.1):
    """
    Створює Гаусів словник із заданою кількістю атомів шляхом зсуву.

    Параметри:
    -----------
    patch_size : int
        Довжина патча сигналу (кількість дискретних точок, N).
    n_atoms : int
        Кількість атомів у словнику.
    sigma : float, optional (default=0.1)
        Ширина Гаусіани.

    Повертає:
    -----------
    D : ndarray, shape (patch_size, n_atoms)
        Матриця словника.
    x : ndarray
        Вісь для візуалізації.
    """
    # Створюємо материнську Гаусіану з розміром у 3 рази більшим
    big_patch_size = 3 * patch_size
    x_big = np.linspace(0, 1, big_patch_size)
    mean = 0.5
    gauss_big = np.exp(-((x_big - mean) ** 2) / (2 * sigma ** 2))
    gauss_big /= np.linalg.norm(gauss_big)

    x = np.linspace(0, 1, patch_size)
    atoms = shift(gauss_big, n_atoms, patch_size)
    D = np.zeros((patch_size, n_atoms), dtype=float)
    for i in range(n_atoms):
        D[:, i] = atoms[i]

    norms = np.linalg.norm(D, axis=0, keepdims=True)
    norms[norms == 0] = 1e-12
    D /= norms
    return D, x


# Поліноміальний словник із параметрами
def create_polynomial_dictionary(patch_size, n_atoms, x_min=-1, x_max=1):
    """
    Створює поліноміальний словник із заданою кількістю атомів.

    Параметри:
    -----------
    patch_size : int
        Довжина патча сигналу (кількість дискретних точок, N).
    n_atoms : int
        Кількість атомів у словнику (максимальна степінь полінома + 1).
    x_min : float, optional (default=-1)
        Мінімальне значення осі x.
    x_max : float, optional (default=1)
        Максимальне значення осі x.

    Повертає:
    -----------
    D : ndarray, shape (patch_size, n_atoms)
        Матриця словника.
    x : ndarray
        Вісь для візуалізації.
    """
    x = np.linspace(x_min, x_max, patch_size)
    D = np.zeros((patch_size, n_atoms), dtype=float)
    for i in range(n_atoms):
        poly_atom = x ** i
        D[:, i] = poly_atom
    norms = np.linalg.norm(D, axis=0, keepdims=True)
    norms[norms == 0] = 1e-12
    D /= norms
    return D, x


#функція генерації материнської вейвлети
def mother_haar(length=2):
    """
    Створює материнський Хаар-вейвлет у дискретному вигляді.

    Параметри:
    -----------
    length : int, optional (default=2)
        Довжина материнського вейвлета (наприклад, 2 для [1, -1]).

    Повертає:
    -----------
    arr : ndarray
        Материнський Хаар-вейвлет.
    """
    arr = np.zeros(length, dtype=float)
    half = length // 2
    arr[:half] = 1.0
    arr[half:] = -1.0
    return arr

# Функція для масштабування та зсуву Хаар-вейвлета
def scale_and_shift_haar(psi_haar, j, k, outlen):
    """
    Масштабує та зсуває материнський Хаар-вейвлет.

    Параметри:
    -----------
    psi_haar : ndarray
        Материнський Хаар-вейвлет.
    j : float
        Масштаб.
    k : float
        Зсув.
    outlen : int
        Довжина вихідного сигналу (N).

    Повертає:
    -----------
    wave_jk : ndarray
        Результуючий атом.
    """
    n_arr = np.arange(outlen)
    x_frac = (n_arr - k) / (2.0 ** j)
    mother_len = len(psi_haar)
    t_mother = np.arange(mother_len)
    wave_vals = np.interp(x_frac, t_mother, psi_haar, left=0.0, right=0.0)
    scale_factor = 2.0 ** (-0.5 * j)
    wave_jk = scale_factor * wave_vals
    return wave_jk

# Хаар-вейвлет словник із заданою кількістю атомів
def create_haar_wavelet_dictionary(patch_size, n_atoms, mother_len=8, j_min=0, j_max=3, k_min=0.2, k_max=0.8):
    """
    Створює словник Хаар-вейвлетів із заданою кількістю атомів.

    Параметри:
    -----------
    patch_size : int
        Довжина патча сигналу (кількість дискретних точок, N).
    n_atoms : int
        Кількість атомів у словнику.
    mother_len : int, optional (default=8)
        Довжина материнського Хаар-вейвлета.
    j_min : float, optional (default=0)
        Мінімальний масштаб.
    j_max : float, optional (default=3)
        Максимальний масштаб.
    k_min : float, optional (default=0.2)
        Мінімальний зсув (у нормалізованій шкалі [0, 1]).
    k_max : float, optional (default=0.8)
        Максимальний зсув.

    Повертає:
    -----------
    D : ndarray, shape (patch_size, n_atoms)
        Матриця словника.
    x : ndarray
        Вісь для візуалізації.
    """
    psi_haar = mother_haar(mother_len)
    D = np.zeros((patch_size, n_atoms), dtype=float)
    x = np.linspace(0, 1, patch_size)
    shifts = np.linspace(k_min * patch_size, k_max * patch_size, n_atoms)
    scales = np.random.uniform(j_min, j_max, n_atoms)
    for i in range(n_atoms):
        j = scales[i]
        k = shifts[i]
        atom = scale_and_shift_haar(psi_haar, j, k, patch_size)
        D[:, i] = atom
    norms = np.linalg.norm(D, axis=0, keepdims=True)
    norms[norms == 0] = 1e-10
    D /= norms
    return D, x



