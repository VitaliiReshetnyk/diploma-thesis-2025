import numpy as np


def mp_1d(patch, D, max_iter=10, tol=1e-6):
    """
    Алгоритм Matching Pursuit (MP) для одновимірного патча сигналу.

    Параметри:
    -----------
    patch : ndarray, shape (N,)
        Вхідний сигнал або його фрагмент (патч), який потрібно апроксимувати.
    D : ndarray, shape (N, K)
        Матриця словника, де N — довжина патча, K — кількість атомів (стовпців).
    max_iter : int, optional (default=10)
        Максимальна кількість ітерацій для пошуку атомів.
    tol : float, optional (default=1e-6)
        Поріг норми залишку, при досягненні якого алгоритм зупиняється.

    Повертає:
    -----------
    patch_approx : ndarray, shape (N,)
        Апроксимований патч як лінійна комбінація вибраних атомів словника.
    coeffs : ndarray, shape (K,)
        Вектор коефіцієнтів, де ненульові значення відповідають обраним атомам.
    chosen_atoms : list of int
        Список індексів атомів із словника D, які були вибрані під час ітерацій.
    residual : ndarray, shape (N,)
        Кінцевий залишок, що дорівнює різниці між початковим патчем і апроксимацією.
    """
    # Створюємо копію вхідного сигналу, щоб не змінювати оригінал
    x = patch.copy()

    # Отримуємо розміри матриці словника: N — довжина сигналу, K — кількість атомів
    patch_size, K = D.shape

    # Ініціалізуємо залишок як повний вхідний сигнал
    residual = x.copy()

    # Ініціалізуємо вектор коефіцієнтів нулями для всіх K атомів
    coeffs = np.zeros(K, dtype=float)

    # Ініціалізуємо список для збереження індексів обраних атомів
    chosen_atoms = []

    # Починаємо ітераційний процес, обмежений max_iter
    for iter_idx in range(max_iter):
        # 1) Обчислюємо кореляції між поточним залишком і кожним атомом словника
        #    D.T @ residual — це скалярний добуток, результат розміром (K,)
        #    corr[j] — кореляція залишку з j-м атомом D[:, j]
        corr = D.T @ residual

        # 2) Знаходимо індекс атома з максимальною абсолютною кореляцією
        #    np.abs(corr) гарантує вибір за модулем, незалежно від знака
        i_best = np.argmax(np.abs(corr))

        # 3) Визначаємо коефіцієнт для обраного атома
        #    alpha — це проєкція залишку на напрямок D[:, i_best]
        alpha = corr[i_best]

        # 4) Оновлюємо залишок, віднімаючи внесок обраного атома
        #    residual = residual - alpha * D[:, i_best] зменшує залишок
        #    на величину, пропорційну вибраному атому
        residual -= alpha * D[:, i_best]

        # 5) Додаємо коефіцієнт до вектора coeffs
        #    Один атом може бути обраний кілька разів, тому += накопичує внесок
        coeffs[i_best] += alpha

        # Зберігаємо індекс обраного атома в список
        chosen_atoms.append(i_best)

        # 6) Перевіряємо критерій зупинки
        #    Обчислюємо норму залишку (np.linalg.norm — евклідова норма)
        residual_norm = np.linalg.norm(residual)
        if residual_norm < tol:
            # Якщо норма залишку менша за заданий поріг, зупиняємо алгоритм
            break

    # Формуємо апроксимований сигнал як лінійну комбінацію атомів
    # D @ coeffs — матричне множення розміром (N, K) на (K,) дає (N,)
    patch_approx = D @ coeffs

    # Повертаємо результат: апроксимацію, коефіцієнти, обрані атоми та залишок
    return patch_approx, coeffs, chosen_atoms, residual



def omp_1d(patch, D, max_iter=10, tol=1e-6):
    """
    Orthogonal Matching Pursuit для одномірного патча сигналу.

    Параметри:
    -----------
    patch : ndarray, shape (patch_size,)
        Вхідний сигнал або його фрагмент (патч), який потрібно апроксимувати.
    D : ndarray, shape (patch_size, n_atoms)
        Матриця словника, де кожен стовпець — атом довжиною patch_size.
    max_iter : int, optional (default=10)
        Максимальна кількість ітерацій для вибору атомів.
    tol : float, optional (default=1e-6)
        Поріг норми залишку для зупинки алгоритму.

    Повертає:
    -----------
    patch_approx : ndarray, shape (patch_size,)
        Апроксимований патч як лінійна комбінація вибраних атомів.
    coeffs : ndarray, shape (n_atoms,)
        Вектор коефіцієнтів, де ненульові значення відповідають обраним атомам.
    chosen_atoms : list of int
        Список індексів атомів із словника D, вибраних під час ітерацій.
    residual : ndarray, shape (patch_size,)
        Кінцевий залишок, що дорівнює різниці між початковим патчем і апроксимацією.
    """
    # Створюємо копію вхідного патча, щоб не змінювати оригінальні дані
    x = patch.copy()

    # Отримуємо розміри матриці словника: patch_size — довжина патча, K — кількість атомів
    patch_size, K = D.shape

    # Ініціалізуємо залишок як повний вхідний сигнал
    residual = x.copy()

    # Ініціалізуємо список для збереження індексів обраних атомів
    chosen_atoms = []

    # Ініціалізуємо вектор коефіцієнтів нулями для всіх K атомів
    coeffs = np.zeros(K, dtype=float)

    # Починаємо ітераційний процес, обмежений max_iter
    for iter_idx in range(max_iter):
        # 1) Обчислюємо кореляції між поточним залишком і всіма атомами словника
        #    D.T @ residual — скалярний добуток, результат розміром (K,)
        #    corr[j] — кореляція залишку з j-м атомом D[:, j]
        corr = D.T @ residual

        # 2) Знаходимо індекс атома з максимальною абсолютною кореляцією
        #    np.abs(corr) враховує модуль, щоб вибрати найбільший внесок незалежно від знака
        i_best = np.argmax(np.abs(corr))

        # 3) Додаємо атом до списку, якщо він ще не був обраний
        if i_best not in chosen_atoms:
            chosen_atoms.append(i_best)

        # 4) Формуємо підсловник D_sub із вибраних атомів
        #    D_sub має розмір (patch_size, len(chosen_atoms))
        D_sub = D[:, chosen_atoms]

        # 5) Розв’язуємо задачу найменших квадратів для підсловника
        #    Знаходимо alpha_sub, що мінімізує ||x - D_sub @ alpha_sub||^2
        #    alpha_sub = (D_sub^T D_sub)^{-1} D_sub^T x
        A = D_sub.T @ D_sub  # Матриця розміром (len(chosen_atoms), len(chosen_atoms))
        b = D_sub.T @ x  # Вектор розміром (len(chosen_atoms),)
        alpha_sub = np.linalg.inv(A) @ b  # Розв’язок системи рівнянь

        # 6) Оновлюємо глобальний вектор коефіцієнтів
        #    Спочатку обнуляємо всі коефіцієнти
        coeffs[:] = 0
        #    Заповнюємо ненульові значення для вибраних атомів
        for idx, c_atom in enumerate(chosen_atoms):
            coeffs[c_atom] = alpha_sub[idx]

        # 7) Оновлюємо залишок як різницю між оригіналом і апроксимацією
        #    residual = x - D_sub @ alpha_sub
        residual = x - D_sub @ alpha_sub

        # 8) Перевіряємо критерій зупинки
        #    Обчислюємо евклідову норму залишку
        residual_norm = np.linalg.norm(residual)
        if residual_norm < tol:
            # Якщо норма залишку менша за поріг, завершуємо ітерації
            break

    # 9) Формуємо остаточну апроксимацію патча
    #    D @ coeffs — матричне множення (patch_size, K) на (K,) дає (patch_size,)
    patch_approx = D @ coeffs

    # Повертаємо результат: апроксимацію, коефіцієнти, обрані атоми та залишок
    return patch_approx, coeffs, chosen_atoms, residual
