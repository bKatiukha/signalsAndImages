import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import get_window
from scipy.fft import fft
import os


def read_audio(file_path: str):
    """
    Зчитує аудіо файл.
    :param file_path: шлях до аудіофайлу
    :return: сигнал та частота дискретизації
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError("Файл не знайдено!")

    signal, sample_rate = sf.read(file_path)
    if signal.ndim > 1:  # Перетворюємо у моно шляхом усереднення каналів
        signal = np.mean(signal, axis=1)

    return signal, sample_rate


def fragment_signal(signal: np.ndarray, sample_rate: int, window_size: float = 0.02, overlap: float = 0.5):
    """
    Фрагментація сигналу на вікна із перекриттям.
    :param signal: аудіосигнал
    :param sample_rate: частота дискретизації
    :param window_size: тривалість одного фрагменту (у секундах)
    :param overlap: частка перекриття (0.0 - без перекриття, 0.5 - 50% перекриття)
    :return: список фрагментів
    """
    win_length = int(window_size * sample_rate)
    step = int(win_length * (1 - overlap))
    fragments = [signal[i:i + win_length] for i in range(0, len(signal) - win_length + 1, step)]

    return np.array(fragments)


def fft_manual(x: np.ndarray):
    """
    test
    Обчислює Швидке Перетворення Фур'є (FFT) для входу x.
    :param x: сигнал (масив комплексних або дійсних чисел)
    :return: результат FFT
    """
    N = len(x)

    # Якщо довжина не є ступенем двійки, доповнюємо до найближчого ступеня двійки
    if not np.log2(N).is_integer():
        N_next_power_of_2 = 2 ** int(np.ceil(np.log2(N)))
        x = np.pad(x, (0, N_next_power_of_2 - N), mode='constant')

    N = len(x)  # Тепер N - це ступінь двійки
    if N <= 1:
        return x

    # Розділяємо на парні і непарні індекси
    even = fft_manual(x[::2])
    odd = fft_manual(x[1::2])

    # Обчислюємо відповідні індекси для комбінування
    twiddle_factors = np.exp(-2j * np.pi * np.arange(N) / N)

    # Комбінуємо парні та непарні
    fft_result = np.concatenate([
        even + twiddle_factors[:N // 2] * odd,
        even + twiddle_factors[N // 2:] * odd
    ])

    return fft_result


def compute_fft(fragments: np.ndarray, sample_rate: int, window_type: str = 'hann'):
    """
    Обчислює Швидке Перетворення Фур'є для кожного фрагменту.
    :param fragments: фрагменти сигналу
    :param sample_rate: частота дискретизації
    :param window_type: тип віконної функції
    :return: спектри та частоти
    """
    window = get_window(window_type, fragments.shape[1])
    fft_results = []
    freqs = np.fft.fftfreq(fragments.shape[1], d=1 / sample_rate)

    for fragment in fragments:
        fragment_windowed = fragment * window
        # fft_result = np.abs(fft(fragment_windowed))[:len(freqs) // 2]
        fft_result = np.abs(fft_manual(fragment_windowed))[:len(freqs) // 2]
        fft_results.append(fft_result)

    freqs = freqs[:len(freqs) // 2]
    return np.array(fft_results), freqs


def plot_fft_spectrogram(fft_results: np.ndarray, freqs: np.ndarray, sample_rate: int, window_size: float):
    """
    Відображає спектрограму результатів Фур'є.
    :param fft_results: матриця спектрів
    :param freqs: частоти
    :param sample_rate: частота дискретизації
    :param window_size: розмір вікна фрагментації
    """
    plt.figure(figsize=(12, 6))
    plt.imshow(20 * np.log10(np.maximum(fft_results.T, 1e-10)), aspect='auto',
               extent=[0, fft_results.shape[0] * window_size, freqs[0], freqs[-1]],
               origin='lower', cmap='jet')
    plt.colorbar(label='Амплітуда (дБ)')
    plt.title('Спектрограма сигналу')
    plt.xlabel('Час (с)')
    plt.ylabel('Частота (Гц)')
    plt.show()


def run_lab1():
    audio_file_path = 'sound2.wav'

    try:
        signal, sample_rate = read_audio(audio_file_path)

        print(f"Частота дискретизації: {sample_rate} Гц")

        # Налаштування параметрів
        window_size_sec = 0.02  # Розмір вікна (20 мс)
        overlap_fraction = 0.5  # Перекриття 50%

        # Фрагментація сигналу
        fragments = fragment_signal(signal, sample_rate, window_size=window_size_sec, overlap=overlap_fraction)
        print(f"Кількість фрагментів: {len(fragments)}")

        # Обчислення спектрів Фур'є
        fft_results, freqs = compute_fft(fragments, sample_rate)

        # Відображення спектрограми
        plot_fft_spectrogram(fft_results, freqs, sample_rate, window_size=window_size_sec)

    except Exception as e:
        print(f"Помилка: {e}")
