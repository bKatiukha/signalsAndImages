import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf


def read_audio(file_path: str):
    """
    Зчитування аудіофайлу та перетворення на моно.
    :param file_path: шлях до аудіофайлу
    :return: сигнал та частота дискретизації
    """
    signal, sample_rate = sf.read(file_path)
    if signal.ndim > 1:  # Перетворення у моно
        signal = np.mean(signal, axis=1)
    return signal, sample_rate


def compute_mfcc(signal: np.ndarray, sample_rate: int, n_mfcc: int = 13, frame_length: float = 0.025, frame_stride: float = 0.01):
    """
    Розрахунок мел-кепстральних коефіцієнтів для аудіосигналу.
    :param signal: аудіосигнал
    :param sample_rate: частота дискретизації
    :param n_mfcc: кількість MFCC коефіцієнтів
    :param frame_length: довжина вікна фрейма у секундах
    :param frame_stride: крок фрейма у секундах
    :return: масив MFCC коефіцієнтів
    """
    # Розрахунок MFCC
    mfcc_features = librosa.feature.mfcc(
        y=signal,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=int(frame_length * sample_rate),
        hop_length=int(frame_stride * sample_rate)
    )
    return mfcc_features


def plot_mfcc(mfcc_features: np.ndarray, sample_rate: int):
    """
    Відображення MFCC коефіцієнтів як спектрограми.
    :param mfcc_features: масив MFCC коефіцієнтів
    :param sample_rate: частота дискретизації
    """
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc_features, sr=sample_rate, x_axis='time')
    plt.colorbar()
    plt.title('Мел-кепстральні коефіцієнти (MFCC)')
    plt.tight_layout()
    plt.show()


def run_lab2():
    audio_file_path = 'sound2.wav'
    try:
        # Зчитування сигналу
        signal, sample_rate = read_audio(audio_file_path)
        print(f"Частота дискретизації: {sample_rate} Гц")

        # Розрахунок MFCC
        mfcc_features = compute_mfcc(signal, sample_rate)
        print(f"MFCC форма: {mfcc_features.shape}")

        # Відображення MFCC
        plot_mfcc(mfcc_features, sample_rate)

    except Exception as e:
        print(f"Помилка: {e}")
