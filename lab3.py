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


def detect_speech_intervals(signal: np.ndarray, sample_rate: int, top_db: float = 30.0):
    """
    Визначення меж пауз і слів у голосовому сигналі.
    :param signal: аудіосигнал
    :param sample_rate: частота дискретизації
    :param top_db: поріг для визначення тиші
    :return: список часових меж активних інтервалів (початок, кінець у секундах)

    """
    intervals = librosa.effects.split(signal, top_db=top_db)
    time_intervals = [(start / sample_rate, end / sample_rate) for start, end in intervals]
    return time_intervals


def plot_speech_intervals(signal: np.ndarray, sample_rate: int, intervals: list):
    """
    Відображення сигналу із позначеними часовими межами слів.
    :param signal: аудіосигнал
    :param sample_rate: частота дискретизації
    :param intervals: часові межі активних інтервалів
change text fix fix again
    """
    plt.figure(figsize=(12, 6))
    time_axis = np.linspace(0, len(signal) / sample_rate, len(signal))
    plt.plot(time_axis, signal, label='Сигнал')

    for start, end in intervals:
        plt.axvspan(start, end, color='green', alpha=0.3, label='Межа слова')

    plt.xlabel('Час (секунди)')
    plt.ylabel('Амплітуда')
    plt.title('Визначення меж слів у голосовому сигналі')
    plt.legend(loc='upper right')
    plt.show()


def run_lab3():
    audio_file_path = 'sound3.wav'
    try:
        # Зчитування сигналу
        signal, sample_rate = read_audio(audio_file_path)
        print(f"Частота дискретизації: {sample_rate} Гц")

        # Визначення меж слів
        speech_intervals = detect_speech_intervals(signal, sample_rate)
        print("Визначені межі слів:")
        for start, end in speech_intervals:
            print(f"Початок: {start:.2f} с, Кінець: {end:.2f} с")

        # Відображення результатів
        plot_speech_intervals(signal, sample_rate, speech_intervals)

    except Exception as e:
        print(f"Помилка: {e}")
