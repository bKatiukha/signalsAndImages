# import cv2
# import numpy as np
# from tkinter import filedialog
# from tkinter import Tk
# import os
#
# # Функція для вибору зображення через діалогове вікно
# def select_image():
#     root = Tk()
#     root.withdraw()  # Приховати головне вікно Tkinter
#     file_path = filedialog.askopenfilename(
#         title="Виберіть файл зображення",
#         filetypes=[("Файли зображень", "*.jpg;*.jpeg;*.png;*.bmp")]
#     )
#     return file_path
#
# # Функція для застосування дискретного косинусного перетворення (DCT) для стиснення
# def dct_compress(image: np.ndarray, block_size: int = 8):
#     # Перетворення зображення у кольорову модель YCrCb
#     ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
#     height, width, _ = ycrcb_image.shape
#
#     # Додавання відступів для кратності розмірів блоку
#     padded_height = (height // block_size + 1) * block_size if height % block_size else height
#     padded_width = (width // block_size + 1) * block_size if width % block_size else width
#     padded_image = np.zeros((padded_height, padded_width, 3), dtype=np.float32)
#     padded_image[:height, :width, :] = ycrcb_image.astype(np.float32)
#
#     # Застосування DCT блоками
#     compressed_image = np.zeros_like(padded_image)
#     for channel in range(3):
#         for i in range(0, padded_height, block_size):
#             for j in range(0, padded_width, block_size):
#                 block = padded_image[i:i + block_size, j:j + block_size, channel]
#                 dct_block = cv2.dct(block - 128)  # Виконання DCT
#                 compressed_image[i:i + block_size, j:j + block_size, channel] = dct_block
#
#     return compressed_image
#
# # Функція для збереження стиснутого зображення у форматі JPEG із заданою якістю
# def save_compressed_image(image_path: str, compressed_image: np.ndarray, output_path: str, quality: int):
#     # Виконання оберненого DCT для реконструкції зображення
#     block_size = 8
#     height, width, _ = compressed_image.shape
#     reconstructed_image = np.zeros_like(compressed_image)
#
#     for channel in range(3):
#         for i in range(0, height, block_size):
#             for j in range(0, width, block_size):
#                 dct_block = compressed_image[i:i + block_size, j:j + block_size, channel]
#                 block = cv2.idct(dct_block) + 128
#                 reconstructed_image[i:i + block_size, j:j + block_size, channel] = block
#
#     # Обрізання зображення до початкового розміру
#     image = reconstructed_image[:height, :width, :].clip(0, 255).astype(np.uint8)
#     image = cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)
#
#     # Збереження у форматі JPEG
#     success = cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
#     if success:
#         print(f"Стиснуте зображення збережено за адресою: {output_path}")
#     else:
#         print("Помилка збереження стиснутого зображення.")
#
#
# # Головний блок виконання програми
# def run_lab4():
#     # Вибір вхідного зображення
#     input_image_path = select_image()
#     if not input_image_path:
#         print("Файл не обрано.")
#         return
#
#     # Завантаження вхідного зображення
#     image = cv2.imread(input_image_path)
#     if image is None:
#         print("Помилка завантаження зображення!")
#         return
#
#     # Запит у користувача рівня якості
#     quality = int(input("Введіть рівень якості стиснення (0-100, більше значення - краща якість): "))
#
#     # Стиснення зображення за допомогою DCT
#     compressed_image = dct_compress(image)
#
#     # Визначення шляху для збереження результату
#     output_image_path = os.path.splitext(input_image_path)[0] + "_dct_compressed.jpg"
#
#     # Збереження стиснутого зображення
#     save_compressed_image(input_image_path, compressed_image, output_image_path, quality)
