import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

# Path ke file CSV yang berisi data gambar
csv_path = 'C:\\Users\\user\\Downloads\\archive (2)\\A_Z Handwritten Data.csv'  # Ganti dengan path sebenarnya

def detect_letter_a(image_data):
    """
    Memproses gambar dari data dan mendeteksi apakah huruf 'a' ada dalam gambar tersebut.

    Args:
    - image_data: Data gambar dalam bentuk array satu dimensi.

    Returns:
    - is_letter_a: Boolean yang menunjukkan apakah huruf 'a' terdeteksi atau tidak.
    """
    # Mengubah data gambar menjadi array 2D (28x28) dan tipe uint8
    image = image_data.reshape((28, 28)).astype(np.uint8)

    # Terapkan Gaussian blur untuk menghaluskan gambar
    image_blur = cv2.GaussianBlur(image, (5, 5), 0)

    # Terapkan deteksi tepi Canny
    edges = cv2.Canny(image_blur, 50, 150, apertureSize=3)

    # Cari kontur di gambar
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Tentukan apakah huruf 'a' terdeteksi berdasarkan karakteristik kontur
    is_letter_a = False

    for contour in contours:
        # Approximate the contour to reduce the number of points
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Filter berdasarkan ukuran dan bentuk kontur
        if len(approx) > 5:  # Mengasumsikan 'a' memiliki kontur lebih dari 5 sudut
            area = cv2.contourArea(contour)
            if 50 < area < 200:  # Mengasumsikan area kontur huruf 'a' dalam rentang ini
                is_letter_a = True
                break

    return is_letter_a

def process_image(image_data):
    """
    Memproses gambar dari data dan mendeteksi garis vertikal dan horizontal menggunakan transformasi Hough.

    Args:
    - image_data: Data gambar dalam bentuk array satu dimensi.

    Returns:
    - vertical_lines: Daftar garis vertikal yang terdeteksi.
    - horizontal_lines: Daftar garis horizontal yang terdeteksi.
    """
    # Mengubah data gambar menjadi array 2D (28x28) dan tipe uint8
    image = image_data.reshape((28, 28)).astype(np.uint8)

    # Terapkan Gaussian blur untuk menghaluskan gambar
    image_blur = cv2.GaussianBlur(image, (5, 5), 0)

    # Terapkan deteksi tepi Canny
    edges = cv2.Canny(image_blur, 50, 150, apertureSize=3)

    # Terapkan Transformasi Hough Line
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 50)

    vertical_lines = []
    horizontal_lines = []

    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            # Klasifikasikan garis sebagai vertikal atau horizontal
            if abs(a) < 0.1:  # Garis vertikal
                vertical_lines.append((x1, y1, x2, y2))
            elif abs(b) < 0.1:  # Garis horizontal
                horizontal_lines.append((x1, y1, x2, y2))

    return vertical_lines, horizontal_lines

# Verifikasi path file CSV
if os.path.exists(csv_path):
    # Membaca file CSV
    df = pd.read_csv(csv_path, header=None)

    # Mengasumsikan kolom pertama adalah label dan sisanya adalah data gambar
    for index, row in df.iloc[1:51].iterrows():  # Batasan dari baris 1 hingga 50
        # Abaikan label (kolom pertama) dan ambil data gambar
        image_data = row[1:].values

        # Deteksi huruf 'a'
        is_letter_a = detect_letter_a(image_data)

        # Proses gambar untuk mendeteksi garis vertikal dan horizontal
        vertical_lines, horizontal_lines = process_image(image_data)

        # Tampilkan output deteksi huruf 'a'
        if is_letter_a:
            print(f"'a' terdeteksi di gambar: {index}")
        else:
            print(f"'a' tidak terdeteksi di gambar: {index}")

        # Tampilkan gambar dengan garis-garis menggunakan matplotlib
        plt.imshow(image_data.reshape((28, 28)), cmap='gray')
        plt.title(f'Detected Lines in image {index}')
        plt.show()

        # Tambahkan penundaan antara permintaan untuk mengurangi beban server
        time.sleep(5)  # Tunda selama 5 detik

else:
    print(f"File tidak ada: {csv_path}")