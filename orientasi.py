import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QFileDialog
from PyQt5.uic import loadUi
import matplotlib.pyplot as plt


class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('tubes.ui', self)
        self.image = None
        self.gray_image = None
        self.classifier = None
        self.labelResult = self.findChild(QtWidgets.QLabel, 'labelResult')

        # Hubungkan tombol
        self.buttonHisto.clicked.connect(self.calculate_histogram)
        self.buttonLoad.clicked.connect(self.load_image)
        self.buttonHog.clicked.connect(self.calculate_hog)
        self.buttonPrediction.clicked.connect(self.classify_from_dataset)
        self.buttonCanny.clicked.connect(self.canny_edge_detection)

        # Pemetaan label ke huruf
        self.label_mapping = {i: chr(65 + i) for i in range(26)}
        self.confidence_threshold = 0.5  # Define a threshold for confidence

    def calculate_histogram(self):
        if self.image is not None:
            gray_image = self.grayscale(self.image)
            hist = self._calculate_histogram(gray_image)

            # Convert grayscale image to array
            gray_array = self._grayscale_array(gray_image)

            # Plot histogram
            plt.figure()
            plt.bar(range(len(hist)), hist, color='blue')
            plt.title('Grayscale Histogram')
            plt.show()

            # Display grayscale image
            self.display_image(gray_image, 2)

            # Display 100 array pertama
            print("\nArray Gambar Grayscale:")
            print(gray_array[:100])

    def _calculate_histogram(self, image):
        hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
        return hist

    def _grayscale_array(self, image):
        return image.flatten()

    def load_image(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg)")
        if file_dialog.exec_():
            image_path = file_dialog.selectedFiles()[0]
            self.image = cv2.imread(image_path)
            self.gray_image = self.grayscale(self.image)
            self.display_image(self.image, 1)

    def calculate_hog(self):
        if self.image is not None:
            try:
                gray_image = self.grayscale(self.image)
                hog_features, hog_image = self._calculate_hog(gray_image)
                print("Fitur HOG:", hog_features)
                plt.figure()
                plt.imshow(hog_image, cmap='gray')
                plt.title('Gambar HOG')
                plt.show()
                self.display_image(hog_image, 2)
            except Exception as e:
                print(f"Kesalahan saat menghitung HOG: {e}")

    def _calculate_hog(self, image):
        hog_features, hog_image = hog(image, pixels_per_cell=(8, 8),
                                      cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
        return hog_features, hog_image

    def grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def display_image(self, image, window):
        if image is not None:
            qformat = QImage.Format_Indexed8

            if len(image.shape) == 3:
                if (image.shape[2]) == 4:
                    qformat = QImage.Format_RGBA8888
                else:
                    qformat = QImage.Format_RGB888

            img = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], qformat)
            img = img.rgbSwapped()

            if window == 1:
                self.boxInput.setPixmap(QPixmap.fromImage(img))
                self.boxInput.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
                self.boxInput.setScaledContents(True)
            elif window == 2:
                self.boxOutput.setPixmap(QPixmap.fromImage(img))
                self.boxOutput.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
                self.boxOutput.setScaledContents(True)

    def classify_from_dataset(self):
        if self.image is not None and self.classifier is not None:
            try:
                gray_image = self.grayscale(self.image)
                image_resized = cv2.resize(gray_image, (28, 28))
                hog_features, _ = hog(image_resized, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True,
                                      block_norm='L2-Hys')
                hog_features = hog_features.reshape(1, -1)
                predictions = self.classifier.predict_proba(hog_features)

                # Ambang batas untuk menerima prediksi
                threshold = 0.5

                # Cari indeks probabilitas tertinggi
                max_prob_index = np.argmax(predictions)

                # Cek apakah probabilitas tertinggi melewati ambang batas
                if predictions[0][max_prob_index] >= threshold:
                    predicted_letter = self.label_mapping[max_prob_index]
                    print(f'Kategori huruf yang diprediksi untuk gambar ini adalah: {predicted_letter}')

                    if self.labelResult:
                        self.labelResult.setText(f'Diprediksi: {predicted_letter}')
                    else:
                        print("labelResult is None. Make sure the QLabel is properly initialized.")
                else:
                    predicted_letter = "Gambar tidak dikenali."
                    print(predicted_letter)

                    if self.labelResult:
                        self.labelResult.setText(predicted_letter)
                    else:
                        print("labelResult is None. Make sure the QLabel is properly initialized.")

                # Tampilkan gambar dan prediksi menggunakan matplotlib
                plt.figure(figsize=(6, 6))
                plt.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
                plt.title(f'Huruf yang Diprediksi untuk gambar ini adalah: {predicted_letter}')
                plt.axis('off')
                plt.show()

            except AttributeError as ae:
                print(f"AttributeError: {ae}")
            except Exception as e:
                print(f"Kesalahan saat mengklasifikasi gambar: {e}")

    def canny_edge_detection(self):
        if self.image is not None:
            canny_image, hasil, gradient_normalized, img_N, img_H2 = self._canny_edge_detection()

            # Plot hasil deteksi tepi Canny
            fig = plt.figure(figsize=(12, 12))
            ax1 = fig.add_subplot(2, 2, 1)
            ax1.imshow(hasil, cmap='gray')
            ax1.title.set_text("Noise Reduction")
            ax2 = fig.add_subplot(2, 2, 2)
            ax2.imshow(gradient_normalized, cmap='gray')
            ax2.title.set_text('Finding Gradient')
            ax3 = fig.add_subplot(2, 2, 3)
            ax3.imshow(img_N, cmap='gray')
            ax3.title.set_text('Non Maximum Suppression')
            ax4 = fig.add_subplot(2, 2, 4)
            ax4.imshow(img_H2, cmap='gray')
            ax4.title.set_text('Hysteresis Thresholding')
            plt.show()

            # Tampilkan hasil deteksi tepi Canny pada GUI
            self.display_image(canny_image, 2)

    def _canny_edge_detection(self):
        if self.image is not None:
            img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            H, W = img.shape

            gauss = (1.0 / 57) * np.array(
                [[0, 1, 2, 1, 0],
                 [1, 3, 5, 3, 1],
                 [2, 5, 9, 5, 2],
                 [1, 3, 5, 3, 1],
                 [0, 1, 2, 1, 0]])
            hasil = self._apply_convolution(img, gauss)

            sobel_x = np.array([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ])
            sobel_y = np.array([
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ])

            Gx = self._apply_convolution(hasil, sobel_x)
            Gy = self._apply_convolution(hasil, sobel_y)

            gradient_magnitude = np.sqrt(Gx ** 2 + Gy ** 2)
            gradient_normalized = ((gradient_magnitude / np.max(gradient_magnitude)) * 255).astype(np.uint8)
            theta = np.arctan2(Gy, Gx)

            angle = theta * 180. / np.pi
            angle[angle < 0] += 180

            img_out = np.zeros((H, W), dtype=np.float64)
            for i in range(1, H - 1):
                for j in range(1, W - 1):
                    try:
                        q = 255
                        r = 255
                        if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                            q = gradient_normalized[i, j + 1]
                            r = gradient_normalized[i, j - 1]
                        elif (22.5 <= angle[i, j] < 67.5):
                            q = gradient_normalized[i + 1, j - 1]
                            r = gradient_normalized[i - 1, j + 1]
                        elif (67.5 <= angle[i, j] < 112.5):
                            q = gradient_normalized[i + 1, j]
                            r = gradient_normalized[i - 1, j]
                        elif (112.5 <= angle[i, j] < 157.5):
                            q = gradient_normalized[i - 1, j - 1]
                            r = gradient_normalized[i + 1, j + 1]
                        if (gradient_normalized[i, j] >= q) and (gradient_normalized[i, j] >= r):
                            img_out[i, j] = gradient_normalized[i, j]
                        else:
                            img_out[i, j] = 0
                    except IndexError as e:
                        pass
            img_N = img_out.astype('uint8')

            weak = np.int32(25)
            strong = np.int32(255)

            for i in range(H):
                for j in range(W):
                    if (img_N[i, j] > weak):
                        img_N[i, j] = strong
                    else:
                        img_N[i, j] = 0

            for i in range(1, H - 1):
                for j in range(1, W - 1):
                    if (img_N[i, j] == weak):
                        try:
                            if ((img_N[i + 1, j - 1] == strong) or (img_N[i + 1, j] == strong) or
                                    (img_N[i + 1, j + 1] == strong) or (img_N[i, j - 1] == strong) or
                                    (img_N[i, j + 1] == strong) or (img_N[i - 1, j - 1] == strong) or
                                    (img_N[i - 1, j] == strong) or (img_N[i - 1, j + 1] == strong)):
                                img_N[i, j] = strong
                        except IndexError as e:
                            pass
            img_H2 = np.where(img_N != strong, 0, img_N)
            img_H2 = np.where(img_H2 == weak, strong, img_H2)

            return img_H2, hasil, gradient_normalized, img_N, img_H2

    def _apply_convolution(self, image, kernel):
        return cv2.filter2D(image, -1, kernel)


def load_dataset(dataset_dir):
    data = pd.read_csv(dataset_dir).astype('float32')
    labels = data['0'].values
    images = data.drop('0', axis=1).values
    features = [hog(image.reshape((28, 28)), pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys') for
                image in images]
    return np.array(features), labels


def train_classifier(dataset_dir):
    features, labels = load_dataset(dataset_dir)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Classifier accuracy on test set: {accuracy * 100:.2f}%")
    return classifier


if __name__ == '__main__':
    dataset_dir = 'C:\\pythonProject2\\Dataset huruf.csv'
    classifier = train_classifier(dataset_dir)

    app = QtWidgets.QApplication([])
    window = ShowImage()
    window.classifier = classifier
    window.show()
    app.exec_()
