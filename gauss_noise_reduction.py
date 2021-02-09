import math
import statistics
import sys
import cv2 as cv2
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget
import matplotlib.pyplot as plt

# kepek beolvasasa es tombbe helyezese
Lake = cv2.imread('Lake.jpg', cv2.IMREAD_GRAYSCALE)
Tower = cv2.imread('Tower.jpg', cv2.IMREAD_GRAYSCALE)
Wall = cv2.imread('Wall.jpg', cv2.IMREAD_GRAYSCALE)
images = {"Lake": Lake,
          "Tower": Tower,
          "Wall": Wall}
kernels = {"3x3": 1,
           "5x5 (time consuming)": 2}
radiuses = {"1": 1,
            "2": 2}
range_sigmas = {"10": 10,
                "20": 20,
                "40": 40,
                "60": 60,
                "80": 80,
                "100": 100}


def window():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = QWidget()
    win.setGeometry(200, 200, 300, 300)
    win.setWindowTitle("Menu")
    layout = QVBoxLayout()
    # algorithm chooser
    label1 = QtWidgets.QLabel()
    label1.setText("Algorithm: ")
    layout.addWidget(label1)
    comboBoxAlgorithm = QtWidgets.QComboBox(win)
    comboBoxAlgorithm.addItems(
        ["Sigma", "Kuwahara", "Gradient inverse weighted method", "Gradient inverse weighted method NEW", "Bilateral",
         "Bilateral constant time"])
    layout.addWidget(comboBoxAlgorithm)
    # Sigma a zajosításhoz
    label2 = QtWidgets.QLabel(win)
    label2.setText("Sigma value: ")
    layout.addWidget(label2)
    comboBoxSigma = QtWidgets.QComboBox(win)
    comboBoxSigma.addItems(["20", "40", "80"])
    layout.addWidget(comboBoxSigma)
    # Kernel size
    label4 = QtWidgets.QLabel(win)
    label4.setText("Kernel size: ")
    layout.addWidget(label4)
    comboBoxKernel = QtWidgets.QComboBox(win)
    comboBoxKernel.addItems(kernels)
    layout.addWidget(comboBoxKernel)
    # input photo
    label3 = QtWidgets.QLabel(win)
    label3.setText("Input photo: ")
    layout.addWidget(label3)
    comboBoxInput = QtWidgets.QComboBox(win)
    comboBoxInput.addItems(images)
    layout.addWidget(comboBoxInput)
    # radius
    label5 = QtWidgets.QLabel(win)
    label5.setText("Radius: ")
    layout.addWidget(label5)
    label5.hide()
    comboBoxR = QtWidgets.QComboBox(win)
    comboBoxR.addItems(radiuses)
    layout.addWidget(comboBoxR)
    comboBoxR.hide()
    # range sigma
    label6 = QtWidgets.QLabel(win)
    label6.setText("Range sigma: ")
    layout.addWidget(label6)
    label6.hide()
    sliderRangeSigma = QtWidgets.QSlider(Qt.Horizontal)
    sliderRangeSigma.setMinimum(10)
    sliderRangeSigma.setMaximum(100)
    sliderRangeSigma.setValue(40)
    layout.addWidget(sliderRangeSigma)
    sliderRangeSigma.hide()
    # spatial sigma
    label7 = QtWidgets.QLabel(win)
    label7.setText("Spatial sigma: ")
    layout.addWidget(label7)
    label7.hide()
    sliderSpaceSigma = QtWidgets.QSlider(Qt.Horizontal)
    sliderSpaceSigma.setMinimum(10)
    sliderSpaceSigma.setMaximum(100)
    sliderSpaceSigma.setValue(40)
    layout.addWidget(sliderSpaceSigma)
    sliderSpaceSigma.hide()
    # run button
    btnRun = QtWidgets.QPushButton(win)
    btnRun.setText("Run algorithm")
    btnRun.setCheckable(True)
    layout.addWidget(btnRun)
    comboBoxAlgorithm.currentIndexChanged.connect(lambda: update_window())

    def update_window():
        comboBoxKernel.clear()
        label5.hide()
        comboBoxR.hide()
        label4.show()
        comboBoxKernel.show()
        label6.hide()
        sliderRangeSigma.hide()
        if comboBoxAlgorithm.currentText() == "Kuwahara":
            comboBoxKernel.addItem("5x5 (time consuming)")
        elif comboBoxAlgorithm.currentText() == "Bilateral":
            comboBoxKernel.addItem("5x5 (time consuming)")
            label6.show()
            sliderRangeSigma.show()
            label7.show()
            sliderSpaceSigma.show()
        else:
            comboBoxKernel.addItems(kernels)

    win.setLayout(layout)
    win.show()

    btnRun.clicked.connect(lambda: call_algorithm(comboBoxAlgorithm.currentText(), comboBoxSigma.currentText(),
                                                  comboBoxInput.currentText(), comboBoxKernel.currentText(),
                                                  sliderRangeSigma.value(), sliderSpaceSigma.value()))

    def sliderRangeSigmaValueChanged():
        pass

    sys.exit(app.exec_())


def call_algorithm(algorithm, sigmaparam, inputphoto, kernelsize, range_sigmaparam, space_sigmaparam):
    global final
    print("\n")
    print(algorithm)
    print(sigmaparam)
    print(inputphoto)
    print(kernels[kernelsize])
    print("\n")
    sigma = int(sigmaparam)
    range_sigma = range_sigmaparam
    if algorithm == "Kuwahara":
        final = kuwahara(images[inputphoto], sigma)
    elif algorithm == "Gradient inverse weighted method":
        final = gradient_inverse_weighted(images[inputphoto], sigma, kernels[kernelsize])
    elif algorithm == "Sigma":
        final = sigmaAlgorithm(images[inputphoto], sigma, kernels[kernelsize])
    elif algorithm == "Bilateral":
        final = bilateral(images[inputphoto], sigma, kernels[kernelsize], range_sigma, space_sigmaparam)
    elif algorithm == "Gradient inverse weighted method NEW":
        final = GIW_new(images[inputphoto], sigma, kernels[kernelsize])
    elif algorithm == "Bilateral constant time":
        final = constant_time_bilateral(images[inputphoto], sigma)
    cv2.imshow('Image after denoising', final)


# ezzel a fuggvennyel toltjuk ki a kepszeleket zajszures elott "kiterjesztessel"
# a kepek szelein talalhato ertekeket terjesztjuk ki, ezeket igy figyelembe vehetik a zajszuro algoritmusok
def border_padding(img, value):
    src = img.copy()
    output = cv2.copyMakeBorder(src, value, value, value, value, cv2.BORDER_REPLICATE, None, None)
    return output


# zajositjuk a kepet
def gaussian_noise(img, sigma):
    original = img.copy()
    # kirajzoljuk az eredeti kepet
    cv2.imshow('Greyscale original photo', original)
    # eloallitjuk a mesterseges zajt
    noise = np.zeros(original.shape, np.int16)
    # varhato ertek: 0 , szoras: SIGMA konstans
    cv2.randn(noise, 0.0, sigma)  # normalis eloszlasu zajhoz kell a randn
    imnoise = cv2.add(original, noise, dtype=cv2.CV_8UC1)
    # Kirajzoljuk a zajjal terhelt kepet
    print('Gaussian noise added!')
    cv2.imshow('Image after noising', imnoise)
    return imnoise


# nem allithato a kernelmeret ennel az algoritmusnal, mert az 5x5-os mar igyis tulsagosan idoigenyes
def kuwahara(img, sigma):
    image = img.copy()

    noisy = gaussian_noise(image, sigma)

    # kitoltjuk a kep szeleit
    # a "kiterjesztett", "padded" kepet nem jelentitjuk meg
    imnoise = border_padding(noisy, 2)

    rows, cols = noisy.shape

    print('Applying the filter...')
    for i in range(2, rows):
        for j in range(2, cols):
            Q1 = [imnoise[i - 2, j - 2], imnoise[i - 2, j - 1], imnoise[i - 2, j],
                  imnoise[i - 1, j - 2], imnoise[i - 1, j - 1], imnoise[i - 1, j],
                  imnoise[i, j - 2], imnoise[i, j - 1], imnoise[i, j]]
            Q2 = [imnoise[i - 2, j], imnoise[i - 2, j + 1], imnoise[i - 2, j + 2],
                  imnoise[i - 1, j], imnoise[i - 1, j + 1], imnoise[i - 1, j + 2],
                  imnoise[i, j], imnoise[i, j + 1], imnoise[i, j + 2]]
            Q3 = [imnoise[i, j - 2], imnoise[i, j - 1], imnoise[i, j],
                  imnoise[i + 1, j - 2], imnoise[i + 1, j + 1], imnoise[i + 1, j],
                  imnoise[i + 2, j - 2], imnoise[i + 2, j - 1], imnoise[i + 2, j]]
            Q4 = [imnoise[i, j], imnoise[i, j + 1], imnoise[i, j + 2],
                  imnoise[i + 1, j], imnoise[i + 1, j + 1], imnoise[i + 1, j + 2],
                  imnoise[i + 2, j], imnoise[i + 2, j + 1], imnoise[i + 2, j + 2]]

            # 4 tizedesre kerekitjuk az ertekeket

            meanq1 = round(cv2.mean(np.int32(Q1))[0], 4)
            meanq2 = round(cv2.mean(np.int32(Q2))[0], 4)
            meanq3 = round(cv2.mean(np.int32(Q3))[0], 4)
            meanq4 = round(cv2.mean(np.int32(Q4))[0], 4)

            devq1 = round(statistics.stdev(np.int32(Q1)), 4)
            devq2 = round(statistics.stdev(np.int32(Q2)), 4)
            devq3 = round(statistics.stdev(np.int32(Q3)), 4)
            devq4 = round(statistics.stdev(np.int32(Q4)), 4)

            mean = {
                'Q1': meanq1,
                'Q2': meanq2,
                'Q3': meanq3,
                'Q4': meanq4,
            }

            deviation = {
                'Q1': devq1,
                'Q2': devq2,
                'Q3': devq3,
                'Q4': devq4
            }

            # print('Deviations of regions:', deviation)
            smallestdevregion = min(deviation, key=deviation.get)
            # print('Region with smallest deviation: ', smallestdevregion)
            meanofregion = mean[smallestdevregion]
            # print('Means: ', mean)
            # print('Mean of region with smallest dev: ', meanofregion)

            noisy[i, j] = meanofregion

    print('Filter applied!\n')
    return noisy


def gradient_inverse_weighted(img, sigma, kernelsize):
    image = img.copy()
    noisy = gaussian_noise(image, sigma)
    imnoise = border_padding(noisy, 1)
    noisy = np.float32(noisy)
    imnoise = np.float32(imnoise)
    rows, cols = noisy.shape
    print('Applying the filter...')

    for i in range(1, rows):
        for j in range(1, cols):
            sum_delta = 0
            sum_weight = 0
            for k in range(-kernelsize, kernelsize):
                if k == 0:
                    continue
                distance = imnoise[i + k, j + k] - imnoise[i, j]
                delta = 1 / distance if distance > 0 else 2
                sum_delta += delta

            for s in range(-kernelsize, kernelsize):
                if s == 0:
                    continue
                distance = imnoise[i + s, j + s] - imnoise[i, j]
                delta = 1 / distance if distance > 0 else 2
                weight = delta / sum_delta
                sum_weight += weight * imnoise[i + s, j + s]

            noisy[i, j] = 0.5 * imnoise[i, j] + 0.5 * sum_weight

    noisy = np.uint8(noisy)
    print('Filter applied!\n')
    return noisy


def sigmaAlgorithm(img, sigma, kernelsize):
    image = img.copy()
    noisy = gaussian_noise(image, sigma)
    imnoise = border_padding(noisy, kernelsize)
    noisy = np.float32(noisy)
    imnoise = np.float32(imnoise)
    rows, cols = noisy.shape
    print('Applying the filter...')
    for i in range(2, rows):
        for j in range(2, cols):
            sum = 0
            count = 0
            # kernelméret: (2n + 1, 2m + 1)
            # 3x3-as vagy 5x5-ös ablakot vizsgálunk
            for k in range(-1 * kernelsize, 1 * kernelsize + 1):
                for l in range(-1 * kernelsize, 1 * kernelsize + 1):
                    # 2sigma-t vizsgalunk, pl 20-as szoras eseten ez az ertek 40
                    if imnoise[i, j] - 2 * sigma < imnoise[i + k, j + l] < imnoise[i, j] + 2 * sigma:
                        sum = sum + imnoise[i + k, j + l]
                        count += 1
            average = round(sum / count, 4)
            noisy[i, j] = average
    noisy = np.uint8(noisy)

    print('Filter applied!\n')
    return noisy


def bilateral(img, sigma, kernelsize, range_sigma, space_sigma):
    image = img.copy()
    imnoise = gaussian_noise(image, sigma)
    filtered = np.zeros([imnoise.shape[0], imnoise.shape[1]])
    imnoise = border_padding(imnoise, 2)
    imnoise = np.float32(imnoise)
    filtered = np.float32(filtered)

    # step 1: set spatial_sigma and range_sigma

    print("Spatial sigma: ", space_sigma)

    # range_szigma = 50
    print("Range szigma: ", range_sigma)

    # A range_szigma csökkentése mellett erősödik a szűrő élmegőrző jellege
    # a space_szigma növekedésével pedig erősödik a szűrő simító hatása.

    rows, cols = imnoise.shape
    print('Applying the filter...')

    # step 2: make gauss kernel

    xdir_gauss = cv2.getGaussianKernel(5, 1.0)
    gaussian_kernel = np.multiply(xdir_gauss.T, xdir_gauss)
    print("Kernel: \n", gaussian_kernel)

    # legyen 5x5-os kernel most
    kernel_s = 5

    for i in range(2, rows - 2):
        for j in range(2, cols - 2):
            # print("i: ", i, "j: ", j)

            p_value = 0.0
            weight = 0.0

            m = kernel_s // 2
            n = kernel_s // 2

            # ha 5x5-os a kernel, akkor ez -2-tol 2-ig fut
            for x in range(i - m, i + m + 1):
                for y in range(j - n, j + n + 1):
                    # print("x: ", x, "y: ", y)

                    # space weight
                    space_weight = gaussian_kernel[x - i + 2, y - j + 2]

                    # range weight
                    range_weight = math.exp(-((imnoise[i, j] - imnoise[x, y]) ** 2 / (2 * range_sigma ** 2)))

                    # osszeszorozzuk ezt a ket sulyerteket a pixelintenzitassal es hozzaadjuk a p ertekehez
                    p_value += (space_weight * range_weight * imnoise[x, y])
                    weight += (space_weight * range_weight)

            # normalizaljuk a p erteket
            # print("weight: ", weight)
            p_value = p_value / weight
            filtered[i - 2, j - 2] = p_value

    filtered = np.uint8(filtered)
    return filtered


def constant_time_bilateral(img, sigma):
    image = img.copy()
    imnoise = gaussian_noise(image, sigma)
    filtered = np.zeros([imnoise.shape[0], imnoise.shape[1]])
    imnoise = border_padding(imnoise, 2)
    imnoise = np.float32(imnoise)
    filtered = np.float32(filtered)

    print(LH(image, 2, 256, image.shape[0], image.shape[1]))

    # h = cv2.calcHist(imnoise, [0], None, [256], [0, 256])
    # plt.hist(imnoise.ravel(), 256, [0, 256])
    # plt.show()

    spatial_szigma = 1
    print("Spatial szigma konstans 1 ertek: ", spatial_szigma)

    range_szigma = 50
    print("Range szigma: ", range_szigma)

    # A range_szigma csökkentése mellett erősödik a szűrő élmegőrző jellege
    # a space_szigma növelésével pedig erősödik a szűrő simító hatása.

    rows, cols = imnoise.shape
    print('Applying the filter...')

    # step 2: make gauss kernel

    xdir_gauss = cv2.getGaussianKernel(5, 1.0)
    gaussian_kernel = np.multiply(xdir_gauss.T, xdir_gauss)
    print("Kernel: \n", gaussian_kernel)

    # legyen 5x5-os kernel most
    kernel_s = 5

    for i in range(2, rows - 2):
        for j in range(2, cols - 2):
            # print("i: ", i, "j: ", j)

            p_value = 0.0
            weight = 0.0

            m = kernel_s // 2
            n = kernel_s // 2

            # ha 5x5-os a kernel, akkor ez -2-tol 2-ig fut
            for x in range(i - m, i + m + 1):
                for y in range(j - n, j + n + 1):
                    # print("x: ", x, "y: ", y)

                    # space weight
                    space_weight = gaussian_kernel[x - i + 2, y - j + 2]

                    # range weight
                    range_weight = math.exp(-((imnoise[i, j] - imnoise[x, y]) ** 2 / (2 * range_szigma ** 2)))

                    # osszeszorozzuk ezt a ket sulyerteket a pixelintenzitassal es hozzaadjuk a p ertekehez
                    p_value += (space_weight * range_weight * imnoise[x, y])
                    weight += (space_weight * range_weight)

            # normalizaljuk a p erteket
            # print("weight: ", weight)
            p_value = p_value / weight
            filtered[i - 2, j - 2] = p_value

    filtered = np.uint8(filtered)
    return filtered


def LH(Ig, ws, no_bins, nrows, ncols, integral_hist=None):
    h = Ig.shape[1]
    # assign bin ids
    bin_width = 255. / no_bins
    I_bin_id = np.floor(Ig / bin_width)
    # one hot encoding
    one_hot_pix = np.zeros((nrows * ncols, no_bins))
    one_hot_pix[np.arange(nrows * ncols), I_bin_id.flatten()] = 1
    one_hot_pix = one_hot_pix.reshape((nrows, ncols, no_bins))

    np.cumsum(one_hot_pix, axis=1, out=integral_hist)  # left to right
    np.cumsum(integral_hist, axis=0, out=integral_hist)  # top to bottom

    # pad integral histogram array
    padding_left = np.zeros((h, ws + 1, no_bins))
    padding_right = np.tile(integral_hist[:, -1:, :], (1, ws, 1))
    integral_hist_pad = np.concatenate([padding_left, integral_hist, padding_right], axis=1)
    padding_top = np.zeros((ws + 1, integral_hist_pad.shape[1], no_bins))
    padding_bottom = np.tile(integral_hist_pad[-1:, :, :], (ws, 1, 1))
    integral_hist_pad = np.concatenate([padding_top, integral_hist_pad, padding_bottom], axis=0)
    # find sub-arrays of four corners
    integral_hist_1 = integral_hist_pad[:-ws - ws - 1, :-ws - ws - 1, :]
    integral_hist_2 = integral_hist_pad[:-ws - ws - 1, ws + 1 + ws:, :]
    integral_hist_3 = integral_hist_pad[ws + 1 + ws:, :-ws - ws - 1, :]
    integral_hist_4 = integral_hist_pad[ws + 1 + ws:, ws + 1 + ws:, :]
    local_hist = integral_hist_4 + integral_hist_1 - integral_hist_2 - integral_hist_3

    return local_hist


def SHcomp(Ig, ws, BinN=11):
    """
    Compute local spectral histogram using integral histograms
    :param Ig: a n-band image
    :param ws: half window size
    :param BinN: number of bins of histograms
    :return: local spectral histogram at each pixel
    """
    h, w = Ig.shape
    bn = 1
    #
    # quantize values at each pixel into bin ID
    # for i in range(bn):
    b_max = np.max(Ig[:, :])
    b_min = np.min(Ig[:, :])

    # normalizalas, eltolja az intenzitastartomanyt (ha nem 0 a minimum vagy nem 255 a maximum)
    b_interval = (b_max - b_min) * 1. / BinN
    Ig[:, :] = np.floor((Ig[:, :] - b_min) / b_interval)

    Ig[Ig >= BinN] = BinN - 1
    Ig = np.int32(Ig)

    # convert to one hot encoding
    one_hot_pix = []
    one_hot_pix_b = np.zeros((h * w, BinN), dtype=np.int32)
    one_hot_pix_b[np.arange(h * w), Ig[:, :].flatten()] = 1
    one_hot_pix.append(one_hot_pix_b.reshape((h, w, BinN)))

    # compute integral histogram
    integral_hist = np.concatenate(one_hot_pix, axis=2)

    np.cumsum(integral_hist, axis=1, out=integral_hist, dtype=np.float32)
    np.cumsum(integral_hist, axis=0, out=integral_hist, dtype=np.float32)

    # compute spectral histogram based on integral histogram
    padding_l = np.zeros((h, ws + 1, BinN * bn), dtype=np.int32)
    padding_r = np.tile(integral_hist[:, -1:, :], (1, ws, 1))

    integral_hist_pad_tmp = np.concatenate([padding_l, integral_hist, padding_r], axis=1)

    padding_t = np.zeros((ws + 1, integral_hist_pad_tmp.shape[1], BinN * bn), dtype=np.int32)
    padding_b = np.tile(integral_hist_pad_tmp[-1:, :, :], (ws, 1, 1))

    integral_hist_pad = np.concatenate([padding_t, integral_hist_pad_tmp, padding_b], axis=0)

    integral_hist_1 = integral_hist_pad[ws + 1 + ws:, ws + 1 + ws:, :]
    integral_hist_2 = integral_hist_pad[:-ws - ws - 1, :-ws - ws - 1, :]
    integral_hist_3 = integral_hist_pad[ws + 1 + ws:, :-ws - ws - 1, :]
    integral_hist_4 = integral_hist_pad[:-ws - ws - 1, ws + 1 + ws:, :]

    sh_mtx = integral_hist_1 + integral_hist_2 - integral_hist_3 - integral_hist_4

    histsum = np.sum(sh_mtx, axis=-1, keepdims=True)

    sh_mtx = np.float32(sh_mtx) / np.float32(histsum)

    return sh_mtx


def GIW_new(img, sigma, kernelsize):
    image = img.copy()
    noisy = gaussian_noise(image, sigma)
    imnoise = border_padding(noisy, 1)
    noisy = np.float32(noisy)
    imnoise = np.float32(imnoise)
    rows, cols = noisy.shape
    print('Applying the filter...')
    for i in range(1, rows):
        for j in range(1, cols):
            sum_delta = 0
            sum_weight_square = 0
            sum_weight = 0
            for k in range(-kernelsize, kernelsize):
                if k == 0:
                    continue
                distance = imnoise[i + k, j + k] - imnoise[i, j]
                delta = 1 / distance if distance > 0 else 2
                sum_delta += delta

            for s in range(-kernelsize, kernelsize):
                if s == 0:
                    continue
                distance = imnoise[i + s, j + s] - imnoise[i, j]
                delta = 1 / distance if distance > 0 else 2
                # innentol van elteres az implementacioban, ez a NEW GIW mar
                weight_square = ((delta / sum_delta) ** 2)
                weight = delta / sum_delta
                sum_weight += weight * imnoise[i + s, j + s]  # kepletben y(i, j)
                sum_weight_square += weight_square  # ez a kepletben a D(i, j)
                # K(i, j) = sum_weight_square / (1+sum_weight_square)

            kij = sum_weight_square / (1 + sum_weight_square)

            noisy[i, j] = kij * imnoise[i, j] + ((1 - kij) * sum_weight)

    noisy = np.uint8(noisy)
    print('Filter applied!\n')
    return noisy


# imnoise a vizsgalt kep, i es j pedig az aktualis pixel
def get_5x5_kernel(imnoise, i, j):
    kernel = [imnoise[i - 2, j - 2], imnoise[i - 2, j - 1], imnoise[i - 2, j], imnoise[i - 1, j - 2],
              imnoise[i - 1, j - 1],
              imnoise[i, j - 2], imnoise[i, j - 1], imnoise[i, j],
              imnoise[i - 2, j + 1], imnoise[i - 2, j + 2], imnoise[i - 1, j], imnoise[i - 1, j + 1],
              imnoise[i - 1, j + 2],
              imnoise[i, j + 1], imnoise[i, j + 2], imnoise[i + 1, j - 2],
              imnoise[i + 1, j + 1], imnoise[i + 2, j - 2], imnoise[i + 2, j - 1], imnoise[i + 2, j],
              imnoise[i + 1, j], imnoise[i + 1, j + 2], imnoise[i + 2, j],
              imnoise[i + 2, j + 1], imnoise[i + 2, j + 2]]
    return kernel


window()
cv2.waitKey(0)
cv2.destroyAllWindows()
