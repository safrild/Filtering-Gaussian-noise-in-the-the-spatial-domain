import math
import statistics
import sys

import cv2 as cv2
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget

# kepek beolvasasa es tombbe helyezese
Lake = cv2.imread('Lake.jpg', cv2.IMREAD_GRAYSCALE)
Tower = cv2.imread('Tower.jpg', cv2.IMREAD_GRAYSCALE)
Wall = cv2.imread('Wall.jpg', cv2.IMREAD_GRAYSCALE)
images = {"Tower": Tower,
          "Wall": Wall,
          "Lake": Lake}
kernels = {"3x3": 1,
           "5x5 (time consuming)": 2}
radiuses = {"1": 1,
            "2": 2}


def window():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = QWidget()
    win.setGeometry(200, 200, 300, 300)
    win.setWindowTitle("Menu")
    layout = QVBoxLayout()
    label1 = QtWidgets.QLabel()
    label1.setText("Algorithm: ")
    layout.addWidget(label1)
    comboBoxAlgorithm = QtWidgets.QComboBox(win)
    comboBoxAlgorithm.addItems(
        ["Sigma", "Kuwahara", "Gradient inverse weighted method", "Gradient inverse weighted method NEW", "Bilateral",
         "Non-local fast"])
    layout.addWidget(comboBoxAlgorithm)
    label2 = QtWidgets.QLabel(win)
    label2.setText("Sigma value: ")
    layout.addWidget(label2)
    comboBoxSigma = QtWidgets.QComboBox(win)
    comboBoxSigma.addItems(["20", "40", "80"])
    layout.addWidget(comboBoxSigma)
    label4 = QtWidgets.QLabel(win)
    label4.setText("Kernel size: ")
    layout.addWidget(label4)
    comboBoxKernel = QtWidgets.QComboBox(win)
    comboBoxKernel.addItems(kernels)
    layout.addWidget(comboBoxKernel)
    label3 = QtWidgets.QLabel(win)
    label3.setText("Input photo: ")
    layout.addWidget(label3)
    comboBoxInput = QtWidgets.QComboBox(win)
    comboBoxInput.addItems(images)
    layout.addWidget(comboBoxInput)
    label5 = QtWidgets.QLabel(win)
    label5.setText("Radius: ")
    layout.addWidget(label5)
    label5.hide()
    comboBoxR = QtWidgets.QComboBox(win)
    comboBoxR.addItems(radiuses)
    layout.addWidget(comboBoxR)
    comboBoxR.hide()
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
        if comboBoxAlgorithm.currentText() == "Kuwahara":
            comboBoxKernel.addItem("5x5")
        else:
            comboBoxKernel.addItems(kernels)
        if comboBoxAlgorithm.currentText() == "Bilateral":
            label5.show()
            comboBoxR.show()
            label4.hide()
            comboBoxKernel.hide()

    win.setLayout(layout)
    win.show()

    btnRun.clicked.connect(lambda: call_algorithm(comboBoxAlgorithm.currentText(), comboBoxSigma.currentText(),
                                                  comboBoxInput.currentText(), comboBoxKernel.currentText(),
                                                  comboBoxR.currentText()))

    sys.exit(app.exec_())


def call_algorithm(algorithm, sigmaparam, inputphoto, kernelSize, r):
    global final
    print("\n")
    print(algorithm)
    print(sigmaparam)
    print(inputphoto)
    print(kernels[kernelSize])
    print("\n")
    sigma = int(sigmaparam)
    if algorithm == "Kuwahara":
        final = kuwahara(images[inputphoto], sigma)
    elif algorithm == "Gradient inverse weighted method":
        final = gradient_inverse_weighted(images[inputphoto], sigma, kernels[kernelSize])
    elif algorithm == "Sigma":
        final = sigmaAlgorithm(images[inputphoto], sigma, kernels[kernelSize])
    elif algorithm == "Bilateral":
        final = bilateral(images[inputphoto], sigma, radiuses[r])
    elif algorithm == "Gradient inverse weighted method NEW":
        final = GIW_new(images[inputphoto], sigma, kernels[kernelSize])
    elif algorithm == "Non-local fast":
        final = non_local_fast(images[inputphoto], sigma, kernels[kernelSize])
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
                # for k in range(-1, 2):
                # for l in range(-1, 2):
                for l in range(-1 * kernelsize, 1 * kernelsize + 1):
                    # 2sigma-t vizsgalunk, pl 20-as szoras eseten ez az ertek 40
                    if imnoise[i, j] - 2 * sigma < imnoise[i + k, j + l] < imnoise[i, j] + 2 * sigma:
                        sum = sum + imnoise[i + k, j + l]
                        count += 1
            average = round(sum / count, 4)
            # print('Sum: ', round(sum, 4))
            # print('Count: ', count)
            # print('Average:  ', average, '\n')
            noisy[i, j] = average
    noisy = np.uint8(noisy)

    print('Filter applied!\n')
    return noisy


def bil(img, sigma, r):
    image = img.copy()
    noisy = gaussian_noise(image, sigma)
    imnoise = border_padding(noisy, 1)

    imnoise = np.float32(imnoise)
    rows, cols = noisy.shape
    print('Applying the filter...')
    t = 12
    finalValue = 0

    for i in range(1, rows):
        for j in range(1, cols):
            for m in range(-r, r):
                if m == 0:
                    continue
                summa = 0
                finalSum = 0
                finalValue = 0
                x1 = np.exp((-(r ** 2 / (2 * sigma ** 2))) - ((imnoise[i, j + m] - imnoise[i, j]) ** 2 / t ** 2))
                summa += x1
                finalSum += imnoise[i, j + m] * x1
                x2 = np.exp((-(r ** 2 / (2 * sigma ** 2))) - ((imnoise[i - m, j] - imnoise[i, j]) ** 2 / t ** 2))
                summa += x2
                finalSum += imnoise[i - m, j] * x2
                if r == 1:
                    finalValue = finalSum / summa
                else:
                    for n in range(-r + 1, r - 1):
                        if n == 0:
                            continue
                        x3 = np.exp(
                            (-(r ** 2 / (2 * sigma ** 2))) - ((imnoise[i + n, j + n] - imnoise[i, j]) ** 2 / t ** 2))
                        summa += x3
                        finalSum += (imnoise[i + n, j + n]) * x3
                        x4 = np.exp(
                            (-(r ** 2 / (2 * sigma ** 2))) - ((imnoise[i - n, j + n] - imnoise[i, j]) ** 2 / t ** 2))
                        summa += x4
                        finalSum += (imnoise[i - n, j + n]) * x4
                        finalValue = finalSum / summa

            noisy[i, j] = finalValue

    print('Filter applied!\n')
    return noisy


def bilateral(img, sigma, r):
    image = img.copy()
    noisy = gaussian_noise(image, sigma)
    # cv2.imshow("nojzi", noisy)
    imnoise = border_padding(noisy, 1)
    imnoise = np.float32(imnoise)
    # cv2.imshow("imnojz", imnoise)

    noisy = np.float32(noisy)

    spatial_szigma = len(np.diagonal(imnoise)) * 0.02
    print("Spatial szigma legyen az atlo hosszanak 2%-a: ", spatial_szigma)
    # TODO: mi alapjan hatarozzuk meg a range szigmat?
    range_szigma = 0.04
    weight = 0

    rows, cols = noisy.shape
    print('Applying the filter...')

    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            summa = 0
            for k in range(1, rows + 1):
                for l in range(1, cols + 1):
                    # space weight
                    # azt hiszem, igy vektorkent ertelmezi majd es a ket vektor normajat szamolja ki, ami az euklideszi tavolsaguk
                    space_weight = round(spatial_szigma * cv2.norm((i, j), (k, l)), 5)
                    # range weight
                    range_weight = round(range_szigma * np.absolute(imnoise[i, j] - imnoise[k, l]), 5)

                    # osszeszorozzuk ezt a ket erteket a pixelintenzitassal
                    result_of_multiplying = round((space_weight * range_weight * imnoise[k, l]), 5)

                    summa = round((summa + result_of_multiplying), 5)

                    # suly kiszamitasa a normalizalashoz
                    weight = np.exp(-((((i - k) ** 2) + ((j - l) ** 2)) / 2 * spatial_szigma ** 2)
                                    - (((np.linalg.norm(imnoise[i, j] - imnoise[k, l])) ** 2) / 2 * range_szigma ** 2))
                    normalized_weight = imnoise[k, l] * weight / weight
                    # print(round(normalized_weight, 5))
                    # a pixel uj erteke
                    if summa != 0 and weight != 0 and normalized_weight != 0:
                        noisy[i - 1, j - 1] = (1 / normalized_weight) * summa
    noisy = np.uint8(noisy)
    print('Filter applied!\n')
    return noisy


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


def non_local_fast(img, sigma, kernelsize):
    #########
    f = open("valami.txt", "a")
    image = img.copy()
    noisy = gaussian_noise(image, sigma)
    imnoise = border_padding(noisy, 2)
    imnoise = np.float32(imnoise)
    rows, cols = noisy.shape
    # creating the weight matrix
    weight_matrix_initial = np.ones(rows * cols)
    print('Applying the filter...')
    treshold_first = (math.sqrt(2) * sigma) / 5
    treshold_second = np.std(imnoise)
    # treshold_second = math.sqrt((1 / ((rows * cols) - 1)))
    # first_moment = 0
    # second_moment = 0
    for i in range(2, rows):
        for j in range(2, cols):
            ############
            counter = i * rows + j
            # First statistical moment: sum
            sum_of_25 = 0
            # Second statistical moment: sum of values - first moment
            sum_of_differences = 0
            ############################
            kernel = get_5x5_kernel(imnoise, i, j)
            # print("Ez a kezdeti kernel most: ", kernel)
            # MENNYI LEGYEN A H????
            h = sigma
            for k in range(2, rows):
                for m in range(2, cols):
                    non_linear_kernel = get_5x5_kernel(imnoise, k, m)
                    # print("Ezekkel kell euklideszi tavolsagokat szamolni: ", non_linear_kernel)
                    distances = np.array([])
                    for n in range(0, 25):
                        # kiszamoljuk egyesevel az euklideszi tavolsagokat az 5x5-os szomszedsagok kozott
                        euclidean_distance = np.linalg.norm(kernel[n] - non_linear_kernel[n])
                        # a tavolsagokat belerakjuk egy kernelbe
                        distances = np.append(distances, euclidean_distance)
                        # kiszamoljuk a valos sulyt
                        weight_value = np.exp(- (distances[n] / (h ** 2)))
                        # print(n, " ", round(weight_value, 5))
                        weight_matrix_initial = np.append(weight_matrix_initial, round(weight_value, 5))
                    # print("Tavolsagok: ", distances)
                    # f.write("Matrix:")
                    # with np.printoptions(threshold=np.inf):
                    # f.write(np.str(weight_matrix_initial))
            noisy[i - 2, j - 2] = (imnoise[i, j] * weight_matrix_initial[counter]) / weight_matrix_initial[counter]
    # cv2.imshow('matrixi', weight_matrix_initial)

    imnoise = np.uint8(imnoise)
    print('Filter applied!\n')
    return imnoise


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
