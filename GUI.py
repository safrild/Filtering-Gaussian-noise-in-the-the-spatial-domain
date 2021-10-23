from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget

from gauss_noise_reduction import *

# Kepek beolvasasa es tombbe helyezese
Lenna = cv2.imread('Lenna_(test_image).png', cv2.IMREAD_GRAYSCALE)
cv2.imwrite('Lena_bw.jpg', Lenna)
Lake = cv2.imread('Lake.jpg', cv2.IMREAD_GRAYSCALE)
Tower = cv2.imread('Tower.jpg', cv2.IMREAD_GRAYSCALE)
Wall = cv2.imread('Wall.jpg', cv2.IMREAD_GRAYSCALE)
Lake256 = cv2.imread('Lake_256.jpg', cv2.IMREAD_GRAYSCALE)
images = {"Lenna": Lenna,
          "Lake": Lake,
          "Lake256": Lake256,
          "Tower": Tower,
          "Wall": Wall}
kernels = {"3x3": 1,
           "5x5 (time consuming)": 3}
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
    # Algoritmusvalaszto
    label1 = QtWidgets.QLabel()
    label1.setText("Algorithm: ")
    layout.addWidget(label1)
    comboBoxAlgorithm = QtWidgets.QComboBox(win)
    comboBoxAlgorithm.addItems(
        ["Sigma", "Kuwahara", "Gradient inverse weighted method", "Gradient inverse weighted method upgrade",
         "Bilateral",
         "Bilateral with integral histogram"])
    layout.addWidget(comboBoxAlgorithm)
    # Sigma a zajositashoz
    label2 = QtWidgets.QLabel(win)
    label2.setText("Sigma value: ")
    layout.addWidget(label2)
    comboBoxSigma = QtWidgets.QComboBox(win)
    comboBoxSigma.addItems(["20", "40", "80"])
    layout.addWidget(comboBoxSigma)
    # Kernelmeret
    label4 = QtWidgets.QLabel(win)
    label4.setText("Kernel size: ")
    layout.addWidget(label4)
    comboBoxKernel = QtWidgets.QComboBox(win)
    comboBoxKernel.addItems(kernels)
    layout.addWidget(comboBoxKernel)
    # Input kep
    label3 = QtWidgets.QLabel(win)
    label3.setText("Input photo: ")
    layout.addWidget(label3)
    comboBoxInput = QtWidgets.QComboBox(win)
    comboBoxInput.addItems(images)
    layout.addWidget(comboBoxInput)
    # Range sigma
    label6 = QtWidgets.QLabel(win)
    label6.setText("Range sigma: ")
    layout.addWidget(label6)
    label6.hide()
    sliderRangeSigma = QtWidgets.QSlider(Qt.Horizontal)
    sliderRangeSigma.setMinimum(1)
    sliderRangeSigma.setMaximum(100)
    sliderRangeSigma.setValue(40)
    layout.addWidget(sliderRangeSigma)
    sliderRangeSigma.hide()
    # Spatial sigma
    label7 = QtWidgets.QLabel(win)
    label7.setText("Spatial sigma: ")
    layout.addWidget(label7)
    label7.hide()
    sliderSpaceSigma = QtWidgets.QSlider(Qt.Horizontal)
    sliderSpaceSigma.setMinimum(1)
    sliderSpaceSigma.setMaximum(100)
    sliderSpaceSigma.setValue(40)
    layout.addWidget(sliderSpaceSigma)
    sliderSpaceSigma.hide()
    # GIW_new tobbszori alkalmazas
    label8 = QtWidgets.QLabel(win)
    label8.setText("Repeat times: ")
    label8.hide()
    layout.addWidget(label8)
    comboBoxGIWRepeat = QtWidgets.QComboBox(win)
    comboBoxGIWRepeat.addItems(["1", "2", "3"])
    layout.addWidget(comboBoxGIWRepeat)
    comboBoxGIWRepeat.hide()
    # Futtatas
    btnRun = QtWidgets.QPushButton(win)
    btnRun.setText("Run algorithm")
    btnRun.setCheckable(True)
    layout.addWidget(btnRun)
    comboBoxAlgorithm.currentIndexChanged.connect(lambda: update_window())

    def update_window():
        comboBoxKernel.clear()
        label4.show()
        comboBoxKernel.show()
        label6.hide()
        sliderRangeSigma.hide()
        label7.hide()
        sliderSpaceSigma.hide()
        label8.hide()
        comboBoxGIWRepeat.hide()
        if comboBoxAlgorithm.currentText() == "Kuwahara":
            comboBoxKernel.addItem("5x5 (time consuming)")
        elif comboBoxAlgorithm.currentText() == "Gradient inverse weighted method upgrade":
            comboBoxKernel.addItems(kernels)
            label8.show()
            comboBoxGIWRepeat.show()
        elif comboBoxAlgorithm.currentText() == "Bilateral":
            comboBoxKernel.addItems(kernels)
            label6.show()
            sliderRangeSigma.show()
            label7.show()
            sliderSpaceSigma.show()
        elif comboBoxAlgorithm.currentText() == "Bilateral with integral histogram":
            comboBoxKernel.addItem("5x5 (time consuming)")
        else:
            comboBoxKernel.addItems(kernels)

    win.setLayout(layout)
    win.show()

    btnRun.clicked.connect(
        lambda: call_algorithm(comboBoxAlgorithm.currentText(), comboBoxSigma.currentText(),
                               comboBoxInput.currentText(), comboBoxKernel.currentText(),
                               sliderRangeSigma.value(), sliderSpaceSigma.value(), comboBoxGIWRepeat.currentText()))

    sys.exit(app.exec_())


def call_algorithm(algorithm, sigmaparam, inputphoto, kernelsize, range_sigmaparam, space_sigmaparam, giw_repeat_times):
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
        cv2.imwrite('kuwahara_denoised.jpg', final)
    elif algorithm == "Gradient inverse weighted method":
        final = gradient_inverse_weighted(images[inputphoto], sigma, kernels[kernelsize])
        cv2.imwrite('giw.jpg', final)
    elif algorithm == "Sigma":
        final = sigmaAlgorithm(images[inputphoto], sigma, kernels[kernelsize])
        cv2.imwrite('sigma.jpg', final)
    elif algorithm == "Bilateral":
        final = bilateral(images[inputphoto], sigma, kernels[kernelsize], range_sigma,
                          space_sigmaparam)
        cv2.imwrite('bilateral.jpg', final)
    elif algorithm == "Gradient inverse weighted method upgrade":
        if giw_repeat_times == "1":
            final = gradient_inverse_weighted_method_upgrade(images[inputphoto], sigma, kernels[kernelsize], False)
            cv2.imwrite('giw_new.jpg', final)
        elif giw_repeat_times == "2":
            first = gradient_inverse_weighted_method_upgrade(images[inputphoto], sigma, kernels[kernelsize], False)
            final = gradient_inverse_weighted_method_upgrade(first, sigma, kernels[kernelsize],
                                                             True)
            cv2.imwrite('giw_new.jpg', final)
        else:
            first = gradient_inverse_weighted_method_upgrade(images[inputphoto], sigma, kernels[kernelsize], False)
            second = gradient_inverse_weighted_method_upgrade(first, sigma, kernels[kernelsize], True)
            final = gradient_inverse_weighted_method_upgrade(second, sigma, kernels[kernelsize], True)
            cv2.imwrite('giw_new.jpg', final)
    elif algorithm == "Bilateral with integral histogram":
        final = new_bilateral(images[inputphoto], sigma, kernels[kernelsize])
        cv2.imwrite('bilateral_constant.jpg', final)
    cv2.imshow('Image after denoising', final)


window()
cv2.waitKey(0)
cv2.destroyAllWindows()
