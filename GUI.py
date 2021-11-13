from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget

from gauss_noise_reduction import *

# Kepek beolvasasa es tombbe helyezese
Lenna = cv2.imread('img/Lenna_(test_image).png', cv2.IMREAD_GRAYSCALE)
Chemical_plant = cv2.imread('img/chemical_plant.tiff', cv2.IMREAD_GRAYSCALE)
Clock = cv2.imread('img/clock.tiff', cv2.IMREAD_GRAYSCALE)
Wall = cv2.imread('img/brick_wall.tiff', cv2.IMREAD_GRAYSCALE)
images = {"Lenna": Lenna,
          "Chemical plant": Chemical_plant,
          "Clock": Clock,
          "Wall": Wall}
kernels = {"3x3": 1,
           "5x5 (time consuming)": 3}
fullkernels = {"3x3": 3,
               "5x5 (time consuming)": 5}
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
        ["Kuwahara", "Sigma", "Gradient inverse weighted method", "Gradient inverse weighted method upgrade",
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
    sliderRangeSigma.setValue(20)
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
    sliderSpaceSigma.setValue(60)
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
            comboBoxKernel.clear()
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
    print("Deviation of the additive noise: ", sigmaparam)
    print("\n")
    sigma = int(sigmaparam)
    print('Applying the filter...')
    if algorithm == "Kuwahara":
        final = kuwahara(images[inputphoto], sigma)
    elif algorithm == "Gradient inverse weighted method":
        final = gradient_inverse_weighted_method(images[inputphoto], sigma, kernels[kernelsize])
    elif algorithm == "Sigma":
        final = sigmaAlgorithm(images[inputphoto], sigma, kernels[kernelsize])
    elif algorithm == "Bilateral":
        final = bilateral(images[inputphoto], sigma, kernels[kernelsize], range_sigmaparam,
                          space_sigmaparam)
    elif algorithm == "Gradient inverse weighted method upgrade":
        if giw_repeat_times == "1":
            final = gradient_inverse_weighted_method_upgrade(images[inputphoto], sigma, kernels[kernelsize], False)
        elif giw_repeat_times == "2":
            first = gradient_inverse_weighted_method_upgrade(images[inputphoto], sigma, kernels[kernelsize], False)
            final = gradient_inverse_weighted_method_upgrade(first, sigma, kernels[kernelsize],
                                                             True)
        else:
            first = gradient_inverse_weighted_method_upgrade(images[inputphoto], sigma, kernels[kernelsize], False)
            second = gradient_inverse_weighted_method_upgrade(first, sigma, kernels[kernelsize], True)
            final = gradient_inverse_weighted_method_upgrade(second, sigma, kernels[kernelsize], True)
    elif algorithm == "Bilateral with integral histogram":
        final = bilateral_with_integral_histogram(images[inputphoto], sigma, fullkernels[kernelsize])
    print('Filter applied!\n')
    psnr_function(images[inputphoto], final)
    ssim_function(images[inputphoto], final)
    # cv2.imwrite('img/denoised images/%(algorithm)s_%(noise)s_%(kernel)s.jpg' % {"algorithm": algorithm, "noise": sigma,
    #                                                                             "kernel": kernelsize}, final)
    cv2.imshow('Image after denoising', final)


window()
cv2.waitKey(0)
cv2.destroyAllWindows()
