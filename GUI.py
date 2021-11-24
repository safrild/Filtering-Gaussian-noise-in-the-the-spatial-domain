from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget

from gauss_noise_reduction import *

# Kepek beolvasasa es tombbe helyezese
Lenna = cv2.imread('img/original_images/Lenna_(test_image).png', cv2.IMREAD_GRAYSCALE)
Chemical_plant = cv2.imread('img/original_images/chemical_plant.tiff', cv2.IMREAD_GRAYSCALE)
Clock = cv2.imread('img/original_images/clock.tiff', cv2.IMREAD_GRAYSCALE)
Wall = cv2.imread('img/original_images/brick_wall.tiff', cv2.IMREAD_GRAYSCALE)
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
    labelAlgorithm = QtWidgets.QLabel()
    labelAlgorithm.setText("Algorithm: ")
    layout.addWidget(labelAlgorithm)
    comboBoxAlgorithm = QtWidgets.QComboBox(win)
    comboBoxAlgorithm.addItems(
        ["Kuwahara", "Sigma", "Gradient inverse weighted method", "Gradient inverse weighted method upgrade",
         "Bilateral",
         "Bilateral with integral histogram"])
    layout.addWidget(comboBoxAlgorithm)
    # Sigma a zajositashoz
    labelSigma = QtWidgets.QLabel(win)
    labelSigma.setText("Sigma value: ")
    layout.addWidget(labelSigma)
    comboBoxSigma = QtWidgets.QComboBox(win)
    comboBoxSigma.addItems(["20", "40", "80"])
    layout.addWidget(comboBoxSigma)
    # Kernelmeret
    labelKernelSize = QtWidgets.QLabel(win)
    labelKernelSize.setText("Kernel size: ")
    layout.addWidget(labelKernelSize)
    comboBoxKernel = QtWidgets.QComboBox(win)
    comboBoxKernel.addItems(kernels)
    layout.addWidget(comboBoxKernel)
    # Input kep
    labelInputImage = QtWidgets.QLabel(win)
    labelInputImage.setText("Input image: ")
    layout.addWidget(labelInputImage)
    comboBoxInput = QtWidgets.QComboBox(win)
    comboBoxInput.addItems(images)
    layout.addWidget(comboBoxInput)
    # Range sigma
    labelRangeSigma = QtWidgets.QLabel(win)
    labelRangeSigma.setText("Range sigma: ")
    layout.addWidget(labelRangeSigma)
    labelRangeSigma.hide()
    sliderRangeSigma = QtWidgets.QSlider(Qt.Horizontal)
    sliderRangeSigma.setMinimum(1)
    sliderRangeSigma.setMaximum(100)
    sliderRangeSigma.setValue(20)
    layout.addWidget(sliderRangeSigma)
    sliderRangeSigma.hide()
    # Spatial sigma
    labelSpatialSigma = QtWidgets.QLabel(win)
    labelSpatialSigma.setText("Spatial sigma: ")
    layout.addWidget(labelSpatialSigma)
    labelSpatialSigma.hide()
    sliderSpaceSigma = QtWidgets.QSlider(Qt.Horizontal)
    sliderSpaceSigma.setMinimum(1)
    sliderSpaceSigma.setMaximum(100)
    sliderSpaceSigma.setValue(60)
    layout.addWidget(sliderSpaceSigma)
    sliderSpaceSigma.hide()
    # GIW_new tobbszori alkalmazas
    labelRepeatTimes = QtWidgets.QLabel(win)
    labelRepeatTimes.setText("Repeat times: ")
    labelRepeatTimes.hide()
    layout.addWidget(labelRepeatTimes)
    comboBoxRepeatTimes = QtWidgets.QComboBox(win)
    comboBoxRepeatTimes.addItems(["1", "2", "3"])
    layout.addWidget(comboBoxRepeatTimes)
    comboBoxRepeatTimes.hide()
    # Futtatas
    btnRun = QtWidgets.QPushButton(win)
    btnRun.setText("Run algorithm")
    btnRun.setCheckable(True)
    layout.addWidget(btnRun)
    comboBoxAlgorithm.currentIndexChanged.connect(lambda: update_window())

    def update_window():
        comboBoxKernel.clear()
        labelKernelSize.show()
        comboBoxKernel.show()
        labelRangeSigma.hide()
        sliderRangeSigma.hide()
        labelSpatialSigma.hide()
        sliderSpaceSigma.hide()
        labelRepeatTimes.hide()
        comboBoxRepeatTimes.hide()
        if comboBoxAlgorithm.currentText() == "Kuwahara":
            comboBoxKernel.clear()
            comboBoxKernel.addItem("5x5 (time consuming)")
        elif comboBoxAlgorithm.currentText() == "Gradient inverse weighted method upgrade":
            comboBoxKernel.addItems(kernels)
            labelRepeatTimes.show()
            comboBoxRepeatTimes.show()
        elif comboBoxAlgorithm.currentText() == "Bilateral":
            comboBoxKernel.addItems(kernels)
            labelRangeSigma.show()
            sliderRangeSigma.show()
            labelSpatialSigma.show()
            sliderSpaceSigma.show()
        else:
            comboBoxKernel.addItems(kernels)

    win.setLayout(layout)
    win.show()

    btnRun.clicked.connect(
        lambda: call_algorithm(comboBoxAlgorithm.currentText(), comboBoxSigma.currentText(),
                               comboBoxInput.currentText(), comboBoxKernel.currentText(),
                               sliderRangeSigma.value(), sliderSpaceSigma.value(), comboBoxRepeatTimes.currentText()))

    sys.exit(app.exec_())


def call_algorithm(algorithm, sigmaparam, image, kernelsize, range_sigmaparam, space_sigmaparam, giw_repeat_times):
    global final
    print("\n")
    print(algorithm)
    print("Deviation of the additive noise: ", sigmaparam)
    print("\n")
    sigma = int(sigmaparam)
    print('Applying the filter...')
    if algorithm == "Kuwahara":
        final = kuwahara(images[image], sigma)
    elif algorithm == "Gradient inverse weighted method":
        final = gradient_inverse_weighted_method(images[image], sigma, kernels[kernelsize])
    elif algorithm == "Sigma":
        final = sigmaAlgorithm(images[image], sigma, kernels[kernelsize])
    elif algorithm == "Bilateral":
        final = bilateral(images[image], sigma, fullkernels[kernelsize], range_sigmaparam,
                          space_sigmaparam)
    elif algorithm == "Gradient inverse weighted method upgrade":
        if giw_repeat_times == "1":
            final = gradient_inverse_weighted_method_upgrade(images[image], sigma, kernels[kernelsize], False)
        elif giw_repeat_times == "2":
            first = gradient_inverse_weighted_method_upgrade(images[image], sigma, kernels[kernelsize], False)
            final = gradient_inverse_weighted_method_upgrade(first, sigma, kernels[kernelsize],
                                                             True)
        else:
            first = gradient_inverse_weighted_method_upgrade(images[image], sigma, kernels[kernelsize], False)
            second = gradient_inverse_weighted_method_upgrade(first, sigma, kernels[kernelsize], True)
            final = gradient_inverse_weighted_method_upgrade(second, sigma, kernels[kernelsize], True)
    elif algorithm == "Bilateral with integral histogram":
        final = bilateral_with_integral_histogram(images[image], sigma, fullkernels[kernelsize])
    print('Filter applied!\n')
    psnr_function(images[image], final)
    ssim_function(images[image], final)
    # cv2.imwrite('img/denoised images/%(algorithm)s_%(noise)s_%(kernel)s.jpg' % {"algorithm": algorithm, "noise": sigma,
    #                                                                             "kernel": kernelsize}, final)
    cv2.imshow('Image after denoising', final)


window()
cv2.waitKey(0)
cv2.destroyAllWindows()
