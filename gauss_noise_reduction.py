import math
import statistics
import sys

import cv2 as cv2
import numpy as np
import time

from math import log10, sqrt

np.set_printoptions(threshold=sys.maxsize)


# Ezzel a fuggvennyel toltjuk ki a kepszeleket zajszures elott "kiterjesztessel"
# A kepek szelein talalhato ertekeket terjesztjuk ki, ezeket igy figyelembe vehetik a zajszuro algoritmusok
def border_padding(img, value):
    src = img.copy()
    output = cv2.copyMakeBorder(src, value, value, value, value, cv2.BORDER_REPLICATE, None, None)
    return output


# Zajositjuk a kepet
def gaussian_noise(img, sigma):
    original = img.copy()
    # Kirajzoljuk az eredeti kepet
    cv2.imshow('Greyscale original photo', original)
    # Eloallitjuk a mesterseges zajt
    noise = np.zeros(original.shape, np.int16)
    # Varhato ertek: 0 , szoras: sigma input parameter
    cv2.randn(noise, 0.0, sigma)  # Normalis eloszlasu zajhoz kell a randn
    imnoise = cv2.add(original, noise, dtype=cv2.CV_8UC1)
    # Kirajzoljuk a zajjal terhelt kepet
    print('Gaussian noise added!')
    cv2.imshow('Image after noising', imnoise)
    return imnoise


# Zajszuro algoritmusok
def kuwahara(img, sigma):
    image = img.copy()

    noisy = gaussian_noise(image, sigma)

    # Kitoltjuk a kep szeleit, a "kiterjesztett", "padded" kepet nem jelentitjuk meg
    imnoise = border_padding(noisy, 2)

    rows, cols = noisy.shape

    denoised = noisy.copy()

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

            # Atlagok
            mean = {
                'Q1': meanq1,
                'Q2': meanq2,
                'Q3': meanq3,
                'Q4': meanq4
            }

            # A regiok szorasai
            deviation = {
                'Q1': devq1,
                'Q2': devq2,
                'Q3': devq3,
                'Q4': devq4
            }

            smallestdevregion = min(deviation, key=deviation.get)  # A legkisebb szorasu regio
            meanofregion = mean[smallestdevregion]  # A legkisebb szorasu regio atlaga

            denoised[i, j] = meanofregion

    return denoised


def sigmaAlgorithm(img, sigma, kernelsize):
    image = img.copy()
    noisy = gaussian_noise(image, sigma)
    imnoise = border_padding(noisy, kernelsize)
    image_to_denoise = np.float32(noisy)
    imnoise = np.float32(imnoise)
    rows, cols = image_to_denoise.shape
    for i in range(2, rows):
        for j in range(2, cols):
            sumofvalues = 0
            count = 0
            # Kernelmeret: (2n + 1, 2m + 1)
            for k in range(-1 * kernelsize, 1 * kernelsize + 1):
                for l in range(-1 * kernelsize, 1 * kernelsize + 1):
                    if imnoise[i, j] - 2 * sigma < imnoise[i + k, j + l] < imnoise[i, j] + 2 * sigma:
                        sumofvalues = sumofvalues + imnoise[i + k, j + l]
                        count += 1
            average = round(sumofvalues / count, 4)
            image_to_denoise[i, j] = average
    denoised = np.uint8(image_to_denoise)

    return denoised


def gradient_inverse_weighted_method(img, sigma, kernelsize):
    image = img.copy()
    noisy = gaussian_noise(image, sigma)
    imnoise = border_padding(noisy, 1)
    denoised = np.float32(noisy)
    imnoise = np.float32(imnoise)
    rows, cols = denoised.shape

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

            denoised[i, j] = 0.5 * imnoise[i, j] + 0.5 * sum_weight

    denoised = np.uint8(denoised)
    return denoised


def gradient_inverse_weighted_method_upgrade(img, sigma, kernelsize, isrepeat):
    image = img.copy()
    if not isrepeat:
        noisy = gaussian_noise(image, sigma)
        imnoise = border_padding(noisy, kernelsize)
    else:
        noisy = image
        imnoise = noisy
    denoised = np.float32(noisy)
    imnoise = np.float32(imnoise)
    imnoise = border_padding(imnoise, kernelsize)
    rows, cols = denoised.shape
    for i in range(0, rows):
        for j in range(0, cols):
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
                # Innentol van elteres az implementacioban, ez a tovabbfejlesztes
                weight_square = ((delta / sum_delta) ** 2)
                weight = delta / sum_delta
                sum_weight += weight * imnoise[i + s, j + s]  # A kepletben ez y(i, j)
                sum_weight_square += weight_square  # A kepletben a D(i, j)

            kij = sum_weight_square / (1 + sum_weight_square)  # K(i, j) kiszamitasa

            denoised[i, j] = kij * imnoise[i, j] + ((1 - kij) * sum_weight)

    denoised = np.uint8(denoised)
    return denoised


def bilateral(img, sigma, kernelsize, range_sigma, space_sigma):
    image = img.copy()
    noised = gaussian_noise(image, sigma)
    filtered = np.zeros([noised.shape[0], noised.shape[1]])
    half_kernelsize = kernelsize // 2
    imnoise = border_padding(noised, half_kernelsize)
    imnoise = np.float32(imnoise)
    filtered = np.float32(filtered)

    # 1. lepes: Beallitjuk a spatial_sigma es a range_sigma ertekeit

    print("Spatial sigma: ", space_sigma)
    print("Range szigma: ", range_sigma)

    # A range_szigma csokkentése mellett erosodik a szuro elmegorzo jellege
    # A space_szigma novekedesevel pedig erosodik a szuro simito hatasa

    rows, cols = imnoise.shape

    # 2. lepes: Eloallitjuk a gauss kernelt

    xdir_gauss = cv2.getGaussianKernel(kernelsize, space_sigma)
    gaussian_kernel = np.multiply(xdir_gauss.T, xdir_gauss)
    print("Kernel: \n", gaussian_kernel)

    start_time = time.perf_counter()
    for i in range(half_kernelsize, rows - half_kernelsize):
        for j in range(half_kernelsize, cols - half_kernelsize):

            p_value = 0.0
            weight = 0.0

            m = half_kernelsize
            n = half_kernelsize

            # 5x5-os a kernel eseten -2-tol 2-ig fut
            for x in range(i - m, i + m + 1):
                for y in range(j - n, j + n + 1):
                    # space weight
                    space_weight = gaussian_kernel[x - i + half_kernelsize, y - j + half_kernelsize]

                    # range weight
                    range_weight = math.exp(-((imnoise[i, j] - imnoise[x, y]) ** 2 / (2 * range_sigma ** 2)))

                    # Osszeszorozzuk ezt a ket sulyerteket a pixelintenzitassal es hozzaadjuk a p ertekehez
                    p_value += (space_weight * range_weight * imnoise[x, y])
                    weight += (space_weight * range_weight)

            # Normalizaljuk a p erteket
            p_value = p_value / weight
            filtered[i - half_kernelsize, j - half_kernelsize] = p_value

    end_time = time.perf_counter()
    print((end_time - start_time) * 1000.0, "ezredmásodperc.")
    filtered = np.uint8(filtered)
    return filtered


def bilateral_with_integral_histogram(img, sigma, kernelsize):
    image = img.copy()
    noised = gaussian_noise(image, sigma)
    filtered = np.zeros([noised.shape[0], noised.shape[1]])
    half_kernelsize = kernelsize // 2
    imnoise = border_padding(noised, half_kernelsize)
    imnoise = np.float32(imnoise)
    filtered = np.float32(filtered)

    integral_histogram = SHcomp(imnoise, half_kernelsize, 256)

    rows, cols = imnoise.shape

    lut = create_lut(kernelsize)  # LUT keszitese

    for i in range(half_kernelsize, rows - half_kernelsize):
        for j in range(half_kernelsize, cols - half_kernelsize):

            weight = 0.0
            normalizalashoz = 0.0
            intenzitas_darabszam_dict = {}

            m = half_kernelsize
            n = half_kernelsize

            for x in range(i - m, i + m + 1):
                for y in range(j - n, j + n + 1):
                    aktualis_intenzitasertek = imnoise[x, y].astype(int)

                    aktualis_intenzitasertek_darabszama = integral_histogram[i][j][aktualis_intenzitasertek]

                    intenzitas_darabszam_dict[aktualis_intenzitasertek] = aktualis_intenzitasertek_darabszama

                    lut_i = int(aktualis_intenzitasertek)
                    lut_ip = int(imnoise[i, j].item())
                    lut_hpi = int(aktualis_intenzitasertek_darabszama)

                    szorzat = lut[lut_i, lut_ip, lut_hpi]

                    normalizalashoz += (szorzat * imnoise[x, y])
                    weight += szorzat

            # Normalizaljuk a sulyerteket
            suly = normalizalashoz / weight
            filtered[i - half_kernelsize, j - half_kernelsize] = suly

    filtered = np.uint8(filtered)
    return filtered


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
    # quantize values at each pixel into bin ID
    b_max = np.max(Ig[:, :])
    b_min = np.min(Ig[:, :])

    # Normalizalas, eltolja az intenzitastartomanyt (ha nem 0 a minimum vagy nem 255 a maximum)
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

    return sh_mtx


# Segedszamitas a Modositott bilateralis algoritmushoz

def create_lut(kernelsize):
    range_sigma = 50
    lut_array = np.ndarray((256, 256, kernelsize * kernelsize + 1), np.float32)
    for x in range(0, lut_array.shape[0]):  # i = imnoise[x, y]
        for y in range(0, lut_array.shape[1]):  # I(p) = imnoise[i, j]
            for z in range(1, lut_array.shape[2]):  # h_p(i)
                range_weight = math.exp(-((y - x) ** 2 / (2 * range_sigma ** 2)))  # g(I(p)-i)
                lut_array[x, y, z] = np.round(x * z * range_weight, 8)
    return lut_array


# Keposszehasonlito metrikak

def psnr_function(original, denoised):
    mse = np.mean((original - denoised) ** 2)
    if mse == 0:  # Ha az MSE 0, akkor nincs zaj a kepen, igy nincs ertelme az osszehasonlitasnak
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    print("PSNR value: ", round(psnr, 4))
    return psnr


def ssim_function(original, denoised):
    # Fenyessegi (luminance) faktor
    original_mean = np.mean(original)
    denoised_mean = np.mean(denoised)
    # L = a pixelertekek dinamikus tartomanya
    l = 255
    k1 = 0.01  # k ajanlott erteke "nagyon kicsi konstans"
    c1 = (k1 * l) ** 2
    luminance = (2 * original_mean * denoised_mean + c1) / (original_mean ** 2 + denoised_mean ** 2 + c1)

    # Kontraszt (contrast) faktor
    original_std = np.std(original)
    denoised_std = np.std(denoised)

    k2 = 0.01
    c2 = (k2 * l) ** 2
    contrast = (2 * original_std * denoised_std + c2) / (original_std ** 2 + denoised_std ** 2 + c2)

    # Strukturalis (structural) faktor
    k3 = 0.01
    c3 = c2 / 2  # Az egyszeruseg kedveert c3 erteke legyen c2 / 2

    rows, cols = denoised.shape
    sumofvalues = 0

    if denoised.shape != original.shape:
        raise ValueError('Different input image sizes!')

    for i in range(0, rows):
        for j in range(0, cols):
            sumofvalues += (original[i, j] - original_mean) * (denoised[i, j] - denoised_mean)

    structural_factor = 1 / (rows * cols - 1) * sumofvalues

    structure = (structural_factor + c3) / (original_std * denoised_std + c3)

    alpha = 1
    beta = 1
    gamma = 1
    ssim = (luminance ** alpha) * (contrast ** beta) * (structure ** gamma)
    print("SSIM value: ", round(ssim, 4))
