import cv2 as cv2
import numpy as np
import statistics

# kepek beolvasasa es tombbe helyezese
im = cv2.imread('im1.jpg', cv2.IMREAD_GRAYSCALE)
im2 = cv2.imread('im2.jpg', cv2.IMREAD_GRAYSCALE)
images = [im2, im]

zeroparam = cv2.imread('zeroparam.jpg', cv2.IMREAD_GRAYSCALE)

print('Algorithms: \n 1 - Kuwahara filter \n 2 - Gradient inverse weighted filter \n 3 - Sigma filter \n')


# ez a resz felelos a kepek kozotti valtasert
def set_image(x):
    i = cv2.getTrackbarPos('Test photo', 'Denoising algorithms')
    j = cv2.getTrackbarPos('Algorithm', 'Denoising algorithms')
    if cv2.getTrackbarPos('Test photo', 'Denoising algorithms') == 0 or cv2.getTrackbarPos('Algorithm',
                                                                                           'Denoising algorithms') == 0:
        # hogyha visszahuzzuk 0 indexre, akkor tunjenek el az ablakok
        final = zeroparam
        cv2.destroyWindow('Photo after noising')
        cv2.destroyWindow('Greyscale original photo')
        cv2.destroyWindow('Photo after denoising')
    elif cv2.getTrackbarPos('Algorithm', 'Denoising algorithms') == 1:
        currentImage = images[i - 1]
        final = kuwahara(currentImage)
    elif cv2.getTrackbarPos('Algorithm', 'Denoising algorithms') == 2:
        currentImage = images[i - 1]
        final = gradient_inverse_weighted(currentImage)
    else:
        currentImage = images[i - 1]
        final = sigma(currentImage)
    cv2.imshow('Photo after denoising', final)


# a csuszkahoz letrehozok egy ablakot
im = np.ndarray((20, 600, 3), np.uint8)
im.fill(192)
cv2.imshow('Denoising algorithms', im)
cv2.createTrackbar('Algorithm', 'Denoising algorithms', 0, 3, set_image)
cv2.createTrackbar('Test photo', 'Denoising algorithms', 0, 2, set_image)


# zajositjuk a kepet
def gaussian_noise(img):
    original = img.copy()
    # Kirajzoljuk az eredeti kepet
    cv2.imshow('Greyscale original photo', original)
    # eloallitjuk a mesterseges zajt
    noise = np.zeros(original.shape, np.int16)
    # varhato ertek: 0 , szoras: 20
    cv2.randn(noise, 0.0, 20.0)  # normalis eloszlasu zajhoz kell a randn
    imnoise = cv2.add(original, noise, dtype=cv2.CV_8UC1)
    # Kirajzoljuk a zajjal terhelt kepet
    print('Gaussian noise added!')
    cv2.imshow('Photo after noising', imnoise)
    return imnoise


def kuwahara(img):
    image = img.copy()

    imnoise = gaussian_noise(image)

    rows, cols = imnoise.shape
    # print("kepmeret:", imnoise.shape)
    # value = [0, 0, 0]
    # top = int(0.05 * imnoise.shape[0])
    # bottom = top
    # left = int(0.05 * imnoise.shape[1])
    # right = left
    # vmi = cv2.copyMakeBorder(imnoise, top, bottom, left, right, value)
    # imnoise = vmi

    print('Applying the filter...')
    for i in range(0, rows):
        for j in range(0, cols):
            # current_pixel = imnoise[i, j]
            # print("aktualis pixel es i es j: ", current_pixel, i, j)
            # kihagyjuk a kep szeleit egyelore
            if j >= cols - 2 or i >= rows - 2:
                break

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

            # print(Q1)
            # print(cv2.mean(np.int32(Q1))[0])
            # print(statistics.stdev(np.int32(Q1)))

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

            imnoise[i, j] = meanofregion

    print('Filter applied!\n')
    return imnoise


def gradient_inverse_weighted(img):
    image = img.copy()
    imnoise = gaussian_noise(image)
    imnoise = np.float32(imnoise)
    rows, cols = imnoise.shape
    print('Applying the filter...')
    for i in range(0, rows):
        for j in range(0, cols):
            # kihagyjuk a kep szeleit egyelore
            if j >= cols - 1 or i >= rows - 1:
                break
            distance1 = round(imnoise[i - 1, j - 1] - imnoise[i, j], 4)
            # print("imnoise i-1 j-1: ", imnoise[i - 1, j - 1], "imnoise i j: ", imnoise[i, j])
            # print("distance1: ", distance1)

            distance2 = imnoise[i - 1, j] - imnoise[i, j]
            distance3 = imnoise[i - 1, j + 1] - imnoise[i, j]
            distance4 = imnoise[i, j - 1] - imnoise[i, j]
            distance5 = imnoise[i, j + 1] - imnoise[i, j]
            distance6 = imnoise[i + 1, j - 1] - imnoise[i, j]
            distance7 = imnoise[i + 1, j] - imnoise[i, j]
            distance8 = imnoise[i + 1, j + 1] - imnoise[i, j]

            delta1 = 1 / distance1 if distance1 > 0 else 2
            delta2 = 1 / distance2 if distance2 > 0 else 2
            delta3 = 1 / distance3 if distance3 > 0 else 2
            delta4 = 1 / distance4 if distance4 > 0 else 2
            delta5 = 1 / distance5 if distance5 > 0 else 2
            delta6 = 1 / distance6 if distance6 > 0 else 2
            delta7 = 1 / distance7 if distance7 > 0 else 2
            delta8 = 1 / distance8 if distance8 > 0 else 2

            sum_delta = delta1 + delta2 + delta3 + delta4 + delta5 + delta6 + delta7 + delta8

            weight1 = delta1 / sum_delta
            weight2 = delta2 / sum_delta
            weight3 = delta3 / sum_delta
            weight4 = delta4 / sum_delta
            weight5 = delta5 / sum_delta
            weight6 = delta6 / sum_delta
            weight7 = delta7 / sum_delta
            weight8 = delta8 / sum_delta

            sum_weight = weight1 * imnoise[i - 1, j - 1] + weight2 * imnoise[i - 1, j] + weight3 * imnoise[
                i - 1, j + 1] + weight4 * imnoise[i, j - 1] + weight5 * imnoise[i, j + 1] + weight6 * imnoise[
                             i + 1, j - 1] + weight7 * imnoise[i + 1, j] + weight8 * imnoise[i + 1, j]

            imnoise[i, j] = 0.5 * imnoise[i, j] + 0.5 * sum_weight
    imnoise = np.uint8(imnoise)

    print('Filter applied!\n')
    return imnoise


def sigma(img):
    original = img.copy()
    imnoise = gaussian_noise(original)
    imnoise = np.float32(imnoise)
    rows, cols = imnoise.shape
    for i in range(0, rows):
        for j in range(0, cols):
            # kihagyjuk a kep szeleit egyelore
            if j >= cols - 1 or i >= rows - 1:
                break
            sum = 0
            count = 0
            # kernelméret: (2n + 1, 2m + 1) ezesetben n, m = 1
            # tehát 3x3-as ablakot vizsgálunk
            for k in range(-1, 2):
                for l in range(-1, 2):
                    # mivel szigma = 20, 2szigma 40, tehat ebben az intervallumban vizsgalodunk
                    if imnoise[i, j] - 40 < imnoise[i + k, j + l] < imnoise[i, j] + 40:
                        sum = sum + imnoise[i + k, j + l]
                        count += 1
            average = round(sum / count, 4)
            # print('Sum: ', round(sum, 4))
            # print('Count: ', count)
            # print('Average:  ', average, '\n')
            imnoise[i, j] = average
    imnoise = np.uint8(imnoise)

    print('Filter applied!\n')
    return imnoise


cv2.waitKey(0)
cv2.destroyAllWindows()
