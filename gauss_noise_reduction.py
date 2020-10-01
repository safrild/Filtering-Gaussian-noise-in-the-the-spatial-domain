import cv2 as cv2
import numpy as np
import statistics

# kepek beolvasasa es tombbe helyezese
im = cv2.imread('im1.jpg', cv2.IMREAD_GRAYSCALE)
im2 = cv2.imread('im2.jpg', cv2.IMREAD_GRAYSCALE)
images = [im2, im]


# ez a resz felelos a kepek kozotti valtasert
def setImage(x):
    i = cv2.getTrackbarPos('Test photo', 'Denoising algorithms')
    if cv2.getTrackbarPos('Test photo', 'Denoising algorithms') == 0:
        # hogyha visszahuzzuk 0 indexre, akkor tunjenek el az ablakok
        final = 0
        cv2.destroyWindow('Photo after noising')
        cv2.destroyWindow('Greyscale original photo')
        cv2.destroyWindow('Photo after denoising')
    else:
        currentImage = images[i - 1]
        final = kuwahara(currentImage)
    cv2.imshow('Photo after denoising', final)


# a csuszkahoz letrehozok egy ablakot
im = np.ndarray((20, 600, 3), np.uint8)
im.fill(192)
cv2.imshow('Denoising algorithms', im)
cv2.createTrackbar('Test photo', 'Denoising algorithms', 0, 2, setImage)


# kesobb a "zajositast" ki kellene szervezni egy kulon fuggvenybe?
def kuwahara(img):
    original = img.copy()
    # Kirajzoljuk az eredeti kepet
    cv2.imshow('Greyscale original photo', original)
    # eloallitjuk a mesterseges zajt
    noise = np.zeros(original.shape, np.int16)
    # varhato ertek: 0 , szoras: 20
    cv2.randn(noise, 0.0, 20.0)  # normalis eloszlasu zajhoz kell a randn
    imnoise = cv2.add(original, noise, dtype=cv2.CV_8UC1)
    # Kirajzoljuk a zajjal terhelt kepet
    cv2.imshow('Photo after noising', imnoise)

    rows, cols = imnoise.shape
    # print("kepmeret:", imnoise.shape)

    for i in range(0, rows):
        for j in range(0, cols):
            current_pixel = imnoise[i, j]
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

    return imnoise


cv2.waitKey(0)
cv2.destroyAllWindows()
