import cv2 as cv2
import numpy as np

# kepek beolvasasa es tombbe helyezese
im = cv2.imread('im1.jpg', cv2.IMREAD_GRAYSCALE)
im2 = cv2.imread('im3.jpg', cv2.IMREAD_GRAYSCALE)
images = [im, im2]


# ez a resz felelos a kepek kozotti valtasert
def setImage(x):
    i = cv2.getTrackbarPos('Test photo', 'Denoising algorithms')
    if cv2.getTrackbarPos('Test photo', 'Denoising algorithms') == 0:
        final = 0
        cv2.destroyWindow('Photo after noising')
        cv2.destroyWindow('Greyscale original photo')
    else:
        currentImage = images[i - 1]
        final = kuwahara(currentImage)
    cv2.imshow('Photo after denoising', final)


# nemtom mit csinalok ezzel tbh
sth = np.ndarray((20, 600, 3), np.uint8)
sth.fill(192)
cv2.imshow('Denoising algorithms', sth)
cv2.createTrackbar('Test photo', 'Denoising algorithms', 0, 2, setImage)


# kesobb a "zajositast" ki kellene szervezni egy kulon fuggvenybe?
def kuwahara(img):
    returnable = img.copy()
    # Kirajzoljuk az eredeti kepet
    cv2.imshow('Greyscale original photo', returnable)
    # eloallitjuk a mesterseges zajt
    noise = np.zeros(returnable.shape, np.int16)
    # varhato ertek: 0 , szoras: 20
    cv2.randn(noise, 0.0, 20.0) # normalis eloszlasu zajhoz kell a randn
    imnoise1 = cv2.add(returnable, noise, dtype=cv2.CV_8UC1)
    # Kirajzoljuk a zajjal terhelt kepet
    cv2.imshow('Photo after noising', imnoise1)

    return imnoise1


cv2.waitKey(0)
cv2.destroyAllWindows()
