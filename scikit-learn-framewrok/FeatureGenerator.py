import cv2
import mahotas

class FeatureGenerator():
    def __init__(self):
        pass
    # feature-descriptor-1: Hu Moments
    def fd_hu_moments(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        feature = cv2.HuMoments(cv2.moments(image)).flatten()
        return feature

    # feature-descriptor-2: Haralick Texture
    def fd_haralick(self, image):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # compute the haralick texture feature vector
        haralick = mahotas.features.haralick(gray).mean(axis=0)
        # return the result
        return haralick

    # feature-descriptor-3: Color Histogram
    def fd_histogram(self, image, mask=None):
        # convert the image to HSV color-space
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # compute the color histogram
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        # normalize the histogram
        cv2.normalize(hist, hist)
        # return the histogram
        return hist.flatten()