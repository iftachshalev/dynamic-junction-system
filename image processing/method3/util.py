import cv2
from skimage.feature import hog
import matplotlib.image as mpimg
import numpy as np


class Util:

    def __init__(self, cspace='RGB', orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=2):
        self.color_space = cspace
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.hog_channel = hog_channel

    # Define a function to return HOG features and visualization
    def get_hog_features(self, img, vis=False, feature_vec=True):
        if vis:
            features, hog_image = hog(img, orientations=self.orient, pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                                      cells_per_block=(self.cell_per_block, self.cell_per_block), transform_sqrt=True,
                                      visualize=vis, feature_vector=feature_vec)
            return features, hog_image

        # Otherwise call with one output
        else:
            features = hog(img, orientations=self.orient, pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                           cells_per_block=(self.cell_per_block, self.cell_per_block), transform_sqrt=True,
                           visualize=vis, feature_vector=feature_vec)
            return features

    # Define a function to extract features from a list of images
    # Have this function call bin_spatial() and color_hist()
    def extract_features(self, imgs):
        features = []
        for image in imgs:
            img_features = self.single_img_features(image)
            features.append(img_features)
        return features

    def single_img_features(self, img, hog_f=True):
        img_features = []
        img = mpimg.imread(img)

        if self.color_space == 'RGB':
            feature_image = np.copy(img)
        elif self.color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif self.color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif self.color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif self.color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif self.color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        else:
            return -1

        if hog_f:
            if self.hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.extend(self.get_hog_features(feature_image[:, :, channel], vis=False, feature_vec=True))
            else:
                hog_features = self.get_hog_features(feature_image[:, :, self.hog_channel], vis=False, feature_vec=True)

            img_features.append(hog_features)

        return np.concatenate(img_features)
