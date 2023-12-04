import cv2
from skimage.feature import hog
import matplotlib.image as mpimg
import numpy as np


class Util:

    def __init__(self):
        pass

    # Define a function to return HOG features and visualization
    def get_hog_features(self, img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
        if vis:
            features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                      visualize=vis, feature_vector=feature_vec)
            return features, hog_image

        # Otherwise call with one output
        else:
            features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                           cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                           visualize=vis, feature_vector=feature_vec)
            return features

    # Define a function to extract features from a list of images
    # Have this function call bin_spatial() and color_hist()
    def extract_features(self, imgs, cspace='RGB', orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=2):
        # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
        for file in imgs:
            # Read in each one by one
            image = mpimg.imread(file)
            # apply color conversion if other than 'RGB'
            if cspace == 'RGB':
                feature_image = np.copy(image)
            elif cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(self.get_hog_features(feature_image[:, :, channel],
                                              orient, pix_per_cell, cell_per_block,
                                              vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = self.get_hog_features(feature_image[:, :, hog_channel], orient,
                                                     pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            features.append(hog_features)
        return features

    def single_img_features(self, img, color_space='RGB', orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=2,
                            hog_f=True):

        img_features = []

        if color_space == 'RGB':
            feature_image = np.copy(img)
        elif color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        else:
            return -1

        if hog_f:
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.extend(self.get_hog_features(feature_image[:, :, channel],
                                                              orient, pix_per_cell, cell_per_block,
                                                              vis=False, feature_vec=True))
            else:
                hog_features = self.get_hog_features(feature_image[:, :, hog_channel], orient,
                                                     pix_per_cell, cell_per_block, vis=False, feature_vec=True)

            img_features.append(hog_features)

        return np.concatenate(img_features)
