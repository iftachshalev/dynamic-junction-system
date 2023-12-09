import cv2
from skimage.feature import hog
import matplotlib.image as mpimg
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.model_selection import train_test_split


class Util:

    def __init__(self, color_space, orient, pix_per_cell, cell_per_block, hog_channel):
        self.color_space = color_space
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.hog_channel = hog_channel

    def get_hog_features(self, img, vis=False, feature_vec=True):
        if vis:
            features, hog_image = hog(img, orientations=self.orient,
                                      pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
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
    def extract_features(self, images):
        return [self.single_img_features(mpimg.imread(image)) for image in images]

    def single_img_features(self, img):

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

        if self.hog_channel == 'ALL':
            hog_features = [self.get_hog_features(feature_image[:, :, channel])
                            for channel in range(feature_image.shape[2])]
        else:
            hog_features = self.get_hog_features(feature_image[:, :, self.hog_channel])

        return hog_features

    def setup_for_training(self, vehicle_image_filenames, non_vehicle_image_filenames, x_scaler_file):
        t1 = time.time()
        vehicle_hog_features = self.extract_features(vehicle_image_filenames)

        non_vehicle_hog_features = self.extract_features(non_vehicle_image_filenames)
        t2 = time.time()
        print(round(t2 - t1, 2), 'Seconds to extract HOG features...')

        x = np.vstack((vehicle_hog_features, non_vehicle_hog_features)).astype(np.float64)
        x_scaler = StandardScaler().fit(x)
        scaled_x = x_scaler.transform(x)
        y = np.hstack((np.ones(len(vehicle_hog_features)), np.zeros(len(non_vehicle_hog_features))))
        rand_state = np.random.randint(0, 100)
        joblib.dump(x_scaler, x_scaler_file)

        return *train_test_split(scaled_x, y, test_size=0.2, random_state=rand_state), x_scaler
