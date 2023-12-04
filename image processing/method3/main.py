import glob
import matplotlib.pyplot as plt
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import util
import cv2
import numpy as np


class Ai:

    VEHICLE_PATH = "data/vehicles/**/*.png"
    NON_VEHICLE_PATH = "data/non-vehicles/**/*.png"
    UTIL = util.Util()

    def __init__(self, pix_per_cell, cell_per_block, orient, color_space, hog_channel):
        self.vehicle_image_filenames = glob.glob(self.VEHICLE_PATH, recursive=True)
        self.non_vehicle_image_filenames = glob.glob(self.NON_VEHICLE_PATH, recursive=True)
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.orient = orient
        self.color_space = color_space
        self.hog_channel = hog_channel

    def show_plt_ex(self, vehicle_idx, non_vehicle_idx):
        if len(self.vehicle_image_filenames) <= vehicle_idx:
            vehicle_img = cv2.imread(self.vehicle_image_filenames[-1])
        else:
            vehicle_img = cv2.imread(self.vehicle_image_filenames[vehicle_idx])
        if len(self.non_vehicle_image_filenames) <= non_vehicle_idx:
            non_vehicle_img = cv2.imread(self.non_vehicle_image_filenames[-1])
        else:
            non_vehicle_img = cv2.imread(self.non_vehicle_image_filenames[non_vehicle_idx])

        figure, (vehicle_plot, non_vehicle_plot) = plt.subplots(1, 2, figsize=(8, 4))
        vehicle_plot.set_title('Vehicle image')
        vehicle_plot.imshow(cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2RGB))
        non_vehicle_plot.set_title('Non-vehicle image')
        non_vehicle_plot.imshow(cv2.cvtColor(non_vehicle_img, cv2.COLOR_BGR2RGB))
        plt.show()

    def show_hog_ex(self, vehicle_idx, non_vehicle_idx):
        if len(self.vehicle_image_filenames) <= vehicle_idx:
            vehicle_img = cv2.imread(self.vehicle_image_filenames[-1])
        else:
            vehicle_img = cv2.imread(self.vehicle_image_filenames[vehicle_idx])
        if len(self.non_vehicle_image_filenames) <= non_vehicle_idx:
            non_vehicle_img = cv2.imread(self.non_vehicle_image_filenames[-1])
        else:
            non_vehicle_img = cv2.imread(self.non_vehicle_image_filenames[non_vehicle_idx])

        ycrcb_vehicle_img = cv2.cvtColor(vehicle_img, cv2.COLOR_RGB2YCrCb)
        ycrcb_non_vehicle_img = cv2.cvtColor(non_vehicle_img, cv2.COLOR_RGB2YCrCb)

        vehicle_features, vehicle_hog_image = self.UTIL.get_hog_features(ycrcb_vehicle_img[:, :, 0], self.orient,
                                                                         self.pix_per_cell,
                                                                         self.cell_per_block,
                                                                         True)
        non_vehicle_features, non_vehicle_hog_image = self.UTIL.get_hog_features(ycrcb_non_vehicle_img[:, :, 0],
                                                                                 self.orient,
                                                                                 self.pix_per_cell,
                                                                                 self.cell_per_block, True)

        figure, (vehicle_hog_plot, non_vehicle_hog_plot) = plt.subplots(1, 2, figsize=(8, 4))

        vehicle_hog_plot.set_title('Vehicle HOG feature')
        vehicle_hog_plot.imshow(vehicle_hog_image, cmap='gray')

        non_vehicle_hog_plot.set_title('Non-vehicle HOG feature')
        non_vehicle_hog_plot.imshow(non_vehicle_hog_image, cmap='gray')

        plt.show()

    def train_svm(self):
        t1 = time.time()
        vehicle_hog_features = self.UTIL.extract_features(self.vehicle_image_filenames, self.color_space, self.orient,
                                                          self.pix_per_cell, self.cell_per_block, self.hog_channel)

        non_vehicle_hog_features = self.UTIL.extract_features(self.non_vehicle_image_filenames,
                                                              self.color_space, self.orient, self.pix_per_cell,
                                                              self.cell_per_block, self.hog_channel)
        t2 = time.time()
        print(round(t2 - t1, 2), 'Seconds to extract HOG features...')

        # Create an array stack of feature vectors
        X = np.vstack((vehicle_hog_features, non_vehicle_hog_features)).astype(np.float64)
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)
        # Define the labels vector
        y = np.hstack((np.ones(len(vehicle_hog_features)), np.zeros(len(non_vehicle_hog_features))))
        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

        print('Using:', self.orient, 'orientations', self.pix_per_cell, 'pixels per cell and', self.cell_per_block,
              'cells per block')
        print('Training data set size: ', len(X_train))
        print('Testing data set size: ', len(X_test))

        # Use a linear SVC
        svc = LinearSVC(dual="auto")

        # Check the training time for the SVC
        t = time.time()
        svc.fit(X_train, y_train)
        t2 = time.time()

        print(round(t2 - t, 2), 'Seconds to train SVC...')

        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

        # Check the prediction time for a single sample
        t = time.time()
        n_predict = 10

        print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
        print('For these', n_predict, 'labels: ', y_test[0:n_predict])

        t2 = time.time()

        print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')


ai = Ai(8, 2, 9, "YCrCb", "ALL")
ai.show_plt_ex(10, 10)
ai.show_hog_ex(10, 10)
ai.train_svm()
