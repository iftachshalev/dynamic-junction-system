import glob
import matplotlib.pyplot as plt
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import util
import cv2
import numpy as np
import joblib
import matplotlib.image as mpimg


# default - color_space='RGB', orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=2, train=False


class Ai:

    VEHICLE_PATH = "data/vehicles/**/*.png"
    NON_VEHICLE_PATH = "data/non-vehicles/**/*.png"
    X_SCALER_FILE = "x_scaler.joblib"
    SVC_FILE = 'svm_model.joblib'

    def __init__(self, color_space='RGB', orient=111, pix_per_cell=8, cell_per_block=2, hog_channel=2, train=False):
        self.vehicle_image_filenames = glob.glob(self.VEHICLE_PATH, recursive=True)
        self.non_vehicle_image_filenames = glob.glob(self.NON_VEHICLE_PATH, recursive=True)
        self.UTIL = util.Util(color_space, orient, pix_per_cell, cell_per_block, hog_channel)
        if train:
            self.train_svm()

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

        vehicle_features, vehicle_hog_image = self.UTIL.get_hog_features(ycrcb_vehicle_img[:, :, 0], vis=True)
        non_vehicle_features, non_vehicle_hog_image = self.UTIL.get_hog_features(ycrcb_non_vehicle_img[:, :, 0],vis=True)

        figure, (vehicle_hog_plot, non_vehicle_hog_plot) = plt.subplots(1, 2, figsize=(8, 4))

        vehicle_hog_plot.set_title('Vehicle HOG feature')
        vehicle_hog_plot.imshow(vehicle_hog_image, cmap='gray')

        non_vehicle_hog_plot.set_title('Non-vehicle HOG feature')
        non_vehicle_hog_plot.imshow(non_vehicle_hog_image, cmap='gray')

        plt.show()

    def train_svm(self):
        x_train, x_test, y_train, y_test = self.UTIL.setup_for_training(self.vehicle_image_filenames,
                                                                        self.non_vehicle_image_filenames,
                                                                        self.X_SCALER_FILE)
        print('Training data set size: ', len(x_train))
        print('Testing data set size: ', len(x_test))

        # Use a linear SVC
        svc = LinearSVC(dual="auto")

        # Check the training time for the SVC
        t = time.time()
        svc.fit(x_train, y_train)
        t2 = time.time()

        print(round(t2 - t, 2), 'Seconds to train SVC...')

        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(svc.score(x_test, y_test), 4))

        # Save the trained model to a file
        joblib.dump(svc, self.SVC_FILE)
        print(f"Trained model saved to {self.SVC_FILE}")

    def img_check(self, img_path):
        hog_features = self.UTIL.single_img_features(img_path)
        x_scaler = joblib.load(self.X_SCALER_FILE)
        svc = joblib.load(self.SVC_FILE)
        scaled_features = x_scaler.transform(np.array(hog_features).reshape(1, -1))
        prediction = svc.predict(scaled_features)
        print(f"prediction: {prediction[0]}")

    def predict(self, img_path):
        t = time.time()
        # Read the image
        image = mpimg.imread(img_path)

        # Resize the image to match the expected size
        resized_image = cv2.resize(image, (64, 64))

        # Extract features from the resized image
        hog_features = self.UTIL.single_img_features(resized_image)

        # Load scaler and classifier
        x_scaler = joblib.load(self.X_SCALER_FILE)
        svc = joblib.load(self.SVC_FILE)

        # Transform and predict
        scaled_features = x_scaler.transform(np.array(hog_features).reshape(1, -1))
        prediction = svc.predict(scaled_features)
        print(f"Prediction: {prediction[0]}")
        t2 = time.time()
        print(f"time: {round(t2 - t, 2)}")


ai = Ai(train=True)
ai.predict("test_images/c.png")
