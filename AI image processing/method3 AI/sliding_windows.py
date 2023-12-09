import cv2
import numpy as np
from main import Ai
import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import label
import matplotlib.pyplot as plt


class SlidingWindows(Ai):

    def draw_boxes(self, img, bboxes, color=(0, 255, 0), thick=3):
        # Make a copy of the image
        draw_img = np.copy(img)

        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)

        # Return the image copy with boxes drawn
        return draw_img

    # Window size (x and y dimensions), and overlap fraction (for both x and y)
    def slide_window(self, img, x_start_stop, y_start_stop, xy_window=(64, 64),
                     xy_overlap=(0.5, 0.5)):
        # Compute the span of the region to be searched
        x_span = x_start_stop[1] - x_start_stop[0]
        y_span = y_start_stop[1] - y_start_stop[0]

        # Compute the number of pixels per step in x/y
        nx_pix_per_step = int(xy_window[0] * (1 - xy_overlap[0]))
        ny_pix_per_step = int(xy_window[1] * (1 - xy_overlap[1]))

        # Compute the number of windows in x/y
        nx_buffer = int(xy_window[0] * (xy_overlap[0]))
        ny_buffer = int(xy_window[1] * (xy_overlap[1]))
        nx_windows = int((x_span - nx_buffer) / nx_pix_per_step)
        ny_windows = int((y_span - ny_buffer) / ny_pix_per_step)

        # Initialize a list to append window positions to
        window_list = []

        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                start_x = xs * nx_pix_per_step + x_start_stop[0]
                end_x = start_x + xy_window[0]
                start_y = ys * ny_pix_per_step + y_start_stop[0]
                end_y = start_y + xy_window[1]

                # Append window position to list
                window_list.append(((start_x, start_y), (end_x, end_y)))

        # Return the list of windows
        return window_list

    def search_windows(self, img, windows):

        # 1) Create an empty list to receive positive detection windows
        cars_found = []

        # 2) Iterate over all windows in the list
        for window in windows:
            # 3) Extract the test window from original image
            if window[1][1] <= img.shape[0] and window[1][0] <= img.shape[1]:
                # Valid window, you can proceed with resizing or other operations
                test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            else:
                print("Invalid window coordinates.")

            # 4) Extract features for that window using single_img_features()
            features = self.UTIL.single_img_features(test_img)

            # 5) Scale extracted features to be fed to classifier
            scaler = StandardScaler().fit(features)
            test_features = scaler.transform(np.array(features).reshape(1, -1))

            # 6) Predict using your classifier
            prediction = self.svc.predict(test_features)

            # 7) If positive (prediction == 1) then save the window
            if prediction == 1:
                cars_found.append(window)

        # 8) Return windows for positive detections
        return cars_found

    def add_heat(self, heatmap, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Return updated heatmap
        return heatmap

    def apply_threshold(self, heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0

        # Return thresholded map
        return heatmap

    def draw_labeled_bboxes(self, img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()

            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)

        # Return the image
        return img

    def detect_car(self, image_path):
        image_to_process = mpimg.imread(image_path)
        draw_image = np.copy(image_to_process)

        heat = np.zeros_like(image_to_process[:, :, 0]).astype(float)

        windows = self.slide_window(image_to_process, x_start_stop=[0, 1280], y_start_stop=[380, 680],
                                    xy_window=(128, 128),
                                    xy_overlap=(0.65, 0.65))

        # draw_image = draw_boxes(draw_image, windows, color=(0, 255, 255))

        hot_windows = self.search_windows(image_to_process, windows)
        # Add heat to each box in box list
        self.add_heat(heat, hot_windows)

        windows = self.slide_window(image_to_process, x_start_stop=[0, 1280], y_start_stop=[390, 620],
                                    xy_window=(96, 96),
                                    xy_overlap=(0.65, 0.65))

        # draw_image = draw_boxes(draw_image, windows, color=(0, 255, 0))

        hot_windows = self.search_windows(image_to_process, windows)
        # Add heat to each box in box list
        self.add_heat(heat, hot_windows)

        windows = self.slide_window(image_to_process,
                               x_start_stop=[0, 1280],
                               y_start_stop=[390, 560],
                               xy_window=(72, 72),
                               xy_overlap=(0.65, 0.65))

        # draw_image = draw_boxes(draw_image, windows, color=(255, 0, 255))

        hot_windows = self.search_windows(image_to_process, windows)
        # Add heat to each box in box list
        self.add_heat(heat, hot_windows)

        windows = self.slide_window(image_to_process,
                               x_start_stop=[0, 1280],
                               y_start_stop=[390, 500],
                               xy_window=(64, 64),
                               xy_overlap=(0.65, 0.65))

        # draw_image = draw_boxes(draw_image, windows, color=(255, 0, 0))

        hot_windows = self.search_windows(image_to_process, windows)
        # Add heat to each box in box list
        self.add_heat(heat, hot_windows)

        # Apply threshold to help remove false positives
        heat = self.apply_threshold(heat, 2)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)

        self.draw_labeled_bboxes(draw_image, labels)

        return draw_image, heatmap, labels


sw = SlidingWindows()

detected_car_image, detected_car_heatmap, detected_car_labales = sw.detect_car("test_images/test4.jpg")

figure, (detected_car_image_plot, detected_car_heatmap_plot, detected_car_labales_plot) = plt.subplots(1, 3,
                                                                                                       figsize=(20, 15))

detected_car_image_plot.set_title('Detected cars')
detected_car_image_plot.imshow(detected_car_image)

detected_car_heatmap_plot.set_title('Heatmap')
detected_car_heatmap_plot.imshow(detected_car_heatmap)

detected_car_labales_plot.set_title('Labels')
detected_car_labales_plot.imshow(detected_car_labales[0], cmap='gray')

