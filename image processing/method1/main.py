import cv2
from time import sleep


class Main:
    def __init__(self, scale_factor=1.1, min_neighbors=2):
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.car_cascade = cv2.CascadeClassifier('haarcascade_cars.xml')

    def img(self, img_path):
        # Load the image
        image = cv2.imread(img_path)
        if image is None:
            print("Error: Could not read the image.")
            return -1
        return self.img_to_name(image)

    def img_to_name(self, image):

        # Display the original image
        # cv2.imshow('Original Image', image)
        # cv2.waitKey(2000)  # Display the original image for 2 seconds

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect cars in the image
        cars = self.car_cascade.detectMultiScale(gray, self.scale_factor, self.min_neighbors)

        # Draw rectangles around the detected cars
        for (x, y, w, h) in cars:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Display the image with rectangles
        cv2.imshow('Detected Cars', image)
        cv2.waitKey(300000)  # Display the image with rectangles for 5 seconds

        cv2.destroyAllWindows()
        print(cars)

        return cars

    def vid(self, vid_path, frame_skip):
        # Capture video/ video path
        cap = cv2.VideoCapture(vid_path)

        if cap is None:
            print("Error: Could not read the image.")
            return -1

        cars = []

        while True:
            # Read frame
            ret, frame = cap.read()

            # Break the loop if there are no more frames
            if not ret:
                break

            # Process the frame and append to the list
            cars.append(self.img_to_name(frame))

            # Skip frames
            for _ in range(frame_skip - 1):
                ret = cap.read()[0]
                if not ret:
                    break

        # Release the video capture object
        cap.release()

        return cars


k = Main()
# print(k.img("26.png"))
print(len(k.vid("q.mp4", 30)))
