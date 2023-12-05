import cv2
import os
import re
import numpy as np
import matplotlib.pyplot as plt


line_y = 10


def extract_numeric_part(filename):
    numeric_part = re.sub(r'\D', '', filename)
    return int(numeric_part) if numeric_part else float('inf')


col_frames = os.listdir('frames/')

# sort file names
col_frames.sort(key=lambda f: extract_numeric_part(f))

# empty list to store the frames
col_images = []

for i in col_frames:
    # read the frames
    img = cv2.imread('frames/'+i)
    # append the frames to the list
    col_images.append(img)

i = 140

for frame in [i, i+1]:
    plt.imshow(cv2.cvtColor(col_images[frame], cv2.COLOR_BGR2RGB))
    plt.title("frame: " + str(frame))
    plt.show()

grayA = cv2.cvtColor(col_images[i], cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(col_images[i+1], cv2.COLOR_BGR2GRAY)

# plot the image after frame differencing
plt.imshow(cv2.absdiff(grayB, grayA), cmap='gray')
plt.show()

diff_image = cv2.absdiff(grayB, grayA)

# perform image thresholding
ret, thresh = cv2.threshold(diff_image, 30, 255, cv2.THRESH_BINARY)

# plot image after thresholding
plt.imshow(thresh, cmap='gray')
plt.show()

# apply image dilation
kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(thresh,kernel,iterations=1)

# plot dilated image
plt.imshow(dilated, cmap='gray')
plt.show()

plt.imshow(dilated)
cv2.line(dilated, (0, line_y),(256,line_y),(100, 0, 0))
plt.show()

valid_cntrs = []
contours, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

for i,cntr in enumerate(contours):
    x,y,w,h = cv2.boundingRect(cntr)
    if (x <= 200) & (y >= line_y) & (cv2.contourArea(cntr) >= 25):
        valid_cntrs.append(cntr)

# count of discovered contours
len(valid_cntrs)

dmy = col_images[i].copy()

cv2.drawContours(dmy, valid_cntrs, -1, (127, 200,0), 2)
cv2.line(dmy, (0, line_y), (256, line_y), (100, 255, 255))
plt.imshow(dmy)
plt.show()

# kernel for image dilation
kernel = np.ones((4, 4), np.uint8)

# font style
font = cv2.FONT_HERSHEY_SIMPLEX

