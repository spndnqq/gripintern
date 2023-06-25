"""
    Name: Spandan Halder
    The Sparks Foundation (GRIP Internship)
    # Task 2: Color Identification in Images
"""

# Importing Libraries
from collections import Counter

import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# Function to convert RGB to HEX format
def rgb2hex(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


# Loading Image
image = cv2.resize(cv2.imread('image2.jpg'), (500, 600), interpolation=cv2.INTER_AREA)
print("Shape: {}".format(image.shape))
modified_image = image.reshape(image.shape[0] * image.shape[1], 3)

# No of Colours
number_of_colors = eval(input('Enter Number of Colours: '))

# Using K-Means Algorithm
clf = KMeans(n_clusters=number_of_colors)
labels = clf.fit_predict(modified_image)
counts = Counter(labels)
center_colors = clf.cluster_centers_
ordered_colors = [center_colors[i] for i in counts.keys()]
hex_colors = [rgb2hex(ordered_colors[i]) for i in counts.keys()]
rgb_colors = [ordered_colors[i] for i in counts.keys()]

# Showing the Image and Plotting a Pie Chart of Identified Colors
plt.figure(figsize=(8, 6))
plt.pie(counts.values(), labels=hex_colors, colors=hex_colors)
cv2.imshow('Original Image', image)
plt.show()
cv2.waitKey(0)
