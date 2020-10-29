import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib import cm
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
from skimage.feature import canny

# Credit to the opencv / scikit pages for example code

# uses Canny Edge Detection (similar to CartoonGAN) given a path to input image
def canny_edge_detection(path_to_image): 
    img = cv2.imread(path_to_image,0)
    edges = cv2.Canny(img,100,200)
    plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()
    # concatenate_edge_with_image(edges, img)

# using (Probabilistic) Hough Transform, as taught in class to detect significant edges within an image
def hough_edge_detection(path_to_image): 
    img = cv2.imread(path_to_image,0)
    edges = canny(img, 2, 1, 25)
    lines = probabilistic_hough_line(edges, threshold=10, line_length=5,
                                 line_gap=3)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(edges * 0)
    for line in lines:
        p0, p1 = line
        ax[0].plot((p0[0], p1[0]), (p0[1], p1[1]))
    ax[0].set_xlim((0, img.shape[1]))
    ax[0].set_ylim((img.shape[0], 0))
    ax[0].set_title('Probabilistic Hough')

    for a in ax:
        a.set_axis_off()

    plt.tight_layout()
    plt.show()






# given an image of just edges and an original image, produce an image with a combination (pure addition) of the two
def concatenate_edge_with_image(edge_image, original_image): 
    # combined_image = edge_image + original_image
    combined_image = cv2.addWeighted(original_image,0.3,edge_image,0.7,0)
    plt.subplot(121),plt.imshow(original_image,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(combined_image,cmap = 'gray')
    plt.title('Combined Image'), plt.xticks([]), plt.yticks([])
    plt.show()


# edge_detection("0.jpg")
hough_edge_detection("0.jpg")