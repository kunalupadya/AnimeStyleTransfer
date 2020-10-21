import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

# uses Canny Edge Detection (similar to CartoonGAN) given a path to input image
def edge_detection(path_to_image): 
    img = cv2.imread(path_to_image,0)
    edges = cv2.Canny(img,100,200)
    plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()
    concatenate_edge_with_image(edges, img)

# given an image of just edges and an original image, produce an image with a combination (pure addition) of the two
def concatenate_edge_with_image(edge_image, original_image): 
    # combined_image = edge_image + original_image
    combined_image = cv2.addWeighted(original_image,0.3,edge_image,0.7,0)
    plt.subplot(121),plt.imshow(original_image,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(combined_image,cmap = 'gray')
    plt.title('Combined Image'), plt.xticks([]), plt.yticks([])
    plt.show()


edge_detection("0.jpg")