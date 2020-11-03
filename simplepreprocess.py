import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib import cm
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
from skimage.feature import canny
import traceback

# credit given to Github user JeeveshN
      
CASCADE="Face_cascade.xml"
FACE_CASCADE=cv2.CascadeClassifier(CASCADE)

def detect_faces(image_path, name):

	image=cv2.imread(image_path)
	image_grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

	faces = FACE_CASCADE.detectMultiScale(image_grey,scaleFactor=1.16,minNeighbors=5,minSize=(25,25),flags=0)

	listToReturn = []


	for x,y,w,h in faces:
		sub_img=image[y:y+h,x:x+w]
		os.chdir("simple")
		cv2.imwrite(str(name)+".jpg",sub_img)
		os.chdir("../")
		# cv2.rectangle(image,(x,y),(x+w,y+h),(255, 255,0),2)
		listToReturn.append((sub_img,[y,y+h,x,x+w]))

	return (listToReturn)

# Credit to the opencv / scikit pages for example code

# uses Canny Edge Detection (similar to CartoonGAN) given a path to input image
def canny_edge_detection(path_to_image): 
    img = cv2.imread(path_to_image)
    edges = cv2.Canny(img)
    # plt.subplot(121),plt.imshow(img,cmap = 'gray')
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    # plt.show()
    return edges
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


def master(path_to_img):
    img = cv2.imread(path_to_img,-1)
    face_extraction = detect_faces(path_to_img, "jo") 
    # print(face_extraction)
    edges = cv2.Canny(face_extraction[0][0],100,200)
    edges = 255 - edges
    edges = cv2.cvtColor(edges,cv2.COLOR_GRAY2RGB)

    # h, w, c = edges.shape
    # edges = np.concatenate([edges, np.full((h, w, 1), 255, dtype=np.uint8)], axis=-1)
    # white = np.all(edges == [255, 255, 255], axis=-1)
    # edges[white, -1] = 0
   
    # h2, w2, c2 = img.shape
    # img = np.concatenate([img, np.full((h2, w2, 1), 255, dtype=np.uint8)], axis=-1)
    # img[white, -1] = 0

    black = np.zeros((face_extraction[0][1][1]-face_extraction[0][1][0], face_extraction[0][1][3]-face_extraction[0][1][2], 3), np.uint8)
    h3, w3, c3 = black.shape
    black = np.concatenate([black, np.full((h3, w3, 1), 255, dtype=np.uint8)], axis=-1)

    print(img.shape)
    print(edges.shape)
    print(black.shape)

    for row in range(edges.shape[0]):
        for col in range(edges.shape[1]):
            if edges[row,col][0] == 0 and edges[row,col][1] == 0 and edges[row,col][2] == 0:
                # print("hi")
                img[row+(face_extraction[0][1][0]),col+(face_extraction[0][1][2])] = [0,0,0]


    # img[face_extraction[0][1][0]:face_extraction[0][1][1],face_extraction[0][1][2]:face_extraction[0][1][3]] = cv2.addWeighted(img[face_extraction[0][1][0]:face_extraction[0][1][1],face_extraction[0][1][2]:face_extraction[0][1][3]],0.6,edges,0.4,0)
    # img[face_extraction[0][1][0]:face_extraction[0][1][1],face_extraction[0][1][2]:face_extraction[0][1][3]] = cv2.addWeighted(img[face_extraction[0][1][0]:face_extraction[0][1][1],face_extraction[0][1][2]:face_extraction[0][1][3]],0.6,black,0.1,0)
    plt.subplot(121),plt.imshow(img)
    plt.title('New'), plt.xticks([]), plt.yticks([])
    cv2.imwrite("new.jpg", img)
    # plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()

master("0.jpg")


