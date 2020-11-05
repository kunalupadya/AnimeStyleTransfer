import torch
import os
import numpy as np
import argparse
from PIL import Image
import cv2

def adjust_gamma(input_array, gamma=1.0):
    # build a lookup table mapping the BGR pixel values [0, 255] to
    # their adjusted gamma values
    img = cv2.cvtColor(input_array, cv2.COLOR_RGB2BGR)
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    output = cv2.LUT(img, table)
    return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default = 'test_img')
parser.add_argument('--load_size', default = 450)
parser.add_argument('--output_dir', default = 'preprocess_img')
parser.add_argument('--gamma', default = 1.5) # inverse gamma value

opt = parser.parse_args()

valid_ext = ['.jpg', '.png']

if not os.path.exists(opt.output_dir): os.mkdir(opt.output_dir)

for files in os.listdir(opt.input_dir):
	ext = os.path.splitext(files)[1]
	if ext not in valid_ext:
		continue
	# load image
	input_image = Image.open(os.path.join(opt.input_dir, files)).convert("RGB")
	# resize image, keep aspect ratio
	h = input_image.size[0]
	w = input_image.size[1]
	ratio = h *1.0 / w
	if ratio > 1:
		h = opt.load_size
		w = int(h*1.0/ratio)
	else:
		w = opt.load_size
		h = int(w * ratio)
	input_image = input_image.resize((h, w), Image.BICUBIC)
	
	#PREPROCESS HERE
	#output_image = anisotropic_diffusion(np.asarray(input_image), niter=10, kappa=50, gamma=0.1, voxelspacing=None, option=1)
	output_image = np.array(input_image)
	output_image = cv2.ximgproc.anisotropicDiffusion(src = np.array(input_image), dst = output_image, alpha = 0.1, K = 0.9, niters=2)
	output_image = adjust_gamma(output_image, float(opt.gamma))

	img = Image.fromarray(output_image, 'RGB')
	img.save(os.path.join(opt.output_dir, files[:-4] + '.jpg'))
	#input_image = np.asarray(input_image)
	# RGB -> BGR
	#input_image = input_image[:, :, [2, 1, 0]]
	#input_image = transforms.ToTensor()(input_image).unsqueeze(0)
	# preprocess, (-1, 1)
	#input_image = -1 + 2 * input_image 
	# forward
	#output_image = input_image
	#output_image = output_image[0]
	# BGR -> RGB
	#output_image = output_image[[2, 1, 0], :, :]
	# deprocess, (0, 1)
	#output_image = output_image.data.cpu().float() * 0.5 + 0.5
	# save
	#vutils.save_image(output_image, os.path.join(opt.output_dir, files[:-4] + '_'  + '.jpg'))

print('Done!')
