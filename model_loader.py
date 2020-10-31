import torch
import os
import numpy as np
import argparse
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as vutils
from network.Transformer import Transformer
import matplotlib.pyplot as plt

# opt = parser.parse_args()

valid_ext = ['.jpg', '.png']

# if not os.path.exists(opt.output_dir): os.mkdir(opt.output_dir)
GPU = False
if torch.cuda.is_available():
	GPU = True

pret = "./pretrained_model"

out_dir = "test_output"

style = "Hayao"

load_size = 450

# load pretrained model
model = Transformer()
model.load_state_dict(torch.load(os.path.join(pret, style + '_net_G_float.pth')))
model.eval()

open_dir = "test_img"

if GPU:
	print('GPU mode')
	model.cuda()
else:
	print('CPU mode')
	model.float()

for files in os.listdir(open_dir):
	ext = os.path.splitext(files)[1]
	if ext not in valid_ext:
		continue
	# load image
	input_image = Image.open(os.path.join(open_dir, files)).convert("RGB")
	# resize image, keep aspect ratio
	h = input_image.size[0]
	w = input_image.size[1]
	ratio = h *1.0 / w
	if ratio > 1:
		h = load_size
		w = int(h*1.0/ratio)
	else:
		w = load_size
		h = int(w * ratio)
	input_image = input_image.resize((h, w), Image.BICUBIC)
	input_image = np.asarray(input_image)
	# RGB -> BGR
	input_image = input_image[:, :, [2, 1, 0]]
	input_image = transforms.ToTensor()(input_image).unsqueeze(0)
	# preprocess, (-1, 1)
	input_image = -1 + 2 * input_image 
	if GPU:
		input_image = Variable(input_image, volatile=True).cuda()
	else:
		input_image = Variable(input_image, volatile=True).float()
	# forward
	output_image = model(input_image)
	output_image = output_image[0]
	# BGR -> RGB
	output_image = output_image[[2, 1, 0], :, :]
	# deprocess, (0, 1)
	output_image = output_image.data.cpu().float() * 0.5 + 0.5
	# save
	vutils.save_image(output_image, os.path.join(out_dir, files[:-4] + '_' + style + '.jpg'))
	plt.imshow()
print('Done!')
