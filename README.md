# CartoonGAN-Test-Pytorch-Torch
Pytorch and Torch testing code of [CartoonGAN](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/2205.pdf) `[Chen et al., CVPR18]`. With the released pretrained [models](http://cg.cs.tsinghua.edu.cn/people/~Yongjin/Yongjin.htm) by the authors, I made these simple scripts for a quick test.

<p>
    <img src='test_output/demo_ori.gif' width=300 />
    <img src='test_output/demo.gif' width=300 />
</p>

## Getting started: Anime Intro
- NOTE: I'd recommend doing the conda/pip commands from Anaconda Prompt (and run it as adminstrator). I had trouble with writing permissions for those installs without the administrator run.
- To install Pytorch 0.3, (note these will take a ludicrous amount of time). The windows version gives a lot of stuff
- - (Windows) conda install -c peterjc123 pytorch
- - (Mac) conda install pytorch=0.3.1 cuda90 -c pytorch
-  Torch has likely already installed by the previous step, but if not,
- - pip install torch
- pip install torchvision
- pip install opencv-contrib-python
- If you get any sort of complaint related to CUDA, you need to be running this code on a machine with an NVIDIA GPU and CUDA installed. The link to CUDA installation is here: https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html
- - To make sure your version of torch is compiled with CUDA, run:
- - - conda install -c pytorch torchvision cudatoolkit=10.2 pytorch
- With all of that set up for windows, you should be set to run the commands below for testing
## Getting started

- Linux
- NVIDIA GPU
- Pytorch 0.3 
- Torch

```
git clone https://github.com/Yijunmaverick/CartoonGAN-Test-Pytorch-Torch
cd CartoonGAN-Test-Pytorch-Torch
```

## Pytorch

The original pretrained models are Torch `nngraph` models, which cannot be loaded in Pytorch through `load_lua`. So I manually copy the weights (bias) layer by layer and convert them to `.pth` models. 

- Download the converted models:

```
sh pretrained_model/download_pth.sh
(If this doesn't work, you are missing the sh command. If you have git terminal installed, you can type .\pretrained_model/download_pth.sh.
You may also be missing the wget command, which you can isntall and setup from this link: http://gnuwin32.sourceforge.net/packages/wget.htm)
```

- For testing:

```
python test.py --input_dir YourImgDir --style Hosoda --gpu 0
```

## Torch

Working with the original models in Torch is also fine. I just convert the weights (bias) in their models from CudaTensor to FloatTensor so that `cudnn` is not required for loading models.

- Download the converted models:

```
sh pretrained_model/download_t7.sh
```

- For testing:

```
th test.lua -input_dir YourImgDir -style Hosoda -gpu 0
```

## Examples (Left: input, Right: output)

<p>
    <img src='test_img/in2.png' width=300 />
    <img src='test_output/in2_Hayao.png' width=300 />
</p>

<p>
    <img src='test_img/in3.png' width=300 />
    <img src='test_output/in3_Hayao.png' width=300 />
</p>

<p>
    <img src='test_img/5--26.jpg' width=300 />
    <img src='test_output/5--26_Hosoda.jpg' width=300 />
</p>

<p>
    <img src='test_img/7--136.jpg' width=300 />
    <img src='test_output/7--136_Hayao.jpg' width=300 />
</p>

<p>
    <img src='test_img/15--324.jpg' width=300 />
    <img src='test_output/15--324_Hosoda.jpg' width=300 />
</p>


## Note

- The training code should be similar to the popular GAN-based image-translation frameworks and thus is not included here.

## Acknowledgement

- Many thanks to the authors for this cool work.

- Part of the codes are borrowed from [DCGAN](https://github.com/soumith/dcgan.torch), [TextureNet](https://github.com/DmitryUlyanov/texture_nets), [AdaIN](https://github.com/xunhuang1995/AdaIN-style) and [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

