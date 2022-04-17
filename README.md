# Medical Segmentation

Train the network to segmentate tumor and organs body from CT 3D image.

## Requirement
- MONAI
- Pytorch
- Visdom

## Usage

Consider the enviornment of single node with four gpus. In the following examples, one can open http://localhost:8097 to see the plot of loss/accuracy, `8097` is the default port number.

### Basic Usage
Consider the following directory tree

```
Medical Segmentation/
├── images
│   ├── image0.nii.gz
│   └── image1.nii.gz
├── labels
│   ├── label0.nii.gz
│   └── label1.nii.gz
├── dataset.py
├── engine.py
├── main.py
├── model.py
├── parser.py
├── tracer.py
└── visualizer.py

```
The basic syntax to run the code
```
torchrun --standalone --nnodes=1 --nproc_per_node=4 main.py --img_dir images --lab_dir labels --roi 96 96 96 --spacing 1 1 1
```
