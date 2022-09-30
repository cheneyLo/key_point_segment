PyTorch code used to train a deep learning model for key points recognition of Maxillofacial bone in CT images .

The example was tested with Python (v3.7.0), PyTorch (v1.8.0+cu101) and TorchVision (v0.11.1) on Ubuntu 16.04.5 and Windows 10.

To run the example `train.sh` for each task, you need to install `pytorch` and `torchvision`.
For detail installation procedure, please refer to https://pytorch.org.

The example can't be run without GPU. However, GPUs are recommended to use if training with large-scale image data.

## 1. Installation (10 min)
```bash
pip install torch==1.8.0 torchvision==0.11.1 visdom einops SimpleITK zipfile numpy nibabel gzip subprocess
```

## 2. Git clone (1 min)
```bash
git clone https://github.com/cheneyLo/key_point_segment
```

## 3. Run key_point_segment example (10 min)
```bash
cd key_point_segment

python start_train_local.py

 
```

## 4. Expected output
Model checkpoint file in weight folder.

