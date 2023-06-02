# Image Classification on ImageNet, Tiny-ImageNet

This project expends `torchvision` to support training on Tiny-ImageNet and add knowledge distillation support on both ImageNet and Tiny-ImageNet.

Code is based on the official implementation for image classification in torchvision: https://github.com/pytorch/vision/tree/main/references/classification

Knowledge Distillation part is based on the official implementation for deit in timm: https://github.com/facebookresearch/deit/blob/main/README_deit.md

## Tiny-ImageNet

[training scripts](./README_tiny_imagenet.md)

### Normal Training

| name                       | acc@1 |
| -------------------------- | :---: |
| ResNet-18                  | 60.69 |
| ResNet-18 + Mixup          |       |
| ResNet-18 + Cutmix         |       |
| ResNet-18 + Mixup + Cutmix |       |
| ResNet-50                  |       |
| ResNet-101                 |       |

### Knowledge Distillation

| teacher name | student name | acc@1 |
| ------------ | ------------ | :---: |
| ResNet-18    | ResNet-18    |       |
| ResNet-50    | ResNet-18    |       |
| ResNet-101   | ResNet-18    |       |

## ImageNet

[official training scripts](./imagenet/README.md)

## Reference

Base Code:
https://github.com/pytorch/vision/tree/main/references/classification

KD Code:
https://github.com/facebookresearch/deit/blob/main/README_deit.md

Blog ImageNet Weight V1 -> V2:
https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/

ImageNet Evaluation Table:
https://pytorch.org/vision/stable/models.html
