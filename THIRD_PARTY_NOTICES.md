# Third-Party Notices

This project uses several open-source libraries and frameworks. Below are their
licenses and acknowledgments:

## PyTorch
Copyright (c) 2016-present, Facebook Inc.
Licensed under the BSD 3-Clause License
https://github.com/pytorch/pytorch/blob/master/LICENSE

## Segmentation Models PyTorch
Copyright (c) 2019, Pavel Yakubovskiy
Licensed under the MIT License
https://github.com/qubvel/segmentation_models.pytorch/blob/master/LICENSE

## Albumentations
Copyright (c) 2018, Buslaev Alexander, Alexander Parinov, Vladimir I. Iglovikov, Eugene Khvedchenya, Alexandr A. Kalinin
Licensed under the MIT License
https://github.com/albumentations-team/albumentations/blob/master/LICENSE

## OpenCV
Copyright (c) 2024, OpenCV team
Licensed under the Apache License 2.0
https://github.com/opencv/opencv/blob/master/LICENSE

## NumPy
Copyright (c) 2005-2024, NumPy Developers
Licensed under the BSD 3-Clause License
https://github.com/numpy/numpy/blob/main/LICENSE.txt

## Pillow
Copyright (c) 1997-2011 by Secret Labs AB
Copyright (c) 1995-2011 by Fredrik Lundh
Licensed under the PIL Software License
https://github.com/python-pillow/Pillow/blob/main/LICENSE

## scikit-learn
Copyright (c) 2007-2024 The scikit-learn developers
Licensed under the BSD 3-Clause License
https://github.com/scikit-learn/scikit-learn/blob/main/COPYING

## torchmetrics
Copyright (c) 2020-2024, PyTorch Lightning team
Licensed under the Apache License 2.0
https://github.com/Lightning-AI/torchmetrics/blob/master/LICENSE

## TensorBoard
Copyright (c) 2015 The TensorFlow Authors
Licensed under the Apache License 2.0
https://github.com/tensorflow/tensorboard/blob/master/LICENSE

## ONNX
Copyright (c) ONNX Project Contributors
Licensed under the Apache License 2.0
https://github.com/onnx/onnx/blob/main/LICENSE

## LabelImg
Copyright (c) <2015-Present> Tzutalin
Copyright (C) 2013  MIT, Computer Science and Artificial Intelligence Laboratory. Bryan Russell,
Antonio Torralba, William T. Freeman
The Pascal VOC, YOLO and CreateML label file readers in `engine/studio.py` and
`app/src/shared/formats.ts` are ported from LabelImg's `libs/pascal_voc_io.py`, `yolo_io.py` and
`create_ml_io.py`, and carry its notice.
Licensed under the MIT License
https://github.com/HumanSignal/labelImg/blob/master/LICENSE

## Inter
Copyright 2016 The Inter Project Authors
Licensed under the SIL Open Font License 1.1
https://github.com/rsms/inter/blob/master/LICENSE.txt

## VisA dataset
The photos in `docs/media` are from VisA, apart from the MVTec AD and D-Fire photos in the
banner images below. VisA is by Yang Zou, Jongheon Jeong, Latha Pemula,
Dongqing Zhang and Onkar Dabeer. They are shown cropped, with labels drawn on them.
Licensed under Creative Commons Attribution 4.0 International
https://github.com/amazon-science/spot-diff

## MVTec AD dataset
The banner images (`docs/media/banner.webp`, `banner-animated.webp` and `banner-phone.webp`) include
photos and defect masks from MVTec AD, by MVTec Software GmbH (Paul Bergmann, Kilian Batzner,
Michael Fauser, David Sattlegger and Carsten Steger). They are shown cropped, in greys on one side,
with labels drawn on them. Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International, so the banner images are shared under the same license, not MIT
https://www.mvtec.com/company/research/datasets/mvtec-ad

## D-Fire dataset
The banner images also include photos and boxes from D-Fire, by Pedro Vinícius Almeida Borges de
Venâncio, Adriano Chaves Lisboa and Adriano Vilela Barbosa. None of its images from third-party
public datasets are used. Released under Creative Commons Zero v1.0 Universal
https://github.com/gaiasd/DFireDataset

## Pretrained encoder weights
Seamline doesn't ship model weights. With **Start from ImageNet weights**, segmentation_models_pytorch
downloads the encoder's ImageNet weights from its mirrors on the Hugging Face Hub (`smp-hub/<encoder>.imagenet`),
which are copies of these originals. The Train page offers only encoders whose weights allow commercial use.
- ResNet-18, 34 and 50, and MobileNetV2: torchvision, by the PyTorch team, licensed under the BSD 3-Clause
  License. https://github.com/pytorch/vision
- EfficientNet-B0 and B4: EfficientNet-PyTorch, by Luke Melas-Kyriazi, ported from Google's TensorFlow
  weights, licensed under the Apache License 2.0. https://github.com/lukemelas/EfficientNet-PyTorch
- DenseNet-121: pretrained-models.pytorch, by Remi Cadene, licensed under the BSD 3-Clause License.
  https://github.com/Cadene/pretrained-models.pytorch

All were trained on ImageNet. As torchvision's documentation says of its own, pretrained weights may carry
terms that come from the data they were trained on, and whether a use is permitted is for their user to judge.
SegFormer's MiT encoders aren't offered: NVIDIA licenses their weights for non-commercial use only.
