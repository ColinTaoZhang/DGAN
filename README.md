# DGAN

This is PyTorch implementation for [Deep Guided Attention Network for Joint Denoising and Demosaicing in Real Image](https://cje.ejournal.org.cn/en/article/doi/10.23919/cje.2022.00.414), Chinese Journal of Electronics, by Tao Zhang, Ying Fu and Jun Zhang.

## Introduction
In this work, we propose a deep guided attention network for real image joint denoising and demosaicing, which considers the high signal-to-noise ratio and high sampling rate of green information for denoising and demosaicing, respectively.
<div align='center'>
  <img src="https://github.com/ColinTaoZhang/DGAN/blob/main/E220414-1.jpg" alt="alt text" style="width:500px; height:auto;">
</div>

## Highlights
* We present a deep guided attention network for real image JDD, that effectively considers the green channel characteristics of high SNR and high sampling rate in raw data.
<div align='center'>
  <img src="https://github.com/ColinTaoZhang/DGAN/blob/main/E220414-2.jpg" alt="alt text" style="width:700px; height:auto;">
</div>

* We propose a guided attention module to adaptively guide RGB image restoration by the information in green channel recovery branch.
<div align='center'>
  <img src="https://github.com/ColinTaoZhang/DGAN/blob/main/E220414-3.jpg" alt="alt text" style="width:500px; height:auto;">
</div>

## Usage
* Training
```
python train_denoising.py
python train_JDD.py
```

* Testing
```
python test_JDD.py
```
## Citation
If you find this work useful for your research, please cite:
```
@articleInfo{zhang2024DGAN,
title = "Deep Guided Attention Network for Joint Denoising and Demosaicing in Real Image",
journal = "Chinese Journal of Electronics",
volume = "33",
number = "E220414,
pages = "303",
year = "2024"
}
