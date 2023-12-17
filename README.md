# Lightweight Facial Attractiveness Prediction Using Dual Label Distribution
Official repo for *Lightweight Facial Attractiveness Prediction Using Dual Label Distribution* (Submitted to TCDS).

## Prerequisite

### Dependency

The code is tested on a CentOS server with a NVIDIA V100 GPU. Run the following commands to install the dependencies.

```shell
conda create --name 2D_FAP python=3.8.8
conda activate 2D_FAP
conda install pytorch=1.8.1 torchvision=0.9.1 torchaudio=0.8.1 cudatoolkit=10.1 -c pytorch
pip install pillow thop torchsummary
```

### Dataset

Extract the archive files of datasets into the `/data` folder. The download links are provided as follows.

#### SCUT-FBP5500

- Google Drive: https://drive.google.com/file/d/1ObjJ8RwYBVI71Eyo_wsM1dWzI0JbhotA/view
- Baidu Netdisk: https://pan.baidu.com/s/1YWalTRX4YwV7XGMcCXJ5Dg?pwd=157g (Pass: 157g)

#### SCUT-FBP

- Google Drive: https://drive.google.com/file/d/1KB9F2eASZgKrnL3FUaR_59XUKjns_Aim/view
- Baidu Netdisk: https://pan.baidu.com/s/13sGf7HypD_z_ZwWdPIfdug?pwd=dpmj (Pass: dpmj)

### VGG Pretrained models

Place the pretrained models with `.pth` format under the `/models` folder.

- Google Drive: https://drive.google.com/drive/folders/1-wStPO8K3fN9Xijp4gYbG_Sl64K5AFwE
- Baidu Netdisk: https://pan.baidu.com/s/1WPeVsRph3vJPNVIxV8EpjQ?pwd=wgnv (Pass:wgnv)

## Instruction

### Training & Testing

Notice that the best settings have been configured for both datasets.

```shell
# SCUT-FBP5500
python 2D_FAP.py --dataset 5500

# SCUT-FBP
python 2D_FAP.py --dataset 500
```

#### Arguments

`loss1`, `loss2`, and `loss3` in the code stand for $L_{ad}$, $L_{rd}$, and $L_{score}$ in the paper, respectively.

- `--fold`: *int*

  - Fold for cross validation on SCUT-FBP5500. Default 1.
  - Available options: 1, 2, 3, 4, 5

- `--alpha`: *int*

  - Weight for $L_{ad}$. Default 1.

- `--beta`: *int*

  - Weight for $L_{rd}$. Default 1.

- `--gamma`: *int*

  - Weight for $L_{score}$. Default 1.

- `--dataset`: *int*

  - Adopted dataset. Default 5500. (SCUT-FBP5500)
  - Available options: 5500(SCUT-FBP5500), 500(SCUT-FBP).

- `--aligned`: *bool*

  - Aligned images, only available on SCUT-FBP5500. Default True.

- `--MTCNN`: *bool*

  - MTCNN-processed SCUT-FBP5500 images. Default False.

- `--sample`: *str*

  - The adopted distribution in the attractiveness distribution. Default 'L'. (Laplace Distribution)
  - Available options: 'L' (Laplace Distribution), 'G' (Gaussian Distribution).

- `--loss1`: *str*

  - Different forms of $L_{ad}$. Default 'ED'. (Euclidean Distance)
  - Available options: 'L1' (L1 Distance), 'ED' (Euclidean Distance), 'KL' (Kullback–Leibler divergence).

- `--loss1_option`: *str*

  - Take the sum/average over a mini-batch as $L_{ad}$ for the batch. Default 'mean'.
  - Available options: 'sum', 'mean'.

- `--loss3`: *str*

  - Different forms of $L_{score}$. Default '3'.

  - Available options

    - '1': $L_{score}= \ln {[\frac12(\exp{(\hat y^{(i)}-y^{(i)})}+\exp{(y^{(i)}-\hat y^{(i)})})]}$

    - '2': $L_{score}= \ln{(1+|\hat y^{(i)}-y^{(i)}|)}$
    - '3': $L_{score}=\exp{(|\hat y^{(i)}-y^{(i)}|)-1}$
    - '4': $L_{score}= \ln{(|\hat y^{(i)}-y^{(i)}|+\sqrt{1+|\hat y^{(i)}-y^{(i)}|^2})}$

- `--loss3_option`: *str*

  - Take the sum/average over a mini-batch as $L_{score}$ for the batch. Default 'sum'.
  - Available options: 'sum', 'mean'.

- `--losses`: *str*

  - The losses used in training. Default '123'.

- `--lr`: *float*

  - Initial learning rate. Default 0.001.

- `--network`: *str*

  - Adopted network architecture. Default 'mobilenet'. (MobileNetV2)
  - Available options
    - 'mobilenet': ImageNet-pretrained MobileNetV2.
    - 'mobilenet_m': ImageNet-pretrained MobileNetV2 without the pretrained parameters on the last fc layer.
    - 'mobilenetv3_large': ImageNet-pretrained MobileNetV3_large.
    - 'mobilenetv3_small': ImageNet-pretrained MobileNetV3_small.
    - 'resnet18': ImageNet-pretrained ResNet-18.
    - 'resnet50': ImageNet-pretrained ResNet-50.
    - 'vgg16': ImageNet-pretrained VGG16.
    - 'vgg19': ImageNet-pretrained VGG19.
    - 'attnet_m': Modified AttNet [1] that adapts our task.
    - 'hmt': Modified HMTNet [2] that adapts our task.

- `--local_rank`: *int*

  - Local rank for `DistributedDataParallel`. Default -1.

- `--batch`: *int*

  - Batch size. Default 256.

- `--count`: *bool*

  - Model parameter counting. Default False.

- `--interval`: *float*

  - Interval length, namely $\Delta l$ in the paper. Default 0.1.

- `--min_score`: *int*

  - Minimum attractiveness score of the dataset. Default 1.

- `--max_score`: *int*

  - Maximum attractiveness score of the dataset. Default 5.

- `--device`: *str*

  - GPU Device ID. Default 0.

## Acknowledgement

- SCUT-FBP5500: https://github.com/HCIILAB/SCUT-FBP5500-Database-Release
- SCUT-FBP: http://www.hcii-lab.net/data/SCUT-FBP/EN/introduce.html

## Reference

[1] Gao, Bin-Bin, et al. "Learning expectation of label distribution for facial age and attractiveness estimation." *arXiv preprint arXiv:2007.01771* (2020).

[2] Xu, Lu, Heng Fan, and Jinhai Xiang. "Hierarchical multi-task network for race, gender and facial attractiveness recognition." *IEEE International conference on image processing (ICIP)*. IEEE, 2019.

