Codes for ["Learning to Steer by Mimicking Features from Heterogeneous Auxiliary Networks"](https://arxiv.org/abs/1811.02759).

Besides, our project page is now available at [FM-Net](https://cardwing.github.io/projects/FM-Net).

<img src='./demo_video/intro.png' width=880>

### Demo video

- Performance of auxiliary networks on unseen target data:

<img src='https://github.com/cardwing/Codes-for-Steering-Control/blob/master/demo_video/demo_AX.gif' width=640>

- Performance of FM-Net:

<img src='https://github.com/cardwing/Codes-for-Steering-Control/blob/master/demo_video/demo_FM.gif' width=640>

# Content:

* [Installation](#Installation)
* [Datasets](#Datasets)
  * [Udacity](#Udacity)
  * [Comma-ai](#Comma-ai)
  * [BDD100K](#BDD100K)
* [Semantic-Segmentation](#Semantic-Segmentation)
* [Steering-Control](#Steering-Control)
  * [Test](#Test)
  * [Train](#Train)
* [Performance](#Performance)
* [Others](#Others)
  * [Citation](#Citation)
  * [Acknowledgement](#Acknowledgement)
  * [Contact](#Contact)

# Installations
    conda create -n tensorflow_gpu pip python=2.7
    source activate tensorflow_gpu
    pip install --upgrade tensorflow-gpu==1.4
    conda install pytorch torchvision -c pytorch
    
# Datasets

## Udacity

The whole dataset is available at [Udacity](https://github.com/udacity/self-driving-car).

## Comma-ai

The whole dataset is available at [Comma-ai](https://github.com/commaai/research).

## BDD100K

The whole dataset is available at [BDD100K](http://bdd-data.berkeley.edu/).
    
# Semantic-Segmentation

FCN (mIoU 71.03%)
```{r, engine='bash', count_lines}
cd semantic-segmentation
python3 main.py VOCAug FCN train val --lr 0.01 --gpus 0 1 2 3 4 5 6 7 --npb
```

PSPNet
```{r, engine='bash', count_lines}
python3 train_pspnet.py VOCAug PSPNet train val --lr 0.01 --gpus 0 1 2 3 4 5 6 7 --npb --test_size 473
```

Note that you can use the code to train models (e.g., PSPNet, SegNet and FCN) in Cityscape.

# Steering-Control

## Test

```{r, engine='bash', count_lines}
cd steering-control
CUDA_VISIBLE_DEVICES="0" python 3d_resnet_lstm.py
```
Note that you need to read [3d_resnet_lstm.py](./steering-control/3d_resnet_lstm.py) and [options.py](./steering-control/options.py) carefully and modify the path accordingly. Note that current setting is used for Udacity dataset. To run the codes for Comma.ai dataset, please refer to [Comma-ai](https://github.com/commaai/research) and [our paper](https://arxiv.org/abs/1811.02759) to modify several parameters.

## Train

```{r, engine='bash', count_lines}
CUDA_VISIBLE_DEVICES="0" python 3d_resnet_lstm.py --flag train
```

# Performance

1. Udacity testing set:

|Model|MAE|RMSE|
| --- |:---:|:---:|
|3D CNN|2.5598|3.6646|
|3D CNN + LSTM|1.8612|2.7167|
|3D ResNet (ours)|1.9167|2.8532|
|3D ResNet + LSTM (ours)|1.7147|2.4899|
|**FM-Net (ours)**|**1.6236**|**2.3549**|

2. Comma-ai testing set:

|Model|MAE|RMSE|
| --- |:---:|:---:|
|3D CNN|1.7539|2.7316|
|3D CNN + LSTM|1.4716|1.8397|
|3D ResNet (ours)|1.5427|2.4288|
|3D ResNet + LSTM (ours)|0.7989|1.1519|
|**FM-Net (ours)**|**0.7048**|**0.9831**|

3. BDD100K testing set:

|Model|Accuracy|
| --- |:---:|
|FCN + LSTM|82.03%|
|3D CNN + LSTM|82.94%|
|3D ResNet + LSTM (ours)|83.69%|
|**FM-Net (ours)**|**85.03%**|

# Others

## Citation

If you use the codes, please cite the following publications:

``` 
@article{hou2018learning,
  title={Learning to Steer by Mimicking Features from Heterogeneous Auxiliary Networks},
  author={Hou, Yuenan and Ma, Zheng and Liu, Chunxiao and Loy, Chen Change},
  journal={arXiv preprint arXiv:1811.02759},
  year={2018}
}
```

## Acknowledgement
This repo is built upon [Udacity](https://github.com/udacity/self-driving-car).


## Contact
If you have any problems in reproducing the results, just raise an issue in this repo.

## To-Do List:

- [x] Release codes for steering control

- [x] Attach original experimental results

- [x] Clean all codes, make them readable and reproducable

- [] Release codes for BDD100K dataset
