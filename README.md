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

## Comma-ai

## BDD100K
    
# Semantic-Segmentation

FCN (mIoU 71.03%)
```{r, engine='bash', count_lines}
python3 main.py VOCAug FCN train val --lr 0.01 --gpus 0 1 2 3 4 5 6 7 --npb
```

PSPNet
```{r, engine='bash', count_lines}
python3 train_pspnet.py VOCAug PSPNet train val --lr 0.01 --gpus 0 1 2 3 4 5 6 7 --npb --test_size 473
```

Note that you can use the code to train models (e.g., PSPNet, SegNet and FCN) in Cityscape.

# Steering-Control

## Test


## Train


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

- [ ] Release codes for steering control

- [ ] Clean all codes, make them readable and reproducable

- [ ] Attach original experimental results
