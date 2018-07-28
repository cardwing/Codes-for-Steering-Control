# Codes for autonomous driving tasks

Content:

- Semantic segmentation codes (based on pytorch, author: Li Xiaoxiao, modified: Hou Yuenan)

- Steering angle prediction codes (based on Tensorflow, author: Hou Yuenan, state: to be released)

- Lane detection codes (based on Torch, author: Pan Xinggang, modified: Hou Yuenan, state: to be written)


To do list:

- Clean all codes, make them readable and reproducable

- Put codes of steering angle prediction (tensorflow) and lane detection (torch and pytorch) here

- Attach original experimental results here to facilitate future research


Instructions on running the code in the pytorch-semantic-segmentation-master repo

FCN (mIoU 71.03%)
```{r, engine='bash', count_lines}
python3 main.py VOCAug FCN train val --lr 0.01 --gpus 0 1 2 3 4 5 6 7 --npb
```

PSPNet
```{r, engine='bash', count_lines}
python3 train_pspnet.py VOCAug PSPNet train val --lr 0.01 --gpus 0 1 2 3 4 5 6 7 --npb --test_size 473
```
