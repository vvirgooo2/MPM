# MPM: A Unified 2D-3D Human Pose Representation via Masked Pose Modeling

Make sure you have the following dependencies installed:

* PyTorch >= 0.4.0
* NumPy
* Matplotlib

## Dataset preparation
## Dataset

Our model is evaluated on [Human3.6M](http://vision.imar.ro/human3.6m) and [MPI-INF-3DHP](https://vcai.mpi-inf.mpg.de/3dhp-dataset/) datasets. 

Dataset setting is same as this repo [P-STMO](https://github.com/paTRICK-swk/P-STMO). You can download the processed .npz file from their repo and put the .npz files in ./dataset folder.


## Evaluating our models
Model checkpoint is not published yet.
## evaluate on Human3.6M (CPN)
```
python trainer.py -f 243 --n_joints 17 --gpu 0,1 --reload 1 --layers 4 -tds 2 --previous_dir x.pth --refine --refine_reload 1 x_refine.pth
```

## evaluate on Human3.6M (GT)
```
python trainer.py -f 243 k gt  --n_joints 17 --gpu 0,1 --reload 1 --layers 4 -tds 2 --previous_dir x.pth --refine --refine_reload 1 --previous_refine_name x_refine.pth
```


## evaluate on MPI_INF_3DHP(GT)
```
python trainer_3dhp.py -f 243 --n_joints 16 --gpu 0,1 --reload 1 --layers 4 -tds 1 --previous_dir x.pth --refine --refine_reload 1 --previous_refine_name x.pth
```

## Pretraining from scratch

### Prepare Poseaug Generator
You should follow the instructions in [poseaug](https://github.com/jfzhang95/PoseAug) and got generator checkpoint for human3.6M. Then put the generator checkpoints in ./Augpart/chk foler. You can put as many as you can get and modify the list in file ./Augpart/gan_preparation.py  

### Prepare AMASS Dataset 
coming soon


### Pretrain a model for 17 joints (only on h36m dataset)
```python
python pretrainer.py --MAE -f 243 --train 1 -k gt --n_joints 17 -b 1024 -tds 2 --layers 4 --dataset h36m --lr 0.0001 -lrd 0.97 -tmr 0.6 -smn 5 --gpu x,y --name task_name 
```

### Pretrain a model for 16 joints (only on h36m dataset with poseaug)
```python
python pretrainer.py --MAE -f 243 --train 1 -k gt --n_joints 16 -b 1024 -tds 2 --dataset h36m --lr 0.0001 -lrd 0.97 --layers 4 -tmr 0.6 -smn 5 --gpu x,y --name task_name 
```

### Pretrain a model for 16 joints (on multiple dataset)
```python
python pretrainer.py --MAE -f 243 --n_joints 16 -b 1024 -k gt -tds 2 --train 1 --dataset h36m,3dhp,amass --layers 3 --lr 0.0001 -lrd 0.97 -tmr 0.6 -smn 5 --gpu x,y --name task_name 
```

## Train HPE Task on h36m from scratch
N_JOINTS x and Layers n hould keep consistent with the pre-trained model.
```python
python trainer.py -f 243 -k gt --train 1 --n_joints x -b 1024 --gpu 0,1 --lr 0.0007 -lrd 0.97  --layers 4 -tds 2 (--MAEreload 1 --previous_dir /path/to/pretrainedcheckpoint)(optional)
```
After training on human3.6M dataset, you can refine the model by: 
```python
python trainer.py -f 243 -k gt --train 1 --n_joints x -b 1024 --gpu 0,1 --lr 0.0001 -lrd 0.97  --layers 4 -tds 2 --reload 1 --previous_dir /path/to/bestcheckpoint --refine
```

## Train 3DHPE on mpi_inf_3dhp from scratch
Finetune 3DHPE Model for MPI_INF_3DHP with 16 joints:
```python
python trainer_3dhp.py -f 243 -k gt --train 1 --n_joints 16 -b 1024 --gpu 0,1 --lr 0.0007 -lrd 0.97 --layers 3 -tds 1 (--MAEreload 1 --previous_dir /path/to/pretrainedcheckpoint)(optional)
```

## Train imcomplete 2D->3D  Model
You can reload pretrained model or train model without reloading checkpoint:
```python
python pretrainer.py -f 27 -b 2048 --model MAE -k gt --train 1 --layers 3 -tds 2 --lr 0.0002 -lrd 0.97 --name maskedliftcam -tmr 0 -smn 6 --gpu 0,1 --dataset h36m --MAE --comp2dlift 1 
(--MAEreload 1 --MAEcp /path/to/model)(optional)  
```


## Train imcomplete 3D -> 3D Model
You can reload pretrained model or train model without reloading checkpoint:
```python 
python pretrainer.py -f 27 -b 2048 --model MAE -k gt --train 1 --layers 3 -tds 2 --lr 0.0002 -lrd 0.97 --name comp3dcam -tmr 0 -smn 3 --gpu 0,1 --dataset h36m --MAE --comp3d 1 
(--MAEreload 1 --MAEcp /path/to/model)(optional)
```


## Acknowledgement
Our code refers to the following repositories.
* [VideoPose3D](https://github.com/facebookresearch/VideoPose3D)
* [StridedTransformer-Pose3D](https://github.com/Vegetebird/StridedTransformer-Pose3D)
* [ST-GCN](https://github.com/vanoracai/Exploiting-Spatial-temporal-Relationships-for-3D-Pose-Estimation-via-Graph-Convolutional-Networks)
* [MAE-pytorch](https://github.com/pengzhiliang/MAE-pytorch)
* [video-to-pose3D](https://github.com/zh-plus/video-to-pose3D)
* [P-STMO](https://github.com/paTRICK-swk/P-STMO)

We thank the authors for releasing their codes.

## Note
This code might have some bugs because we temporarily consolidated scattered code together.

