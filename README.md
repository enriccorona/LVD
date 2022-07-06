

# Learned Vertex Descent:

## SUMMARY:

[[Project]](http://www.iri.upc.edu/people/ecorona/lvd/) [[arXiv]](https://arxiv.org/abs/2205.06254)<!-- TODO: Fitting SMPLicit -->

<img src='https://www.iri.upc.edu/people/ecorona/lvd/lvd_teaser.png' width=800>

## DATA:
To train our model, we require downloading the SMPL 

### SMPL or MANO files:
Download the SMPL file from this link:
https://drive.google.com/file/d/1gwU794SottM4Nk66ig87GPJm7TjQQEwi/view?usp=sharing
and put it under the folder utils/

With MANO, we follow the pytorch implementation from https://github.com/hassony2/manopth

### Training data:
We use the following datasets:
RenderPeople: https://renderpeople.com/
AXYZ: https://secure.axyz-design.com/
Twindom: https://web.twindom.com/

The pre-processing follows a similar approach as in other works and, in particular, should follow the same procedure as in IP-Net. Please go to their repo to follow these steps: https://github.com/bharat-b7/IPNet

### Trained checkpoints:
Get the trained checkpoints from this link:
https://drive.google.com/file/d/19Z0dm1ZkOafkGignvIljyqqbHoHeukfW/view?usp=sharing
and unzip the file on the main folder

## TESTING:

We provide a few examples to run tests with the trained checkpoints, on all the tasks. Results will be saved under results/

### SMPL estimation from images:
By default we fit SMPL after the prediction, to correct any imperfection or get pose and shape SMPL parameters. Feel free to disactivate this option to get much faster predictions.
```
python test_LVD_images.py --model LVD_images_SMPL --name LVD_images_SMPL
```
or 
```
python test_LVD_images.py --model LVD_images_wmask_SMPL --name LVD_images_wmask_SMPL
```

### SMPL estimation from 3D scans:
```
python test_LVD_3Dscans.py --model LVD_voxels_SMPL --name LVD_voxels_SMPL --batch_size 1
```

### MANO estimation from 3D scans:
```
python test_LVD_MANO.py --model LVD_voxels_MANO --name LVD_voxels_MANO --batch_size 1
```






## TRAINING:

We provide code for training but our training used the RenderPeople/Axyz datasets. If you have access to these datasets, follow the instructions from IP-Net (https://github.com/bharat-b7/IPNet) to fit SMPL and SMPL+D on the scans.

### Overfitting training example:

We also provide data of a single user to show how the training works. This 3D scan is freely available on the RenderPeople website, but we here provide pre-processed data and the SMPL fit. For all scans we followed PIFu and rendered images with their code, with different lighting and viewpoint. After downloading the SMPL model, you can train this example with the following instruction

```
python train.py --dataset_mode overfit_demo_image_SMPL --model LVD_images_SMPL --nepochs_no_decay 300 --nepochs_decay 300 --lr_G 0.001 --name model_images_overfitting_one_user --batch_size 4 --display_freq_s 600 --save_latest_freq_s 600
```

On another tab, one can check the progress with tensorboard by running:
```
tensorboard --logdir=.
```


### Training SMPL estimation method from images:
If you want to train LVD on a larger data, we provide an example of our dataloader for the RenderPeople dataset, where we train with the following commands:

Training original model:
```
xvfb-run -a python train.py --dataset_mode images_SMPL --model LVD_images_SMPL --nepochs_no_decay 500 --nepochs_decay 500 --lr_G 0.001 --name model_images_smplprediction_nomask --batch_size 2 --display_freq_s 3600 --save_latest_freq_s 3600
```

The model is slightly more stable if we also add the mask as a fourth channel on input images:
```
python train.py --dataset_mode images_SMPL --model LVD_images_wmask_SMPL --nepochs_no_decay 500 --nepochs_decay 500 --lr_G 0.001 --name model_images_smplprediction_wmask --batch_size 2 --display_freq_s 3600 --save_latest_freq_s 3600
```

### Training SMPL estimation from 3D scans:
We also provide code to train for the task of SMPL estimation from input volumetric representations, following:
```
python train.py --dataset_mode voxels_SMPL --model LVD_voxels_SMPL --nepochs_no_decay 50 --nepochs_decay 50 --lr_G 0.001 --name hand_lvd_v3 --batch_size 4 --display_freq_s 300
```

### Training MANO estimation from 3D scans:
And finally use the following command to run on volumetric representations of hands, which include noisy hands or hands grasping objects:
```
python train.py --dataset_mode voxels_MANO --model LVD_voxels_MANO --nepochs_no_decay 50 --nepochs_decay 50 --lr_G 0.001 --name hand_lvd_v3 --batch_size 4 --display_freq_s 300
```

## Running on a headless server:
If you have any issue when running on a headless server, run any of the previous commands with XVFB:
```
xvfb-run -a python train...
```

## Acknowledgments:

- IP-Net
- PIFuHD (some functions taken from them)
- Template from GANimation:
- SMPL and MANO

## Citation:
```
@article{corona2022learned,
  title={Learned Vertex Descent: A New Direction for 3D Human Model Fitting},
  author={Corona, Enric and Pons-Moll, Gerard and Aleny{\`a}, Guillem and Moreno-Noguer, Francesc},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```
