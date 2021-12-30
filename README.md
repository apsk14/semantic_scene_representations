# Semantic Implicit Neural Scene Representations With Semi-Supervised Training

[![Paper](https://img.shields.io/badge/paper-%09arXiv%3A2003.12673-yellow.svg)](https://arxiv.org/abs/2003.12673)
[![Conference](https://img.shields.io/badge/3DV-2020-blue.svg)](http://3dv2020.dgcv.nii.ac.jp/)

This is the official implementation of the 3DV 2020 submission "Semantic Implicit Neural Scene Representations With Semi Supervised Training"

Existing implicit representations for appearance and geometry of 3D scenes&mdash;such as Scene Representations Networks (SRNs)&mdash;can be updated to also perform semantic segmentation with only a few training examples. The resulting semantic scene representations offer a continuous, multimodal representation of a 3D scene which may find use in downstream applications such as robotics.

In this repository we guide the user through the construction of such a representation, by first pretraining an SRN and then updating it via our semi-supervised, few-shot training strategy. The primary focus is on the second step since the [SRNS repository](https://github.com/vsitzmann/scene-representation-networks) offers a comprehensive overview of the first step.

[![video](https://img.youtube.com/vi/iVubC_ymE5w/0.jpg)](https://www.youtube.com/watch?v=iVubC_ymE5w)

## Usage
### Installation
This code was tested with python 3.7.8 and pytorch 1.8.1 I recommend using anaconda for dependency management. 
You can create an environment with name "srns" with all dependencies like so:
```
conda env create -f environment.yml
```

This repository depends on a git submodule, [pytorch-prototyping](https://github.com/vsitzmann/pytorch_prototyping). 
To clone both the main repo and the submodule, use
```
git clone --recurse-submodules https://github.com/vsitzmann/scene-representation-networks.git
```

### High-Level structure
The code is organized as follows:
* dataio.py loads training and testing data.
* data_util.py and util.py contain utility functions.
* train.py contains the training code.
* test.py contains the testing code.
* srns.py contains the core SRNs model.
* hyperlayers.py contains implementations of different hypernetworks.
* custom_layers.py contains implementations of the raymarcher and the DeepVoxels U-Net renderer.
* geometry.py contains utility functions for 3D and projective geometry.
* util.py contains misc utility functions.

### Pre-Trained models
There are pre-trained models for the shapenet car and chair datasets available, including tensorboard event files of the
full training process. 

Please download them [here](https://drive.google.com/open?id=1IdOywOSLuK6WlkO5_h-ykr3ubeY9eDig).

The checkpoint is in the "checkpoints" directory - to load weights from the checkpoint, simply pass the full path to the checkpoint
to the "--checkpoint_path" command-line argument. 

To inspect the progress of how I trained these models, run tensorboard in the "events" subdirectory. 

### Data
The dataset used in the paper was custom rendered from Blender by registering pairs of objects (chairs and tables) from [Partnet v0](https://partnet.cs.stanford.edu/) and [Shapenet v2](https://shapenet.org/).

The dataset along with pretrained models are stored [here](https://drive.google.com/drive/folders/1OkYgeRcIcLOFu1ft5mRODWNQaPJ0ps90?usp=sharing). Information for the dataset can be found at [semantic_scenes_dataset](https://github.com/apsk14/semantic_scenes_dataset)

Alternatively one can simply run setup.sh in the desired location for the dataset to download it. Be wary, the dataset is fairly large (~46GB).


### Training

Obtaining a semantic scene representation requires 4 main steps.

1) Training a vanilla SRN (Training)
Please refer to the original SRNS repository for this step. See training_scripts/vanilla_srn.sh for an example call.

2) Updating SRN for semantic segmentation (Training)
In this step the features of a pretrained SRN are linearly regressed to semantic labels---the goal being to learn the regression coefficents. An example call for this step is found in seg_scripts/linear_update.sh.

3) Learning latent codes from test time observations (Testing)
In this step, a number of views from a test time, unseen object are used to obtain the SRN that is most consistent with the observations. This can be done with as few as a single image/view of a test time object. An example call is found in test_scripts/single_shot.sh

4) Rendering results from the learned semantic SRN
Finally, with an SRN in hand for each test object, this final step produces samples of the semantic SRN in the form of rgb images an point clouds as well as their corresponding semantic segmentation maps and point clouds. An example call is found in result_scripts/single_shot.sh



See `python train.py --help` for all train options. 
Example train call:
```
python train.py --data_root [path to directory with dataset] \
                --val_root [path to directory with train_val dataset] \
                --logging_root [path to directory where tensorboard summaries and checkpoints should be written to] 
```
To monitor progress, the training code writes tensorboard summaries every 100 steps into a "events" subdirectory in the logging_root.

For experiments described in the paper, config-files are available that configure the command-line flags according to
the settings in the paper. You only need to edit the dataset path. Example call:
```
[edit train_configs/cars.yml to point to the correct dataset and logging paths]
python train.py --config_filepath train_configs/cars.yml
```

### Testing
Example test call:
```
python test.py --data_root [path to directory with dataset] ] \
               --logging_root [path to directoy where test output should be written to] \
               --num_instances [number of instances in training set (for instance, 2433 for shapenet cars)] \
               --checkpoint [path to checkpoint]
```
Again, for experiments described in the paper, config-files are available that configure the command-line flags according to
the settings in the paper. Example call:
```
[edit test_configs/cars.yml to point to the correct dataset and logging paths]
python test.py --config_filepath test_configs/cars_training_set_novel_view.yml
```

## Misc
### Citation
If you find our work useful in your research, please cite:
```
@inproceedings{sitzmann2019srns,
	author = {Sitzmann, Vincent 
	          and Zollh{\"o}fer, Michael
	          and Wetzstein, Gordon},
	title = {Scene Representation Networks: Continuous 3D-Structure-Aware Neural Scene Representations},
	booktitle = {Advances in Neural Information Processing Systems},
	year={2019}
}
```

### Submodule "pytorch_prototyping"
The code in the subdirectory "pytorch_prototyping" comes from a library of custom pytorch modules that I use throughout my 
research projects. You can find it [here](https://github.com/vsitzmann/pytorch_prototyping).

### Contact
If you have any questions, please email Vincent Sitzmann at sitzmann@cs.stanford.edu.
