# Scene Representation Networks

Scene Representation Networks (SRNs) are a continuous, 3D-structure-aware scene representation that encodes both geometry and appearance. 
SRNs represent scenes as continuous functions that map world coordinates to a feature representation of local scene properties. 
By formulating the image formation as a neural, 3D-aware rendering algorithm, SRNs can be trained end-to-end from only 2D observations, 
without access to depth or geometry. SRNs do not discretize space, smoothly parameterizing scene surfaces, and their 
memory complexity does not scale directly with scene resolution. This formulation naturally generalizes across scenes, 
learning powerful geometry and appearance priors in the process.

[![srns_video](https://img.youtube.com/vi/6vMEBWD8O20/0.jpg)](https://youtu.be/6vMEBWD8O20f)

## Usage
### Installation
This code was developed in python 3.7 and pytorch 1.0. I recommend to use anaconda for dependency management. 
You can create an environment with name "deepvoxels" with all dependencies like so:
```
conda env create -f environment.yml
```

### High-Level structure
The code is organized as follows:
* dataio.py loads training and testing data.
* data_util.py and util.py contain utility functions.
* run_srns.py contains the training and testing code as well as setting up the dataset, dataloading, command line arguments etc.
* srns.py contains the core SRNs model.
* custom_layers.py contains implementations of the integration and occlusion submodules.
* geometry.py contains utility functions for 3D and projective geometry.

### Data:
The datasets have been rendered from a set of high-quality 3D scans of Shapenet v2 chairs and car classes.

### Training
* See `python run_srns.py --help` for all train options. 
Example train call:
```
python run_srns.py --train_test train \
                   --data_root [path to directory with dataset] \
                   --logging_root [path to directory where tensorboard summaries and checkpoints should be written to] 
```
To monitor progress, the training code writes tensorboard summaries every 100 steps into a "runs" subdirectory in the logging_root.

### Testing
Example test call:
```
python run_srns.py --train_test test \
                   --data_root [path to directory with dataset] ]
                   --logging_root [path to directoy where test output should be written to] \
                   --checkpoint [path to checkpoint]
```

## Misc
### Citation:  
If you find our work useful in your research, please consider citing:
```
@inproceedings{sitzmann2019srns,
	author = {Sitzmann, Vincent 
	          and Zollh{\"o}fer, Michael
	          and Wetzstein, Gordon},
	title = {Scene Representation Networks: Continuous 3D-Structure-Aware Neural Scene Representations},
	booktitle = {arXiv},
	year={2019}
}
```

### Submodule "pytorch_prototyping"
The code in the subdirectory "pytorch_prototyping" comes from a little library of custom pytorch modules that I use throughout my 
research projects. You can find it [here](https://github.com/vsitzmann/pytorch_prototyping).

### Contact:
If you have any questions, please email Vincent Sitzmann at sitzmann@cs.stanford.edu.
