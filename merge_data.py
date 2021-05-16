from pytorch_prototyping import pytorch_prototyping
# import configargparse
# import os, time, datetime
#
# import torch
# from torch.utils.tensorboard import SummaryWriter
# import numpy as np
#
# import class_dataio as dataio
# from torch.utils.data import DataLoader
# #from srns_vincent import *
# import util
import pdb



def main():
    data_root = '/home/apsk14/data/final_data/Chair/Chair.train'
    stat_root = '/media/data1/apsk14/srn_seg_data/Chair/ Chair.train'

    instance_dirs = sorted(glob(os.path.join(root_dir, "*/")))
    for i in range(len(self.instance_dirs)):
        self.stat_dirs.append(os.path.join(stat_dir, self.instance_dirs[i].split('/')[-2] + '/', ))
        assert (self.instance_dirs[i].split('/')[-2] == self.stat_dirs[i].split('/')[-2]), "Misaligned!" + \
                                                                                           self.instance_dirs[i].split(
                                                                                               '/')[-2] + 'vs' + \
                                                                                           self.stat_dirs[i].split('/')[
                                                                                               -2]

    pdb.set_trace()

if __name__ == '__main__':
    main()
