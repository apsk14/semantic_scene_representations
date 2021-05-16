import os
import torch
import numpy as np
from glob import glob
import data_util
import util

import imageio
import cv2
import pdb
import shutil



def main():

    splits = ['Chair.train', 'Chair.val', 'Chair.test', 'Table.train', 'Table.val', 'Table.test']
    base_root_dir = '/home/apsk14/data/final_data/'
    base_stat_dir = '/media/data1/apsk14/srn_seg_data/'
    for sp in splits:
        print(sp)
        root_dir = base_root_dir + sp.split('.')[0] + '/' + sp
        stat_dir = base_stat_dir + sp.split('.')[0] + '/' + sp

        stat_dirs = list()
        instance_dirs = sorted(glob(os.path.join(root_dir, "*/")))
        for i in range(len(instance_dirs)):
            stat_dirs.append(os.path.join(stat_dir, instance_dirs[i].split('/')[-2] + '/', ))
            assert (instance_dirs[i].split('/')[-2] == stat_dirs[i].split('/')[-2]), "Misaligned!" + \
                                                                                               instance_dirs[i].split(
                                                                                                   '/')[-2] + 'vs' + \
                                                                                               stat_dirs[i].split('/')[
                                                                                                   -2]

        for i in range(len(instance_dirs)):
            pn_dir = os.path.join(stat_dirs[i], 'partnet')
            transfer_path = os.path.join(pn_dir, 'point_sample', 'sample-points-all-label-10000.txt')
            util.cond_mkdir(os.path.join(instance_dirs[i], 'point_cloud'))
            desired_path = os.path.join(instance_dirs[i], 'point_cloud','sample-points-all-label-10000.txt')
            print(stat_dirs[i].split('/')[-2])
            print(instance_dirs[i].split('/')[-2] + '\n')
            shutil.copy(transfer_path, desired_path)


if __name__ == '__main__':
    main()
