import argparse

import functools

import cv2
import numpy as np
import imageio
from skimage import io, transform
from glob import glob
import os
import torch
import util

import shlex
import subprocess
import shutil
import struct
import collections

import skimage
import skimage.transform

import pandas as pd


def params_to_filename(params):
    params_to_skip = ['batch_size', 'max_epoch', 'train_test', 'no_preloading', 'logging_root', 'checkpoint']
    fname = ''
    for key, value in vars(params).items():
        if key in params_to_skip:
            continue
        if key == 'checkpoint' or key == 'data_root' or key == 'logging_root' or key == 'val_root':
            if value is not None:
                value = os.path.basename(os.path.normpath(value))

        fname += "%s_%s_"%(key,value)
    return fname


def load_rgb(path, sidelength=None):
    img = imageio.imread(path)[:,:,:3]
    img = skimage.img_as_float32(img)

    if sidelength is not None:
        img = cv2.resize(img, (sidelength, sidelength), interpolation=cv2.INTER_AREA)

    img -= 0.5
    img *= 2.
    img = img.transpose(2,0,1)
    return img


def load_rgb_linear(path):
    img = imageio.imread(path)[:,:,:3]
    img = skimage.img_as_float32(img)
    img -= 0.5
    img *= 2.
    img = img.reshape(-1, 3)
    return img


def load_depth_linear(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    img *= 1e-4

    if len(img.shape) == 3:
        img = img[:,:,:1]

    return img.reshape(-1,1)


def load_depth(path, sidelength=None):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)

    if sidelength is not None:
        img = cv2.resize(img, (sidelength, sidelength), interpolation=cv2.INTER_NEAREST)

    img *= 1e-4

    if len(img.shape) ==3:
        img = img[:,:,:1]
        img = img.transpose(2,0,1)
    else:
        img = img[None,:,:]
    return img


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def square_crop_img(img):
    min_dim = np.amin(img.shape[:2])
    center_coord = np.array(img.shape[:2]) // 2
    img = img[center_coord[0] - min_dim // 2:center_coord[0] + min_dim // 2,
          center_coord[1] - min_dim // 2:center_coord[1] + min_dim // 2]
    return img


def load_img(filepath, target_size=None, anti_aliasing=True, downsampling_order=None, square_crop=False):
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

    if img is None:
        print("Error: Path %s invalid" % filepath)
        return None

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if square_crop:
        img = square_crop_img(img)

    if target_size is not None:
        if downsampling_order == 1:
            img = cv2.resize(img, tuple(target_size), interpolation=cv2.INTER_AREA)
        else:
            img = transform.resize(img, target_size,
                                   order=downsampling_order,
                                   mode='reflect',
                                   clip=False, preserve_range=True,
                                   anti_aliasing=anti_aliasing)
    return img


def train_val_split(object_dir, train_dir, val_dir):
    dirs = [os.path.join(object_dir, x) for x in ['pose', 'rgb', 'depth']]
    data_lists = [sorted(glob(os.path.join(dir, x)))
                                           for dir, x in zip(dirs, ['*.txt', "*.png", "*.png"])]

    cond_mkdir(train_dir)
    cond_mkdir(val_dir)

    [cond_mkdir(os.path.join(train_dir, x)) for x in ['pose', 'rgb', 'depth']]
    [cond_mkdir(os.path.join(val_dir, x)) for x in ['pose', 'rgb', 'depth']]

    shutil.copy(os.path.join(object_dir, 'intrinsics.txt'), os.path.join(val_dir, 'intrinsics.txt'))
    shutil.copy(os.path.join(object_dir, 'intrinsics.txt'), os.path.join(train_dir, 'intrinsics.txt'))

    for data_name, data_ending, data_list in zip(['pose', 'rgb', 'depth'], ['.txt', '.png', '.png'], data_lists):
        val_counter = 0
        train_counter = 0
        for i, item in enumerate(data_list):
            if not i%3:
                shutil.copy(item, os.path.join(train_dir, data_name, "%06d"%train_counter+data_ending))
                train_counter += 1
            else:
                shutil.copy(item, os.path.join(val_dir, data_name, "%06d"%val_counter+data_ending))
                val_counter += 1



def load_pose(filename):
    lines = open(filename).read().splitlines()
    if len(lines)==1:
        pose = np.zeros((4,4),dtype=np.float32)
        for i in range(16):
            pose[i//4, i%4] = lines[0].split(" ")[i]
        return pose.squeeze()
    else:
        lines = [[x[0],x[1],x[2],x[3]] for x in (x.split(" ") for x in lines[:4])]
        return np.asarray(lines).astype(np.float32).squeeze()


def load_params(filename):
    lines = open(filename).read().splitlines()

    params = np.array([float(x) for x in lines[0].split()]).astype(np.float32).squeeze()
    return params


def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs


def remove_margin(img, margin):
    return img[margin:-margin, margin:-margin, :]


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def read_view_direction_rays(direction_file):
    img = cv2.imread(direction_file, cv2.IMREAD_UNCHANGED).astype(np.float32)
    img -= 40000
    img /= 10000
    return img

def get_pose_img(poses_file):
    pose = load_pose(poses_file)
    img = np.tile(pose.reshape(-1)[None,None,:], (512,512,1))
    return img

def process_ray_dirs(pose_dir, target_dir):
    ray_dir = os.path.join(target_dir, 'ray_dirs_high')
    view_dir = os.path.join(target_dir, 'view_dirs_high')

    print(ray_dir)

    for dir in [ray_dir, view_dir]:
        if not os.path.exists(dir):
            os.mkdir(dir)

    all_poses = sorted(glob(os.path.join(pose_dir, '*.txt')))
    full_intrinsic = np.array([[525., 0., 319.5, 0.], [0., 525., 239.5, 0], [0., 0, 1, 0], [0, 0, 0, 1]])

    high_res_intrinsic = np.copy(full_intrinsic)
    high_res_intrinsic[:2, :3] *= 512. / 480.
    high_res_intrinsic[:2, 2] = 512. / 480 * 239.5

    high_res_intrinsic = torch.Tensor(high_res_intrinsic).float()

    ray_min, ray_max, view_min, view_max = 1000., -1., 1000., -1.
    for i, pose_file in enumerate(all_poses):
        print(pose_file)
        pose = load_pose(pose_file)

        view_rays = util.compute_view_directions(high_res_intrinsic,
                                                 torch.from_numpy(pose).squeeze(),
                                                 img_height_width=(512,512),
                                                 voxel_size=1,
                                                 frustrum_depth=1).squeeze().cpu().permute(1,2,0).numpy()
        view_direction = np.tile(pose[:3,2].squeeze()[None,None,:], (512,512,1))
        view_direction /= np.linalg.norm(view_direction, axis=2, keepdims=True)

        # ray_min = min(ray_min, np.amin(view_rays)) # -1.03
        # ray_max = max(ray_max, np.amax(view_rays)) # 1.03
        # view_min = min(view_min, np.amin(view_direction)) # -1.69871
        # view_max = max(view_max, np.amax(view_direction)) # 1.699954

        view_rays *= 10000
        view_rays += 40000

        view_direction *= 10000
        view_direction += 40000

        cv2.imwrite(os.path.join(ray_dir, "%05d.png"%i), view_rays.round().astype(np.uint16))
        cv2.imwrite(os.path.join(view_dir, "%05d.png"%i), view_direction.round().astype(np.uint16))


def shapenet_train_test_split(shapenet_path, synset_id, name, csv_path):
    '''

    :param synset_id: synset ID as a string.
    :param name:
    :param csv_path:
    :return:
    '''
    parsed_csv = pd.read_csv(filepath_or_buffer=csv_path)
    synset_df = parsed_csv[parsed_csv['synsetId']==int(synset_id)]

    print(len(synset_df))

    train = synset_df[synset_df['split']=='train']
    val = synset_df[synset_df['split']=='val']
    test = synset_df[synset_df['split']=='test']
    print(len(train), len(val), len(test))

    train_path, val_path, test_path = [os.path.join(shapenet_path, str(synset_id) + '_' + name + '_' + x)
                                       for x in ['train', 'val', 'test']]
    cond_mkdir(train_path)
    cond_mkdir(val_path)
    cond_mkdir(test_path)

    for split_df, trgt_path in zip([train, val, test], [train_path, val_path, test_path]):
        for row_no, row in split_df.iterrows():
            try:
                shutil.copytree(os.path.join(shapenet_path, str(synset_id), str(row.modelId)),
                                os.path.join(shapenet_path, trgt_path, str(row.modelId)))
            except FileNotFoundError:
                print("%s does not exist"%str(row.modelId))


def load(data_dir, dataset="shepard_metzler_5_parts", mode="train", image_kwargs=None, viewpoint_kwargs=None):
    """Use image_kwargs and viewpoint_kwargs to set e.g. mmap_mode."""

    if image_kwargs is None: image_kwargs = {}
    if viewpoint_kwargs is None: viewpoint_kwargs = {}

    images = np.load(os.path.join(data_dir, "{}_{}_images.npy".format(dataset,mode)), **image_kwargs)
    viewpoints = np.load(os.path.join(data_dir, "{}_{}_viewpoints.npy".format(dataset,mode)), **viewpoint_kwargs)

    return {"data": images, "viewpoints": viewpoints}


def transform_viewpoint(v):
    """Transforms the viewpoint vector into a consistent representation"""

    return np.concatenate([v[:, :3],
                           np.cos(v[:, 3:4]),
                           np.sin(v[:, 3:4]),
                           np.cos(v[:, 4:5]),
                           np.sin(v[:, 4:5])], 1)


def euler2mat(z=0, y=0, x=0):
    Ms = []
    if z:
        cosz = np.cos(z)
        sinz = np.sin(z)
        Ms.append(np.array(
                [[cosz, -sinz, 0],
                 [sinz, cosz, 0],
                 [0, 0, 1]]))
    if y:
        cosy = np.cos(y)
        siny = np.sin(y)
        Ms.append(np.array(
                [[cosy, 0, siny],
                 [0, 1, 0],
                 [-siny, 0, cosy]]))
    if x:
        cosx = np.cos(x)
        sinx = np.sin(x)
        Ms.append(np.array(
                [[1, 0, 0],
                 [0, cosx, -sinx],
                 [0, sinx, cosx]]))
    if Ms:
        return functools.reduce(np.dot, Ms[::-1])
    return np.eye(3)


def look_at(vec_pos, vec_look_at):
    z = vec_look_at - vec_pos
    z = z / np.linalg.norm(z)

    x = np.cross(z, np.array([0., 1., 0.]))
    x = x / np.linalg.norm(x)

    y = np.cross(x, z)
    y = y / np.linalg.norm(y)

    view_mat = np.zeros((3, 3))

    view_mat[:3, 0] = x
    view_mat[:3, 1] = y
    view_mat[:3, 2] = z

    return view_mat


def gqn_viewpoint2pose(viewpoint):
    ''' Transforms the pitch, yaw & xyz coordinates of the gqn paper to a cam2world transform matrix.
    :param viewpoint:
    :return:
    '''
    xyz = viewpoint[:3]

    # pitch = viewpoint[3]
    # yaw = viewpoint[4]

    # assemble rotation matrix
    # Forward vector is simply the normalized xyz coordinates.
    rotation_mat = look_at(xyz, np.zeros((3)))

    # assemble the camera matrix
    pose = np.zeros((4,4))
    pose[:3,3] = xyz
    pose[:3,:3] = rotation_mat
    pose[3,3] = 1.

    return pose


def prepare_gqn(trgt_dir, data_dir, dataset):
    fovy = 45.
    f = 0.5 * 64 / np.tan(fovy * np.pi / 360)

    viewpoints = np.load(os.path.join(data_dir, "{}_{}_viewpoints.npy".format(dataset,'train')))
    viewpoints = viewpoints.reshape(-1, 15, 5)[:5000]

    # images = np.load(os.path.join(data_dir, "{}_{}_images.npy".format(dataset,'train')))
    # images = data['data'].reshape(-1, 15, 64, 64, 3)[:5000]

    for i in range(len(viewpoints)):
        if not i%100:
            print(i)
        instance_dir = os.path.join(trgt_dir, "%06d"%i)
        rgb_dir = os.path.join(instance_dir, 'rgb')
        pose_dir = os.path.join(instance_dir, 'pose')
        intr_dir = os.path.join(instance_dir, 'intrinsics')

        cond_mkdir(instance_dir)
        cond_mkdir(rgb_dir)
        cond_mkdir(pose_dir)
        cond_mkdir(intr_dir)

        with open(os.path.join(instance_dir, 'intrinsics.txt'), 'w') as file:
            file.write(' '.join(map(str, [f, 32, 32, 0.])) + '\n')
            file.write(' '.join(map(str, [0.,0.,0.])) + '\n')
            file.write(str(1.) + '\n')
            file.write(str(64) + ' ' + str(64) + '\n')

        for j in range(len(viewpoints[i])):
            # imageio.imsave(os.path.join(rgb_dir, "%06d.png"%j), images[i][j])
            pose = gqn_viewpoint2pose(viewpoints[i][j])

            with open(os.path.join(pose_dir, "%06d.txt"%j), 'w') as file:
                file.write(' '.join(map(str, pose.reshape(-1).tolist())) + '\n')

            with open(os.path.join(intr_dir, "%06d.txt"%j), 'w') as file:
                file.write(' '.join(map(str, [f, 32, 32, 0.])) + '\n')
                file.write(' '.join(map(str, [0.,0.,0.])) + '\n')
                file.write(str(1.) + '\n')
                file.write(str(64) + ' ' + str(64) + '\n')


if __name__ == '__main__':
    # for object in ["greek", "cube", "vase", "armchair"]:
    #     train_dir = os.path.join('/media/staging/deep_space/deepvoxels_train/'+object)
    #     val_dir = os.path.join('/media/staging/deep_space/deepvoxels_val/'+object)
    #     cond_mkdir(train_dir)
    #     cond_mkdir(val_dir)
    #
    #     train_val_split('/media/staging/deep_space/deepvoxels_old_idcs/%s_gt_train_random'%object,
    #                     train_dir=train_dir,
    #                     val_dir=val_dir)

    shapenet_train_test_split('/media/data/ShapeNetCore.v1',
                              '02958343',
                              name='cars',
                              csv_path='/media/data/ShapeNetCore.v1/all.csv')

    # opt = parser.parse_args()
    # print('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))
    #
    # cond_mkdir(opt.trgt_dir)
    #
    # reconst_dir = os.path.join(opt.trgt_dir, 'reconstruction')
    # pose_dir = os.path.join(opt.trgt_dir, 'pose')
    #
    # cond_mkdir(reconst_dir)
    # cond_mkdir(pose_dir)
    #
    # print("Bundle Adjusting")
    # bundle_adjust(opt.img_dir, reconst_dir, opt.dense)
    # print("Extracting poses")
    # images = read_poses(reconst_dir)
    # print("Writing Poses")
    # write_poses(images, pose_dir)
    # print("Extracting intrinsics")
    # cameras = read_cameras(reconst_dir)
    # print("Writing intrinsics")
    # write_intrinsic(cameras, opt.trgt_dir)

    #object = 'face'
    #process_ray_dirs('C:/Users/vincent/Documents/Data/Processed/cube_gt_test_no_cvpr/pose',
    #                 'C:/Users/vincent/Documents/Data/Processed/cube_gt_test_no_cvpr')
    #process_ray_dirs('C:/Users/vincent/Documents/Data/Processed/%s_gt_train_random/pose'%object,
    #                 'C:/Users/vincent/Documents/Data/Processed/%s_gt_train_random'%object)
    #process_ray_dirs('C:/Users/vincent/Documents/Data/Processed/%s_gt_test/pose'%object,
    #                'C:/Users/vincent/Documents/Data/Processed/%s_gt_test'%object)
