import os
import torch
import numpy as np
from glob import glob
import data_util
import util
import pdb
from torch.utils.data import DataLoader
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import cv2


def pick(list, item_idcs):
    if not list:
        return list
    return [list[i] for i in item_idcs]


class SceneInstanceDataset():
    """This creates a dataset class for a single object instance (such as a single car)."""

    def __init__(self,
                 instance_idx,
                 instance_dir,
                 specific_observation_idcs=None,  # For few-shot case: Can pick specific observations only
                 img_sidelength=None,
                 num_images=-1,
                 part_name2id={},
                 part_old2new={},
                 specific_class=0):

        self.instance_idx = instance_idx
        self.img_sidelength = img_sidelength
        self.instance_dir = instance_dir
        self.specific_class= specific_class

        #og_dir

        color_dir = os.path.join(instance_dir, "rgb")
        pose_dir = os.path.join(instance_dir, "pose")
        seg_dir = os.path.join(instance_dir, 'seg')

        if not os.path.isdir(color_dir):
            print("Error! root dir %s is wrong" % instance_dir)
            return

        self.color_paths = sorted(data_util.glob_imgs(color_dir))
        self.seg_paths = sorted(data_util.glob_imgs(seg_dir))
        self.pose_paths = sorted(glob(os.path.join(pose_dir, "*.txt")))

        self.pts_path = os.path.join(instance_dir, 'point_cloud', 'sample-points-all-pts-nor-rgba-10000.txt')
        self.labels_path = os.path.join(instance_dir, 'point_cloud', 'sample-points-label-10000.txt')
        self.mapping_path = os.path.join(instance_dir, 'result.json')
        self.transfer_path = os.path.join(instance_dir, 'result_after_merging.json')

        if specific_observation_idcs is not None:
            self.color_paths = pick(self.color_paths, specific_observation_idcs)
            self.pose_paths = pick(self.pose_paths, specific_observation_idcs)
            self.seg_paths = pick(self.seg_paths, specific_observation_idcs)
        elif num_images != -1:
            idcs = np.linspace(0, stop=len(self.color_paths), num=num_images, endpoint=False, dtype=int)
            self.color_paths = pick(self.color_paths, idcs)
            self.pose_paths = pick(self.pose_paths, idcs)
            self.seg_paths = pick(self.seg_paths, idcs)

        #self.transfer_map = data_util.load_transfer_map(self.transfer_path, part_name2id)
        self.pts, self.rgb_pts = data_util.load_pts(self.pts_path)

        #label_map = data_util.load_label_map(self.mapping_path)
        self.labels = data_util.load_labels(self.labels_path)

        test_rgb = data_util.load_rgb(self.color_paths[0])
        self.img_width, self.img_height = test_rgb.shape[1], test_rgb.shape[2]

        print(instance_dir)
        # #colors = np.concatenate([np.array([[1., 1., 1.]]), cm.rainbow(np.linspace(0, 1, 11 - 1))[:, :3]], axis=0)
        # #print(os.path.join(instance_dir, 'seg'))
        # #pdb.set_trace()
        # #os.system('rm ' + os.path.join(instance_dir, 'seg') + "/*.png")
        # #labs = map(str, self.labels)
        # # if instance_dir == '/media/hugespace/amit/semantic_srn_data/Chair/Chair.train/1006be65e7bc937e9141f9b58470d646/':
        # #     print('yooooooö')
        # #     pdb.set_trace()

        # # pdb.set_trace()
        # #os.system('rm ' + os.path.join(instance_dir, 'point_cloud') + "/sample-points-all-label-10000.txt")
        # #test = np.loadtxt(self.labels_path_new, dtype=int)
        # #assert( (test-self.labels).sum() < 1e-10 )

        # for ind, ins in enumerate(self.seg_paths):
        #     #segs = data_util.transfer_labels(self.seg_paths[ind], self.transfer_map, self.img_sidelength, self.specific_class).squeeze()
        #     #segs = segs.reshape(1, -1).transpose(1, 0)
        #     segs = data_util.load_seg(self.seg_paths[ind])
        #     pdb.set_trace()
        #     #np.save(ins.split('.png')[0]+'.npy', segs)




    def set_img_sidelength(self, new_img_sidelength):
        """For multi-resolution training: Updates the image sidelength with whichimages are loaded."""
        self.img_sidelength = new_img_sidelength

    def __len__(self):
        return len(self.pose_paths)

    def __getitem__(self, idx):
        intrinsics, _, _, _ = util.parse_intrinsics(os.path.join(self.instance_dir, "intrinsics.txt"),
                                                                  trgt_sidelength=self.img_sidelength)
        intrinsics = torch.Tensor(intrinsics).float()

        rgb = data_util.load_rgb(self.color_paths[idx], sidelength=self.img_sidelength)
        rgb = rgb.reshape(4, -1).transpose(1, 0)

        pose = data_util.load_pose(self.pose_paths[idx])

        uv = np.mgrid[0:self.img_sidelength, 0:self.img_sidelength].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).long()
        uv = uv.reshape(2, -1).transpose(1, 0)

        #segs = data_util.transfer_labels(self.seg_paths[idx], self.transfer_map, self.img_sidelength, self.specific_class)
        segs = data_util.load_seg(self.seg_paths[idx], sidelength=self.img_sidelength)

        segs = segs.reshape(1, -1).transpose(1, 0)

        instance_id = self.instance_dir.split('/')[-2]
        sample = {
            "instance_idx": torch.Tensor([self.instance_idx]).squeeze(),
            "rgb": torch.from_numpy(rgb).float(),
            "pose": torch.from_numpy(pose).float(),
            "uv": uv,
            "seg": torch.from_numpy(segs).int(),
            "pts": torch.from_numpy(self.pts),
            "rgb_pts": torch.from_numpy(self.rgb_pts),
            "labels": torch.from_numpy(self.labels), 
            "intrinsics": intrinsics,
            "instance_id": instance_id
        }
        return sample


class SceneClassDataset(torch.utils.data.Dataset):
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""

    def __init__(self,
                 root_dir,
                 obj_name,
                 img_sidelength=None,
                 max_num_instances=-1,
                 max_observations_per_instance=-1,
                 specific_observation_idcs=None,  # For few-shot case: Can pick specific observations only
                 samples_per_instance=1,
                 specific_ins = [],
                 specific_class = 0):


        self.samples_per_instance = samples_per_instance
        self.instance_dirs = sorted(glob(os.path.join(root_dir, "*/")))
        #self.stat_dirs = sorted(glob(os.path.join(stat_dir, "*/")))

        # for i in range(len(self.instance_dirs)):
        #     self.stat_dirs.append(os.path.join(stat_dir,self.instance_dirs[i].split('/')[-2] + '/',))
        #     assert (self.instance_dirs[i].split('/')[-2] == self.stat_dirs[i].split('/')[-2]), "Misaligned!" + self.instance_dirs[i].split('/')[-2] + 'vs' + self.stat_dirs[i].split('/')[-2]

        assert (len(self.instance_dirs) != 0), "No objects in the data directory"

        if max_num_instances != -1:
            self.instance_dirs = self.instance_dirs[:max_num_instances]

        if specific_ins is not None and len(specific_ins) != 0:
            print('Using Specific Instances')
            specific_ins_id = [None] * len(specific_ins)
            specific_dirs = [None] * len(specific_ins)
            for idx in range(len(self.instance_dirs)):
                if self.instance_dirs[idx].split('/')[-2] in specific_ins:
                    idc = specific_ins.index(self.instance_dirs[idx].split('/')[-2])
                    specific_ins_id[idc] = idx
                    specific_dirs[idc] = self.instance_dirs[idx]
            self.instance_dirs = specific_dirs
        else:
            specific_ins_id = range(0,len(self.instance_dirs))


        # seg_level_fn = os.path.join(os.path.dirname(root_dir), obj_name+'-level-1.txt')
        # obj_name = ''.join(e for e in obj_name if e.isalnum())
        seg_level_fn = os.path.join(os.path.dirname(os.path.normpath(root_dir)), ''.join([obj_name,'-level-1.txt']))
        with open(seg_level_fn, 'r') as fin:
            part_name2id = {d.split()[1]: int(d.split()[0]) for d in fin.readlines()}
        seg_mapping_fn = os.path.join(os.path.dirname(os.path.normpath(root_dir)), ''.join([obj_name,'.txt']))
        with open(seg_mapping_fn, 'r') as fin:
            part_old2new = {d.rstrip().split()[0]: d.rstrip().split()[1] for d in fin.readlines()}

        self.all_instances = [SceneInstanceDataset(instance_idx=id,
                                                   instance_dir=dir,
                                                   specific_observation_idcs=specific_observation_idcs,
                                                   img_sidelength=img_sidelength,
                                                   num_images=max_observations_per_instance,
                                                   part_name2id=part_name2id,
                                                   part_old2new=part_old2new,
                                                   specific_class=specific_class)
                              for idx, (dir,id) in enumerate(zip(self.instance_dirs, specific_ins_id))]

        self.num_per_instance_observations = [len(obj) for obj in self.all_instances]
        self.num_instances = len(self.all_instances)

    def set_img_sidelength(self, new_img_sidelength):
        """For multi-resolution training: Updates the image sidelength with whichimages are loaded."""
        for instance in self.all_instances:
            instance.set_img_sidelength(new_img_sidelength)

    def __len__(self):
        return np.sum(self.num_per_instance_observations)

    def get_instance_idx(self, idx):
        """Maps an index into all tuples of all objects to the idx of the tuple relative to the other tuples of that
        object
        """
        obj_idx = 0
        while idx >= 0:
            idx -= self.num_per_instance_observations[obj_idx]
            obj_idx += 1
        return obj_idx - 1, int(idx + self.num_per_instance_observations[obj_idx - 1])

    def collate_fn(self, batch_list):
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            # make them all into a new dict
            ret = {}
            for k in entry[0][0].keys():
                ret[k] = []
            # flatten the list of list
            for b in entry:
                for k in entry[0][0].keys():
                    ret[k].extend([bi[k] for bi in b])
            for k in ret.keys():
                if type(ret[k][0]) == torch.Tensor:
                   ret[k] = torch.stack(ret[k])
            all_parsed.append(ret)

        return tuple(all_parsed)

    def __getitem__(self, idx):
        """Each __getitem__ call yields a list of self.samples_per_instance observations of a single scene (each a dict),
        as well as a list of ground-truths for each observation (also a dict)."""
        obj_idx, rel_idx = self.get_instance_idx(idx)

        observations = []
        observations.append(self.all_instances[obj_idx][rel_idx])

        for i in range(self.samples_per_instance - 1):
            observations.append(self.all_instances[obj_idx][np.random.randint(len(self.all_instances[obj_idx]))])

        ground_truth = [{'rgb':ray_bundle['rgb'], 'seg':ray_bundle['seg']} for ray_bundle in observations]

        return observations, ground_truth


class SceneClassDatasetWithContext(SceneClassDataset):
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""

    def __init__(self,
                 root_dir,
                 stat_dir,
                 obj_name,
                 num_context,
                 img_sidelength=None,
                 max_num_instances=-1,
                 max_observations_per_instance=-1,
                 specific_observation_idcs=None):
        super().__init__(root_dir=root_dir, stat_dir=stat_dir, obj_name=obj_name, img_sidelength=img_sidelength,
                         max_num_instances=max_num_instances, max_observations_per_instance=max_observations_per_instance,
                         specific_observation_idcs=specific_observation_idcs, samples_per_instance=num_context+1)

    def __getitem__(self, idx):
        observations, ground_truth = super().__getitem__(idx)

        new_observations = {'trgt_' + key:value for key, value in observations[0].items()}

        collated_context = self.collate_fn([(observations[1:],ground_truth[1:])])[0]
        new_observations.update(collated_context)

        return [new_observations], [ground_truth[0]]

def main():
    # TODO: goal is to write out a new segmentation folder for each instance.

    # go to data path and read errythang like you normally do but now write out the final seg maps as npy files.

    name_list = ['Chair.test']

    #name_list = ['Chair.test']
    for name in name_list:
        print(name)
        data_path = '/media/hugespace/amit/semantic_srn_data/Chair/' + name

        train_dataset = SceneClassDataset(root_dir=data_path,
                                             obj_name='Chair',
                                             max_num_instances=20,
                                             max_observations_per_instance=-1,
                                             img_sidelength=128,
                                             specific_observation_idcs = None,
                                             samples_per_instance=1)


        train_dataloader = DataLoader(train_dataset,
                                      batch_size=1,
                                      shuffle=False,
                                      drop_last=False,
                                      collate_fn=train_dataset.collate_fn)
    


    train_dataset[1]
    pdb.set_trace()
    # for model_input, ground_truth in train_dataloader:
    #     print(model_input['instance_id'])


if __name__ == "__main__":

    main()


