import argparse
from dataio import *
import matplotlib.pyplot as plt

# params
parser = argparse.ArgumentParser()
parser.add_argument('--data_root', required=True, help='path to file list of h5 train data')
parser.add_argument('--stat_root', required=True, help='path to file list of h5 train data')
opt = parser.parse_args()
print('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))

dataset = SceneClassDataset(root_dir=opt.data_root,
                            preload=False,
                            samples_per_object=1,
                            num_objects=-1,
                            num_images=1,
                            num_samples=-1,
                            img_sidelength=128,
                            mode='val')

while True:
    ray_bundle = dataset[np.random.randint(len(dataset))]

    print(ray_bundle['instance_idx']
    rgb = ray_bundle['rgb'].cpu().numpy().squeeze().reshape(128, 128, 3)

    plt.imshow(rgb)
    plt.show()
