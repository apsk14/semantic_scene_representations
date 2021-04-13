import torch
import torch.nn as nn
import numpy as np
import unittest

def look_at_deepvoxels(translation, point):
    '''Look at function that conforms with DeepVoxels, DeepSpace and DeepClouds coordinate system: Camera
    looks in negative z direction.

    :param translation:
    :param point:
    :return:
    '''
    target_pose = np.zeros((4,4))
    target_pose[:3,3] = translation

    direction = point - translation

    dir_norm = direction / np.linalg.norm(direction)

    tmp = np.array([0.,1.,0.])
    right_vector = np.cross(tmp,-1. * dir_norm) # Camera points in negative z-direction
    right_vector /= np.linalg.norm(right_vector)

    up_vector = np.cross(dir_norm, right_vector)

    target_pose[:3,2] = dir_norm
    target_pose[:3,1] = up_vector
    target_pose[:3,0] = right_vector

    target_pose[3,3] = 1.

    return target_pose


def look_at_cars(translation, point):
    '''Look at function that conforms with DeepVoxels, DeepSpace and DeepClouds coordinate system: Camera
    looks in negative z direction.

    :param translation:
    :param point:
    :return:
    '''
    target_pose = np.zeros((4,4))
    target_pose[:3,3] = translation

    direction = point - translation

    dir_norm = direction / np.linalg.norm(direction)

    tmp = np.array([0.,0.,1.])
    right_vector = np.cross(tmp,-1. * dir_norm) # Camera points in negative z-direction
    right_vector /= np.linalg.norm(right_vector)

    up_vector = np.cross(dir_norm, right_vector)

    target_pose[:3,2] = dir_norm
    target_pose[:3,1] = up_vector
    target_pose[:3,0] = right_vector

    target_pose[3,3] = 1.

    return target_pose


def around(look_at_fn, radius=1, num_samples=200, altitude=45):
    '''

    :param radius:
    :param num_samples:
    :param altitude: Altitude in degree.
    :return:
    '''
    trajectory = []

    z_coord = np.sin(np.deg2rad(altitude)) * radius
    virtual_radius = np.cos(np.deg2rad(altitude)) * radius

    for angle in np.linspace(-np.pi, np.pi, num_samples):
        translation = np.array([virtual_radius*np.sin(angle),
                                virtual_radius*np.cos(angle),
                                z_coord])
        trajectory.append(look_at_fn(translation, np.array([0.,0.,0.])))

    return trajectory


def back_and_forth(look_at_fn, radius=1, num_samples=100, altitude=0):
    '''

    :param radius:
    :param num_samples:
    :param altitude: Altitude in degree.
    :return:
    '''
    trajectory = []

    z_coord = np.sin(np.deg2rad(altitude)) * radius
    virtual_radius = np.cos(np.deg2rad(altitude)) * radius

    angles = np.linspace(-0.2*np.pi, 0.2*np.pi, num_samples).tolist()

    for angle in angles + angles[::-1]:
        translation = np.array([virtual_radius*np.sin(angle),
                                virtual_radius*np.cos(angle)*-1.,
                                z_coord])
        trajectory.append(look_at_fn(translation, np.array([0.,0.,0.])))

    return trajectory



####
# Test cases
class TestTrajectories(unittest.TestCase):
    def test_car_pose(self):
        translation = np.array([-0.2810470163822174, 0.8494009375572205, 0.9431492686271667])
        gt_pose = np.array([[-0.9493807554244995,-0.22789952158927917,0.21619005501270294,-0.2810470163822174],
                            [-0.31412792205810547,0.6887753009796143,-0.6533854007720947,0.8494009375572205],
                            [-1.490116545710407e-07,-0.6882228255271912,-0.7254995107650757,0.9431492686271667],
                            [0.0,0.0,-0.0,1.0]])

        pose = look_at_cars(translation, np.array([0., 0., 0.]))
        print(pose)

        self.assertTrue(np.allclose(pose, gt_pose, atol=1e-6))


if __name__ == '__main__':
    unittest.main()
