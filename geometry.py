import numpy as np
import torch
import unittest

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from torch.nn import functional as F
import util


def compute_normal_map(x_img, y_img, z, intrinsics):
    cam_coords = lift(x_img, y_img, z, intrinsics)
    cam_coords = util.lin2img(cam_coords)

    shift_left = cam_coords[:,:,2:,:]
    shift_right = cam_coords[:,:,:-2,:]

    shift_up = cam_coords[:,:,:,2:]
    shift_down = cam_coords[:,:,:,:-2]

    diff_hor = F.normalize(shift_right - shift_left, dim=1)[:,:,:,1:-1]
    diff_ver = F.normalize(shift_up - shift_down, dim=1)[:,:,1:-1,:]

    cross = torch.cross(diff_hor, diff_ver, dim=1)
    return cross


def get_ray_directions_cam(xy, intrinsics):
    '''Translates meshgrid of xy pixel coordinates to normalized directions of rays through these pixels,
    in camera coordinates.
    '''
    batch_size, num_samples, _ = xy.shape

    x_cam = xy[:,:,0].view(batch_size, -1)
    y_cam = xy[:,:,1].view(batch_size, -1)
    z_cam = torch.ones((batch_size, num_samples)).cuda()

    pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics, homogeneous=False) # (batch_size, -1, 4)
    ray_dirs = F.normalize(pixel_points_cam, dim=2)
    return ray_dirs



def reflect_vector_on_vector(vector_to_reflect, reflection_axis):
    refl = F.normalize(vector_to_reflect.cuda())
    ax = F.normalize(reflection_axis.cuda())

    r = 2 * (ax * refl).sum(dim=1, keepdim=True)*ax - refl
    return r


def parse_intrinsics(intrinsics):
    intrinsics = intrinsics.cuda()

    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    return fx, fy, cx, cy


def expand_as(x, y):
    if len(x.shape) == len(y.shape):
        return x

    for i in range(len(y.shape) - len(x.shape)):
        x = x.unsqueeze(-1)

    return x


def lift(x, y, z, intrinsics, homogeneous=False):
    '''

    :param self:
    :param x: Shape (batch_size, num_points)
    :param y:
    :param z:
    :param intrinsics:
    :return:
    '''
    fx, fy, cx, cy = parse_intrinsics(intrinsics)

    x_lift = (x - expand_as(cx, x)) / expand_as(fx, x) * z
    y_lift = (y - expand_as(cy, y)) / expand_as(fy, y) * z

    if homogeneous:
        return torch.stack((x_lift, y_lift, z, torch.ones_like(z).cuda()), dim=-1)
    else:
        return torch.stack((x_lift, y_lift, z), dim=-1)


def project(x, y, z, intrinsics):
    '''

    :param self:
    :param x: Shape (batch_size, num_points)
    :param y:
    :param z:
    :param intrinsics:
    :return:
    '''
    fx, fy, cx, cy = parse_intrinsics(intrinsics)

    x_proj = expand_as(fx, x) * x / z + expand_as(cx, x)
    y_proj = expand_as(fy, y) * y / z + expand_as(cy, y)

    return torch.stack((x_proj, y_proj, z), dim=-1)


def world_from_xy_depth(xy, depth, cam2world, intrinsics):
    '''Translates meshgrid of xy pixel coordinates plus depth to  world coordinates.
    '''
    batch_size, _, _ = cam2world.shape

    x_cam = xy[:,:,0].view(batch_size, -1)
    y_cam = xy[:,:,1].view(batch_size, -1)
    z_cam = depth.view(batch_size, -1)

    pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics, homogeneous=True) # (batch_size, -1, 4)

    # permute for batch matrix product
    pixel_points_cam = pixel_points_cam.permute(0,2,1)

    world_coords = torch.bmm(cam2world, pixel_points_cam).permute(0,2,1)[:,:,:3] # (batch_size, -1, 3)

    return world_coords


def project_point_on_line(projection_point, line_direction, point_on_line, dim):
    '''Projects a batch of points on a batch of lines as defined by their direction and a point on each line. '''
    assert torch.allclose((line_direction**2).sum(dim=dim, keepdim=True).cuda(), torch.Tensor([1]).cuda())
    return point_on_line + ((projection_point - point_on_line) * line_direction).sum(dim=dim, keepdim=True) * line_direction


def points_in_canonical_halfspace(ray_dirs):
    # Make sure that all ray directions always point into the same halfspace
    x_less_0 = ray_dirs[:,:,0:1] < 0
    x_equal_0 = ray_dirs[:,:,0:1] == 0
    y_less_0 = ray_dirs[:,:,1:2] < 1
    y_equal_0 = ray_dirs[:,:,1:2] == 0
    z_less_0 = ray_dirs[:,:,2:3] < 0

    mask = (x_less_0 | (x_equal_0 & y_less_0) | (x_equal_0 & y_equal_0 & z_less_0)).float()
    return mask


def canonical_ray_dir_flip(ray_dirs):
    flip_mask = points_in_canonical_halfspace(ray_dirs)

    flip_mask *= 2.
    flip_mask -= 1.
    flip_mask *= -1
    return ray_dirs * flip_mask


def get_canonical_coordinate_on_line(point, reference_points):
    batch_size, num_samples, _ = reference_points.shape

    vec = point - reference_points

    signs = points_in_canonical_halfspace(vec) # returns 1 for all those that don't point in correct halfspace

    signs *= 2.
    signs -= 1.
    signs *= -1.

    vec_lens = vec.norm(dim=2, keepdim=True)
    canonical_coords = vec_lens * signs

    return canonical_coords

def roman_map(ray_dirs):
    x = ray_dirs[:,:,0] * ray_dirs[:,:,1] # x*y
    y = ray_dirs[:,:,0] * ray_dirs[:,:,2] # x*z
    z = ray_dirs[:,:,1]**2 - ray_dirs[:,:,2]**2 # y**2 - z**2
    w = 2*ray_dirs[:,:,1] * ray_dirs[:,:,2] # 2*y*z

    return torch.stack((x,y,z,w), dim=2)


def get_ray_directions(xy, cam2world, intrinsics):
    '''Translates meshgrid of xy pixel coordinates to normalized directions of rays through these pixels.
    '''
    batch_size, num_samples, _ = xy.shape

    z_cam = torch.ones((batch_size, num_samples)).cuda()
    pixel_points = world_from_xy_depth(xy, z_cam, intrinsics=intrinsics, cam2world=cam2world) # (batch, num_samples, 3)

    cam_pos = cam2world[:,:3,3]
    ray_dirs = pixel_points - cam_pos[:,None,:] # (batch, num_samples, 3)
    ray_dirs = F.normalize(ray_dirs, dim=2)
    return ray_dirs


def depth_from_world(world_coords, cam2world):
    batch_size, num_samples, _ = world_coords.shape

    points_hom = torch.cat((world_coords, torch.ones((batch_size, num_samples, 1)).cuda()), dim=2) # (batch, num_samples, 4)

    # permute for bmm
    points_hom = points_hom.permute(0,2,1)

    points_cam = torch.inverse(cam2world).bmm(points_hom) # (batch, 4, num_samples)
    depth = points_cam[:,2,:][:,:,None] # (batch, num_samples, 1)
    return depth


# noinspection PyUnresolvedReferences
class ProjectionHelper():
    def __init__(self,
                 frustrum_dims,
                 subpixel_noise=False):
        self.frustrum_dims = frustrum_dims
        self.subpixel_noise = subpixel_noise

        self.pos_y, self.pos_x = torch.meshgrid([
            torch.arange(self.frustrum_dims[0]).float(),
            torch.arange(self.frustrum_dims[1]).float()
        ])

        self.pos_y = self.pos_y.cuda()
        self.pos_x = self.pos_x.cuda()

        self.xy = torch.cat((self.pos_x.contiguous().view(1, 1, -1),
                             self.pos_y.contiguous().view(1, 1, -1)),
                            dim=1).cuda()

    def world_from_xy_depth(self, cam2world, depth, intrinsics):
        batch_size, _, h, w = depth.shape
        cam2world = cam2world.cuda().float()
        intrinsics = intrinsics.cuda().float()

        x = self.xy[:, 0, :].repeat(batch_size, 1)
        y = self.xy[:, 1, :].repeat(batch_size, 1)
        z = depth.view(batch_size, -1)

        if self.subpixel_noise:
            x += torch.zeros_like(x).uniform_(-(h/512)/2, (h/512)/2)
            y += torch.zeros_like(y).uniform_(-(h/512)/2, (h/512)/2)

        return world_from_xy_depth(x, y, z, cam2world, intrinsics)

    def get_ray_direction(self, intrinsics, cam2world):
        batch_size, _, _ = cam2world.shape
        cam2world = cam2world.cuda().float()
        intrinsics = intrinsics.cuda().float()

        x = self.xy[:, 0, :].repeat(batch_size, 1)
        y = self.xy[:, 1, :].repeat(batch_size, 1)
        z = (torch.ones_like(x)).cuda()

        xyz = lift(x, y, z, intrinsics, homogeneous=False)
        xyzw = torch.cat((xyz, torch.zeros(batch_size, 1, self.frustrum_dims[0]*self.frustrum_dims[1]).cuda()), dim=1)

        world_coords = torch.bmm(cam2world, xyzw.view(batch_size, 4, -1))[:, :3, :]
        ray_directions = F.normalize(ray_directions, dim=1)
        return ray_directions


    def get_ray_points(self, cam2world, intrinsics):
        batch_size, _, _ = cam2world.shape
        intrinsics = intrinsics.cuda().float()

        cam_translation = cam2world[:, :3, 3]

        pos_y, pos_x = torch.meshgrid([
            torch.arange(self.frustrum_dims[0]),
            torch.arange(self.frustrum_dims[1])
        ])

        pos_y, pos_x = pos_y.contiguous().view(1, -1).cuda().float(), pos_x.contiguous().view(1, -1).cuda().float()
        xy = torch.cat((pos_x, pos_y), dim=0)[None, :, :].repeat(batch_size, 1, 1)
        z_values = (torch.ones_like(pos_y) * 1.8)[None, :, :].repeat(batch_size, 1, 1)
        xyzw = torch.cat((xy, z_values, torch.ones_like(z_values)), dim=1)

        xyzw[:, 0, :] = (xyzw[:, 0, :] - intrinsics[:, 0, 2][:, None]) / intrinsics[:, 0, 0][:, None]
        xyzw[:, 1, :] = (xyzw[:, 1, :] - intrinsics[:, 1, 2][:, None]) / intrinsics[:, 1, 1][:, None]
        xyzw[:, :2, :] *= xyzw[:, 2, :][:, None, :]

        world_coords = torch.bmm(cam2world, xyzw)

        return torch.stack((world_coords[:, :3, :],
                            cam_translation[:, :, None].repeat(1, 1, self.frustrum_dims[0] * self.frustrum_dims[1])),
                           dim=1)

    def get_frustrum_points(self, cam2world, intrinsics):
        intrinsics = intrinsics.float().cuda()

        batch_size, _, _ = cam2world.shape

        xyzw = self.xyzw[None, :, :].repeat(batch_size, 1, 1)

        xyzw[:, 0, :] = (xyzw[:, 0, :] - intrinsics[:, 0, 2][:, None]) / intrinsics[:, 0, 0][:, None]
        xyzw[:, 1, :] = (xyzw[:, 1, :] - intrinsics[:, 1, 2][:, None]) / intrinsics[:, 1, 1][:, None]
        xyzw[:, :2, :] *= xyzw[:, 2, :][:, None, :]

        world_coords = torch.bmm(cam2world, xyzw)

        return world_coords[:, :3, :].contiguous()

    def lift_keypoints_raw(self, cam2world, feature_map, depth_map, points_xy, intrinsics):
        batch_size, num_feats, _, _ = feature_map.shape
        num_points, _ = points_xy.shape

        depth = depth_map.squeeze()
        feature_map = feature_map.squeeze()

        # Get x,y coordinates
        x_org = points_xy[:, 0]
        y_org = points_xy[:, 1]

        x_indices = y_org.squeeze()
        y_indices = x_org.squeeze()

        x0 = x_indices.floor().long().cuda()
        y0 = y_indices.floor().long().cuda()

        x0 = torch.clamp(x0, 0, self.frustrum_dims[0] - 1)
        y0 = torch.clamp(y0, 0, self.frustrum_dims[1] - 1)

        x1 = (x0 + 1).long()
        y1 = (y0 + 1).long()

        x1 = torch.clamp(x1, 0, self.frustrum_dims[0] - 1)
        y1 = torch.clamp(y1, 0, self.frustrum_dims[1] - 1)

        x = x_indices - x0.float()
        y = y_indices - y0.float()

        z = (1 - x) * (1 - y) * depth[x0, y0] + x * (1 - y) * depth[x1, y0] + (1 - x) * y * depth[x0, y1] + x * y * \
            depth[x1, y1]
        feats = (1 - x) * (1 - y) * feature_map[:, x0, y0] + x * (1 - y) * feature_map[:, x1, y0] + \
                (1 - x) * y * feature_map[:, x0, y1] + x * y * feature_map[:, x1, y1]

        xyzw = torch.stack([x_org,
                            y_org,
                            z,
                            torch.ones_like(x_org).cuda()], 0)  # batch_size, 4, num_points

        # Unproject points
        p_proj = xyzw.clone()
        p_proj[0, :] = (xyzw[0, :] - self.intrinsics[0][2]) * xyzw[2, :] / self.intrinsics[0][0]
        p_proj[1, :] = (xyzw[1, :] - self.intrinsics[1][2]) * xyzw[2, :] / self.intrinsics[1][1]

        # Transform to world coordinates
        p_proj = torch.mm(cam2world, p_proj)
        p_proj = p_proj.permute(1, 0)

        return p_proj[:, :3], feats.transpose(1, 0), xyzw[:3, :].transpose(1, 0)

    def lift_keypoints(self, cam2world, feature_map, depth, attention_maps):
        batch_size, num_feats, _, _ = feature_map.shape
        _, num_points, _, _ = attention_maps.shape

        attention_maps = attention_maps.squeeze()
        depth = depth.squeeze()
        feature_map = feature_map.squeeze()

        # Get x,y coordinates
        expected_x = torch.sum(self.pos_y.view(1, -1) * attention_maps.view(num_points, -1), dim=-1)
        expected_y = torch.sum(self.pos_x.view(1, -1) * attention_maps.view(num_points, -1), dim=-1)
        # expected_z = torch.sum(depth * attention_maps, dim=(2, 3))
        # expected_feats = torch.sum(feature_map[:,None,:,:,:] * attention_maps[:,:,None,:,:], dim=(3,4)) # batch_size, num_points, num_feats

        x_indices = expected_y.squeeze()
        y_indices = expected_x.squeeze()

        x0 = x_indices.floor().long().cuda()
        y0 = y_indices.floor().long().cuda()

        x1 = (x0 + 1).long()
        y1 = (y0 + 1).long()

        x1 = torch.clamp(x1, 0, self.frustrum_dims[0] - 1)
        y1 = torch.clamp(y1, 0, self.frustrum_dims[1] - 1)

        x = x_indices - x0.float()
        y = y_indices - y0.float()

        expected_z = (1 - x) * (1 - y) * depth[x0, y0] + x * (1 - y) * depth[x1, y0] + (1 - x) * y * depth[
            x0, y1] + x * y * depth[x1, y1]
        expected_feats = (1 - x) * (1 - y) * feature_map[:, x0, y0] + x * (1 - y) * feature_map[:, x1, y0] + \
                         (1 - x) * y * feature_map[:, x0, y1] + x * y * feature_map[:, x1, y1]

        xyzw = torch.stack([expected_x,
                            expected_y,
                            expected_z,
                            torch.ones_like(expected_x).cuda()], 0)  # batch_size, 4, num_points

        # Unproject points
        p_proj = xyzw.clone()
        p_proj[0, :] = (xyzw[0, :] - self.intrinsics[0][2]) * xyzw[2, :] / self.intrinsics[0][0]
        p_proj[1, :] = (xyzw[1, :] - self.intrinsics[1][2]) * xyzw[2, :] / self.intrinsics[1][1]

        # Transform to world coordinates
        p_proj = torch.mm(cam2world, p_proj)
        p_proj = p_proj.permute(1, 0)

        return p_proj[:, :3], expected_feats.transpose(1, 0), xyzw[:3, :].transpose(1, 0)


def interpolate_lifting(image, lin_ind_3d, query_points, grid_dims):
    batch, num_feats, height, width = image.shape

    image = image.cuda()
    lin_ind_3d = lin_ind_3d.cuda()
    query_points = query_points.cuda()

    x_indices = query_points[1, :]
    y_indices = query_points[0, :]

    x0 = x_indices.floor().long().cuda()
    y0 = y_indices.floor().long().cuda()

    x1 = (x0 + 1).long()
    y1 = (y0 + 1).long()

    x1 = torch.clamp(x1, 0, width - 1)
    y1 = torch.clamp(y1, 0, height - 1)

    x = x_indices - x0.float()
    y = y_indices - y0.float()

    output = torch.zeros(1, num_feats, grid_dims[0] * grid_dims[1] * grid_dims[2]).cuda()
    output[:, :, lin_ind_3d] += image[:, :, x0, y0] * (1 - x) * (1 - y)
    output[:, :, lin_ind_3d] += image[:, :, x1, y0] * x * (1 - y)
    output[:, :, lin_ind_3d] += image[:, :, x0, y1] * (1 - x) * y
    output[:, :, lin_ind_3d] += image[:, :, x1, y1] * x * y

    output = output.view(batch, num_feats, grid_dims[0], grid_dims[1], grid_dims[2])  # Width first

    return output


def interpolate_trilinear(grid, lin_ind_frustrum, voxel_coords, img_shape, frustrum_depth):
    batch, num_feats, height, width, depth = grid.shape

    lin_ind_frustrum = lin_ind_frustrum.long()

    x_indices = voxel_coords[2, :]
    y_indices = voxel_coords[1, :]
    z_indices = voxel_coords[0, :]

    x0 = x_indices.floor().long()
    y0 = y_indices.floor().long()
    z0 = z_indices.floor().long()

    x0 = torch.clamp(x0, 0, width - 1)
    y0 = torch.clamp(y0, 0, height - 1)
    z0 = torch.clamp(z0, 0, depth - 1)

    x1 = (x0 + 1).long()
    y1 = (y0 + 1).long()
    z1 = (z0 + 1).long()

    x1 = torch.clamp(x1, 0, width - 1)
    y1 = torch.clamp(y1, 0, height - 1)
    z1 = torch.clamp(z1, 0, depth - 1)

    x = x_indices - x0.float()
    y = y_indices - y0.float()
    z = z_indices - z0.float()

    # output = torch.zeros(batch, num_feats, img_shape[0]*img_shape[1]*depth).cuda()
    output = torch.zeros(batch, num_feats, img_shape[0] * img_shape[1] * frustrum_depth).cuda()
    output[:, :, lin_ind_frustrum] += grid[:, :, x0, y0, z0] * (1 - x) * (1 - y) * (1 - z)
    output[:, :, lin_ind_frustrum] += grid[:, :, x1, y0, z0] * x * (1 - y) * (1 - z)
    output[:, :, lin_ind_frustrum] += grid[:, :, x0, y1, z0] * (1 - x) * y * (1 - z)
    output[:, :, lin_ind_frustrum] += grid[:, :, x0, y0, z1] * (1 - x) * (1 - y) * z
    output[:, :, lin_ind_frustrum] += grid[:, :, x1, y0, z1] * x * (1 - y) * z
    output[:, :, lin_ind_frustrum] += grid[:, :, x0, y1, z1] * (1 - x) * y * z
    output[:, :, lin_ind_frustrum] += grid[:, :, x1, y1, z0] * x * y * (1 - z)
    output[:, :, lin_ind_frustrum] += grid[:, :, x1, y1, z1] * x * y * zgg

    output = output.contiguous().view(batch, num_feats, frustrum_depth, img_shape[0], img_shape[1])

    return output


####
# Test cases
class TestStringMethods(unittest.TestCase):
    def test_lift_projection(self):
        x = torch.zeros(1, 5).uniform_(0, 64).cuda().double()
        y = torch.zeros(1, 5).uniform_(0, 64).cuda().double()
        z = torch.zeros(1, 5).uniform_(0, 64).cuda().double()
        xyz = torch.stack((x, y, z), dim=-1)

        intrinsics = torch.Tensor([[70., 0., 31.95, 0.],
                                   [0., 70., 31.93333333, 0.],
                                   [0., 0., 1., 0.],
                                   [0., 0., 0., 1.]]).unsqueeze(0).cuda().double()

        lifted = lift(x, y, z, intrinsics)
        projected = project(lifted[:, :, 0], lifted[:, :, 1], lifted[:, :, 2], intrinsics)

        self.assertTrue(torch.allclose(xyz, projected))


    def test_point_line_projection_trivial(self):
        rand_direction = F.normalize(torch.randn(1,1,3), dim=2)
        rand_point = torch.randn(1,1,3)

        rand_distance = torch.randn(10,1,1)
        rand_points_on_line = rand_point + rand_distance * rand_direction

        reference_points = rand_point.repeat(10,1,1)
        projected_points = project_point_on_line(rand_points_on_line, rand_direction, reference_points, dim=2)

        self.assertTrue(torch.allclose(projected_points, rand_points_on_line))


    def test_point_line_projection(self):
        # First, construct points on the line
        rand_direction = F.normalize(torch.randn(1,1,3), dim=2).cuda()
        rand_point = torch.randn(1,1,3).cuda()

        rand_distance = torch.randn(10,1,1).cuda()
        rand_points_on_line = rand_point + rand_distance * rand_direction

        # Now, construct a vector orthogonal to the line direction
        unit_vec = torch.Tensor([1,0,0]).cuda().unsqueeze(0).unsqueeze(0)
        orth_vector = unit_vec.cross(rand_direction, dim=2)

        # Now, add a random orthogonal offset to each of the random points
        rand_orth_dist = torch.randn(10,1,1).cuda()
        rand_points = rand_points_on_line + rand_orth_dist * orth_vector

        reference_points = rand_point.repeat(10,1,1)
        projected_points = project_point_on_line(rand_points.cuda(),
                                                 rand_direction.cuda(),
                                                 reference_points.cuda(), dim=2)

        self.assertTrue(torch.allclose(projected_points, rand_points_on_line))


    def test_ray_directions(self):
        intrinsics = torch.Tensor([[70., 0., 31.95, 0.],
                                   [0., 70., 31.93333333, 0.],
                                   [0., 0., 1., 0.],
                                   [0., 0., 0., 1.]]).unsqueeze(0).cuda().float()

        pose = torch.Tensor([[-0.953334,-0.056063,0.296668,-0.741669],
                             [-3.72529e-009,0.982608,0.185689,-0.464223],
                             [0.301918,-0.177024,0.936754,-2.34188],
                             [-0,-0,-0,1]]).unsqueeze(0).cuda().float()

        xy = np.mgrid[0:64, 0:64].astype(np.int32)
        xy = torch.from_numpy(np.flip(xy, axis=0).copy()).float().cuda()

        ray_dirs = get_ray_directions(xy, cam2world=pose, intrinsics=intrinsics)
        ray_dir_lengths = torch.sqrt((ray_dirs**2).sum(dim=-1,keepdim=True))
        self.assertTrue(np.allclose(1., ray_dir_lengths.cpu()))


    def test_canonical_line_coordinate(self):
        rand_direction = F.normalize(torch.randn(1,1,3), dim=2)
        rand_point = torch.randn(1,1,3)

        rand_distance = torch.randn(1,1,1)
        rand_point_on_line = rand_point + rand_distance * rand_direction
        rand_neg_point_on_line = rand_point - rand_distance * rand_direction

        rand_points = torch.cat((rand_point_on_line, rand_neg_point_on_line), dim=0)
        reference_points = torch.cat((rand_point, rand_point), dim=0)

        canonical_coords = get_canonical_coordinate_on_line(rand_points, reference_points).cpu().squeeze()

        self.assertTrue(np.allclose(canonical_coords[0], -1*canonical_coords[1]))


    def test_world_from_xy_depth(self):
        intrinsics = torch.Tensor([[70., 0., 31.95, 0.],
                                   [0., 70., 31.93333333, 0.],
                                   [0., 0., 1., 0.],
                                   [0., 0., 0., 1.]]).unsqueeze(0).cuda().float()

        cam2world = torch.Tensor([[-0.953334,-0.056063,0.296668,-0.741669],
                                  [-3.72529e-009,0.982608,0.185689,-0.464223],
                                  [0.301918,-0.177024,0.936754,-2.34188],
                                  [-0,-0,-0,1]]).unsqueeze(0).cuda().float()

        xy = np.mgrid[0:64, 0:64].astype(np.int32)
        xy = torch.from_numpy(np.flip(xy, axis=0).copy()).float().cuda()
        batch_size, num_samples, _ = xy.shape

        z_cam = torch.ones((batch_size, num_samples)).cuda()
        pixel_points = world_from_xy_depth(xy, z_cam, intrinsics=intrinsics, cam2world=cam2world) # (batch, num_samples, 3)

    def test_roman_surface(self):
        # Get points on a circle in R^3
        t = np.linspace(0, np.pi, num=100)

        axis_a = [np.sqrt(2), 0, np.sqrt(2)]
        axis_b = [0, 1., 0]

        x = np.cos(t) * axis_a[0] + np.sin(t) * axis_b[0]
        y = np.cos(t) * axis_a[1] + np.sin(t) * axis_b[1]
        z = np.cos(t) * axis_a[2] + np.sin(t) * axis_b[2]

        # Reshape as valid input for roman_map
        xyz = torch.from_numpy(np.stack((x,y,z), axis=1)[None,:,:]).float().cuda()
        rom_embedding = roman_map(xyz)

        x_rom, y_rom, z_rom, w_rom = np.split(rom_embedding.cpu().numpy().squeeze(), 4, axis=1)

        fig = plt.figure()
        ax = fig.add_subplot(1, 3, 1, projection='3d')
        ax.plot(x, y, z, label='parametric curve')

        ax = fig.add_subplot(1, 3, 2)
        ax.plot(x)
        ax.plot(y)
        ax.plot(z)

        ax = fig.add_subplot(1, 3, 3)
        ax.plot(x_rom)
        ax.plot(y_rom)
        ax.plot(z_rom)
        ax.plot(w_rom)

        plt.show()


if __name__ == '__main__':
    import numpy as np

    unittest.main()
