from custom_layers import *
from pytorch_prototyping.pytorch_prototyping import *

import skimage.measure, skimage.transform
import torch
import torch.nn as nn
import numpy as np

import matplotlib.cm as cm

def weights_init(m):
    if type(m) in [nn.Conv2d, nn.Linear]:
        nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class LinearModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes
        self.linear = nn.ConvTranspose2d(32, num_classes, kernel_size=5, stride=2, padding=2, output_padding=1, bias=True)
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean')

    def get_seg_loss(self, prediction, ground_truth):
        '''Computes loss on predicted image (L_{img} in eq. 6 in paper)

        :param prediction (tuple): Output of forward pass.
        :param ground_truth: Ground-truth (unused).
        :return: image reconstruction loss.
        '''
        pred_segs = prediction['seg']
        pred_segs = pred_segs.permute(0, 2, 1)

        trgt_segs = ground_truth['seg']
        trgt_segs = trgt_segs.permute(0, 2, 1).squeeze().long().cuda()

        loss = self.cross_entropy_loss(pred_segs, trgt_segs)
        return loss

    def forward(self, input, z=None):
        batch_size = input.shape[0]
        self.logs = list()
        novel_views_seg = self.linear(input.cuda())
        novel_views_seg = novel_views_seg.permute(0,2,3,1).reshape(batch_size, 128*128, -1)
        return {'seg':novel_views_seg}

# noinspection PyCallingNonCallable
class TatarchenkoAutoencoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.img_sidelength = 128
        self.nf0 = 64
        self.rend_in_ch = 256
        self.rendering_input_res = 4
        self.num_classes = num_classes

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
        )
        self.encoder.apply(weights_init)

        self.encoder_post = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.LeakyReLU(inplace=True, negative_slope=0.2)
        )
        self.encoder_post.apply(weights_init)

        # Angle processing
        self.pose_prep_net = nn.Sequential(
            nn.Linear(16, 64),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),

            nn.Linear(64, 64),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),

            nn.Linear(64, 64),
            nn.LeakyReLU(inplace=True, negative_slope=0.2)
        )
        self.pose_prep_net.apply(weights_init)

        # joint processing
        self.geom_reasoning_net = nn.Sequential(
            nn.Linear(4096+64, 4096),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),

            nn.Linear(4096, 4096),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),

            nn.Linear(4096, 4096),
            nn.LeakyReLU(inplace=True, negative_slope=0.2)
        )
        self.geom_reasoning_net.apply(weights_init)

        # Decoder
        self.rendering_net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),

            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),

            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),

            nn.ConvTranspose2d(32, 4, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Tanh()
        )
        self.rendering_net.apply(weights_init)

        self.counter = 0
        self.l2_loss = nn.MSELoss(reduction='mean')

        # List of logs
        self.logs = list()
        self.cuda()

        self.colors = np.concatenate([np.array([[1.,1.,1.]]),
                                      cm.rainbow(np.linspace(0, 1, num_classes-1))[:,:3]],
                                     axis=0)

        print("*"*100)
        print(self)
        print("*"*100)
        # Ours has 53,0497,485 parameters
        print('encoder:')
        util.print_network(self.encoder)
        print("Number of parameters:")
        util.print_network(self)
        print("*"*100)

    def get_output_img(self, model_outputs):
        rgb = model_outputs['rgb']
        img = util.lin2img(rgb)
        return img

    def get_IOU_vals(self, pred_segs, trgt_segs, confusion, part_intersect, part_union): # had arg confusion
        pred_segs, seg_idx = torch.max(pred_segs, dim=2)
        pred_labels = seg_idx.cpu().numpy().squeeze()
        real_label = trgt_segs.cpu().numpy().squeeze()

        num_classes = self.num_classes
        cur_shape_iou_tot = 0.0
        cur_shape_iou_cnt = 0
        for cur_class in range(0, num_classes):

            cur_gt_mask = (real_label == cur_class)
            cur_pred_mask = (pred_labels == cur_class)

            has_gt = (np.sum(cur_gt_mask) > 0)
            has_pred = (np.sum(cur_pred_mask) > 0)

            if has_gt or has_pred:
                intersect = np.sum(cur_gt_mask & cur_pred_mask)
                union = np.sum(cur_gt_mask | cur_pred_mask)
                iou = intersect / union

                cur_shape_iou_tot += iou
                cur_shape_iou_cnt += 1

                part_intersect[cur_class - 1] += intersect
                part_union[cur_class - 1] += union

        if cur_shape_iou_cnt == 0:
            return 1
        return cur_shape_iou_tot / cur_shape_iou_cnt


    def get_comparisons(self, model_input, pred, ground_truth=None):
        pred_rgb, pred_seg = pred['rgb'], pred['seg']
        batch_size = pred_rgb.shape[0]

        pred_rgb = util.lin2img(pred_rgb)[:,:3,:,:]
        pred_seg = self.get_output_seg(pred)
        pred_seg = torch.from_numpy(pred_seg)

        if ground_truth is not None:
            trgt_imgs, trgt_segs = model_input['trgt_rgb'], model_input['trgt_seg']
            trgt_imgs = util.lin2img(trgt_imgs)[:,:3,:,:]
            trgt_segs = util.lin2img(trgt_segs)
            trgt_segs = (self.colors[trgt_segs.cpu().numpy()])[:,0,:,:].transpose(0, 3, 1, 2)
            trgt_segs = torch.from_numpy(trgt_segs)

            return torch.cat((pred_rgb.cpu(), trgt_imgs.cpu(), pred_seg.cpu().float(), trgt_segs.cpu().float()), dim=3).numpy()
        else:
            return torch.cat((normals.cpu(), pred_rgb.cpu()), dim=3).numpy()

    def get_output_seg(self, prediction, trgt=False):
        if not trgt:
            pred = prediction['seg']
            pred_segs, pred = torch.max(pred, dim=2)
            pred = pred[:, :, None]
        else:
            pred = prediction
        output_seg = util.lin2img(pred)
        output_seg = (self.colors[output_seg.cpu().numpy().astype(int)])[:,0,:,:].transpose(0, 3, 1, 2)
        return output_seg.astype(np.float32)

    def get_psnr(self, prediction, ground_truth):
        '''Compute PSNR of model image predictions.

        :param prediction: Return value of forward pass.
        :param ground_truth: Ground truth.
        :return: (psnr, ssim): tuple of floats
        '''
        pred_imgs = prediction['rgb']
        trgt_imgs = ground_truth['rgb']

        trgt_imgs = trgt_imgs.cuda()
        batch_size = pred_imgs.shape[0]

        if not isinstance(pred_imgs, np.ndarray):
            pred_imgs = util.lin2img(pred_imgs).detach().cpu().numpy()

        if not isinstance(trgt_imgs, np.ndarray):
            trgt_imgs = util.lin2img(trgt_imgs).detach().cpu().numpy()

        psnrs, ssims = list(), list()
        for i in range(batch_size):
            p = pred_imgs[i].squeeze().transpose(1, 2, 0)
            trgt = trgt_imgs[i].squeeze().transpose(1, 2, 0)

            p = (p / 2.) + 0.5
            p = np.clip(p, a_min=0., a_max=1.)

            trgt = (trgt / 2.) + 0.5

            ssim = skimage.measure.compare_ssim(p, trgt, multichannel=True, data_range=1)
            psnr = skimage.measure.compare_psnr(p, trgt, data_range=1)

            psnrs.append(psnr)
            ssims.append(ssim)

        return psnrs, ssims


    def get_regularization_loss(self, prediction, ground_truth):
        return 0

    def get_image_loss(self, prediction, ground_truth):
        pred_imgs = prediction['rgb']
        trgt_imgs = ground_truth['rgb'].cuda()

        loss = self.l2_loss(pred_imgs, trgt_imgs)
        return loss

    def get_latent_loss(self):
        '''Computes loss on latent code vectors (L_{latent} in eq. 6 in paper)
        :return: Latent loss.
        '''
        return 0.

    def write_eval(self, model_inputs, prediction, ground_truth, path):
        cond_rgb = model_inputs['rgb']
        pred_rgb = prediction['rgb']
        trgt_rgb = ground_truth['rgb']

        gt = util.convert_image(trgt_rgb)
        pred = util.convert_image(pred_rgb)
        cond_rgb = util.convert_image(cond_rgb)

        output = np.concatenate((cond_rgb, pred, gt), axis=1)
        util.write_img(output, path)

    def write_updates(self, writer, model_inputs, ground_truth, predictions, iter, prefix=''):
        cond_rgb = model_inputs['rgb']
        pred_rgb = predictions['rgb']

        sanity_check_gt_rgb = model_inputs['trgt_rgb'].cuda()
        gt_rgb = ground_truth['rgb'].cuda()

        batch_size = pred_rgb.shape[0]

        # Module's own log
        for type, name, content in self.logs:
            if type=='image':
                writer.add_image(prefix + name, util.lin2img(content.detach().cpu().numpy())[:,:3,:,:], iter)
                writer.add_scalar(prefix + name + '_min', content.min(), iter)
                writer.add_scalar(prefix + name + '_max', content.max(), iter)
            elif type=='figure':
                writer.add_figure(prefix + name, content, iter, close=True)
            elif type=='scalar':
                writer.add_scalar(prefix + name, content.detach().cpu().numpy(), iter)

        if not iter % 500:
            output_vs_gt = util.lin2img(torch.cat((pred_rgb, gt_rgb), dim=0))
            writer.add_image(prefix + "Output_vs_gt",
                    torchvision.utils.make_grid(output_vs_gt[:,:3,:,:],
                                                         scale_each=False,
                                                         nrow=int(np.ceil(np.sqrt(batch_size))),
                                                         normalize=True).cpu().detach().numpy(),
                             iter)
            writer.add_image(prefix + "sanity_check_gt_rgb",
                    torchvision.utils.make_grid(util.lin2img(sanity_check_gt_rgb)[:,:3,:,:],
                                                         scale_each=False,
                                                         nrow=int(np.ceil(np.sqrt(batch_size))),
                                                         normalize=True).cpu().detach().numpy(),
                             iter)
            writer.add_image(prefix + "Conditional_rgb",
                    torchvision.utils.make_grid(util.lin2img(cond_rgb)[:,:3,:,:],
                                                         scale_each=False,
                                                         nrow=int(np.ceil(np.sqrt(batch_size))),
                                                         normalize=True).cpu().detach().numpy(),
                             iter)

        writer.add_scalar(prefix + "out_min", pred_rgb.min(), iter)
        writer.add_scalar(prefix + "out_max", pred_rgb.max(), iter)

        writer.add_scalar(prefix + "trgt_min", gt_rgb.min(), iter)
        writer.add_scalar(prefix + "trgt_max", gt_rgb.max(), iter)

        if 'seg' in predictions:
            pred_seg = predictions['seg']
            trgt_segs = model_inputs['trgt_seg'].cuda()
            colors = self.colors
            pred_seg, seg_idx = torch.max(pred_seg, dim=2)
            seg_idx = seg_idx[:,:,None]

            # Segmentation image outputs
            output_vs_gt_seg = torch.cat((seg_idx.int(), trgt_segs.int()), dim=0)
            output_vs_gt_seg = util.lin2img(output_vs_gt_seg).int()
            output_vs_gt_seg = torch.from_numpy(colors[output_vs_gt_seg.cpu().numpy()].squeeze()).permute(0,3,1,2)
            print('DISPSEG', output_vs_gt_seg.shape)
            writer.add_image(prefix + "Output_vs_gt_seg",
                             torchvision.utils.make_grid(output_vs_gt_seg[:,:3,:,:],
                                                         scale_each=False,
                                                         normalize=False).cpu().detach().numpy(),iter)

    def forward(self, input):
        self.logs = list()

        context_rgb = input['rgb'].cuda()
        trgt_pose = input['trgt_pose'].cuda()

        batch_size = context_rgb.shape[0]

        posenet_input = trgt_pose.view(batch_size, -1)
        pose_prep = self.pose_prep_net(posenet_input)

        context_rgb = context_rgb.permute(0, 2, 1).reshape(batch_size, 4, 128, 128)
        self.embed = self.encoder(context_rgb)
        self.embed = self.encoder_post(self.embed.view(batch_size, self.rendering_input_res ** 2 * self.rend_in_ch))

        geom_reasoning_in = torch.cat((self.embed, pose_prep), dim=-1)
        rendering_input = self.geom_reasoning_net(geom_reasoning_in)

        rendering_input = rendering_input.view(batch_size,
                                               self.rend_in_ch,
                                               self.rendering_input_res,
                                               self.rendering_input_res)
        novel_views = self.rendering_net(rendering_input)

        features = self.rendering_net[:-2](rendering_input)
        self.counter += 1

        novel_views = novel_views.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
        return {'rgb':novel_views, 'features':features}
