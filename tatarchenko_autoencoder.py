from custom_layers import *
from pytorch_prototyping.pytorch_prototyping import *

import skimage.measure, skimage.transform
import torch
import torch.nn as nn
import numpy as np

def weights_init(m):
    if type(m) in [nn.Conv2d, nn.Linear]:
        nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

# noinspection PyCallingNonCallable
class TatarchenkoAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_sidelength = 128
        self.nf0 = 64
        self.rend_in_ch = 256
        self.rendering_input_res = 4

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

        print("*"*100)
        print(self)
        print("*"*100)
        # Ours has 53,0497,485 parameters
        print('encoder:')
        util.print_network(self.encoder)
        print("Number of parameters:")
        util.print_network(self)
        print("*"*100)


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

    def write_updates(self, writer, model_inputs, predictions, iter, prefix=''):
        cond_rgb = model_inputs['rgb']
        pred_rgb = predictions['rgb']
        gt_rgb = model_inputs['trgt_rgb'].cuda()

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
            writer.add_image(prefix + "Conditional_rgb",
                             torchvision.utils.make_grid(util.lin2img(cond_rgb[:,0])[:,:3,:,:],
                                                         scale_each=False,
                                                         nrow=int(np.ceil(np.sqrt(batch_size))),
                                                         normalize=True).cpu().detach().numpy(),
                             iter)

        writer.add_scalar(prefix + "out_min", pred_rgb.min(), iter)
        writer.add_scalar(prefix + "out_max", pred_rgb.max(), iter)

        writer.add_scalar(prefix + "trgt_min", gt_rgb.min(), iter)
        writer.add_scalar(prefix + "trgt_max", gt_rgb.max(), iter)


    def forward(self, input):
        self.logs = list()

        context_rgb = input['rgb'].cuda()
        trgt_pose = input['trgt_pose'].cuda()

        batch_size = context_rgb.shape[0]

        posenet_input = trgt_pose.view(batch_size, -1)
        pose_prep = self.pose_prep_net(posenet_input)

        context_rgb = context_rgb.permute(0,1,3,2).reshape(batch_size, 4, 128, 128)
        self.embed = self.encoder(context_rgb)
        self.embed = self.encoder_post(self.embed.view(batch_size, self.rendering_input_res**2 * self.rend_in_ch))

        geom_reasoning_in = torch.cat((self.embed, pose_prep), dim=-1)
        rendering_input = self.geom_reasoning_net(geom_reasoning_in)

        rendering_input = rendering_input.view(batch_size,
                                               self.rend_in_ch,
                                               self.rendering_input_res,
                                               self.rendering_input_res)
        novel_views = self.rendering_net(rendering_input)
        self.counter += 1

        novel_views = novel_views.permute(0,2,3,1).reshape(batch_size, -1, 4)

        return {'rgb':novel_views}
