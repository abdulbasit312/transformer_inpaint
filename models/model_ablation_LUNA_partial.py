import torch.nn as nn
import logging, torch, yaml
import numpy as np
from .RDB_layer import RDB
from einops import rearrange
from .Nested_Deformed_transformer_partial import res_bottleneck
from .functions import PartialConv2d
import torch.nn.functional as F


logger = logging.getLogger(__name__)
class UpSample(nn.Module):
    """ sample with convolutional operation
    :param input_nc: input channel
    :param with_conv: use convolution to refine the feature
    :param kernel_size: feature size
    :param return_mask: return mask for the confidential score
    """
    def __init__(self, input_nc,output_nc, with_conv=True, kernel_size=3, return_mask=False,stride=1):
        super(UpSample, self).__init__()
        self.with_conv = with_conv
        self.return_mask = return_mask
        if self.with_conv:
            self.conv = PartialConv2d(input_nc, output_nc, kernel_size=kernel_size, stride=stride,
                                      padding=int(int(kernel_size-1)/2), return_mask=True,multi_channel=False)

    def forward(self, x, mask=None):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        mask = F.interpolate(mask, scale_factor=2, mode='bilinear', align_corners=True) if mask is not None else mask
        if self.with_conv:
            x, mask = self.conv(x, mask)
        if self.return_mask:
            return x, mask
        else:
            return x


class DownSample(nn.Module):
    """ sample with convolutional operation
        :param input_nc: input channel
        :param with_conv: use convolution to refine the feature
        :param kernel_size: feature size
        :param return_mask: return mask for the confidential score
    """
    def __init__(self, input_nc,output_nc, with_conv=True, kernel_size=3, return_mask=True,stride=2):
        super(DownSample, self).__init__()
        self.with_conv = with_conv
        self.return_mask = return_mask
        if self.with_conv:
            self.conv = PartialConv2d(input_nc, output_nc, kernel_size=kernel_size, stride=stride,
                                      padding=int(int(kernel_size-1)/2), return_mask=True, multi_channel=False)

    def forward(self, x, mask=None):
        if self.with_conv:
            x, mask = self.conv(x, mask)
        if self.return_mask:
            return x, mask
        else:
            return x

class inpaint_model(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, args):
        super().__init__()

        #first three layers 256>128>64>32
        self.downsample1=DownSample(3,64,stride=1)
        self.downsample2=DownSample(64,128)
        self.downsample3=DownSample(128,256)
        self.downsample4=DownSample(256,256)
        self.downsample5=DownSample(256,256)

    
        # self.n_head = args['n_head']


        self.deformed_conv1=res_bottleneck(128,4)
        self.deformed_conv2=res_bottleneck(128,4)
        # decoder, input: 32*32*config.n_embd
        self.ln_f = nn.LayerNorm(128)
        self.ln_xLxG = nn.LayerNorm(32)
        self.act = nn.ReLU(True)

        # layer local (RDB)
        self.RDB1 = RDB(args['nFeat'], args['nDenselayer'], args['growthRate'])
        self.RDB2 = RDB(args['nFeat'], args['nDenselayer'], args['growthRate'])
        self.RDB3 = RDB(args['nFeat'], args['nDenselayer'], args['growthRate'])
        self.RDB4 = RDB(args['nFeat'], args['nDenselayer'], args['growthRate'])

        self.GFF_1x1 = nn.Conv2d(args['nFeat'] * 4, args['nFeat'], kernel_size=1, padding=0, bias=True)  
        self.GFF_3x3 = nn.Conv2d(args['nFeat'], args['nFeat'], kernel_size=3, padding=1, bias=True)

    
        self.convt_LG = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.batchNorm = nn.BatchNorm2d(256)

        #last three layers 32>64>128>256>256
        self.convt1 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.convt2 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)

        self.convt3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)

        self.convt4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)


        self.padt = nn.ReflectionPad2d(3)
        self.convt5 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0)


        self.act_last = nn.Sigmoid()    # ZitS first stage

        self.block_size = 32
        #self.config = config

        self.apply(self._init_weights)

        # calculate parameters
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):    #https://wandb.ai/wandb_fc/tips/reports/How-to-Initialize-Weights-in-PyTorch--VmlldzoxNjcwOTg1
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d, nn.ConvTranspose2d)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, args, new_lr):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.BatchNorm2d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias') or pn.endswith("temperature"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        # no_decay.add('pos_emb')
        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)
        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": float(args['weight_decay'])},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=float(new_lr), betas=(0.9, 0.95))
        return optimizer

    def forward(self, img_idx, masks=None):
        mask_inverted=(1-masks)
        img_idx=img_idx*mask_inverted
        x1,mask1=self.downsample1(img_idx,mask_inverted) #bs * 64* 256 * 256
        x1 = self.act(x1)

        x2,mask2=self.downsample2(x1,mask1) # bs *128 * 128 * 128
        x2=self.act(x2)
        
        x3,mask3=self.downsample3(x2,mask2) # bs * 256 * 64 * 64
        x3=self.act(x3)

        x4,mask4=self.downsample4(x3,mask3) #bs * 256 * 32 * 32
        x4=self.act(x4)

        x5,mask5=self.downsample5(x4,mask4) # bs * 256 * 32 * 32
        x5=self.act(x5)

        xG, xL = torch.split(x5, [128, 128], dim=1)


        [b, c, h, w] = xG.shape
        
        #xG => [bs, 128, 32, 32]
        xG,res1,res1_p1=self.deformed_conv1(xG,xG)

        #xG, att2, att4 = self.cswin_Transformer_layers(xG)
        #xG.shape=>[bs,128,32,32]
        #
        #xL1 = self.RDB1(xL)   # 4 cnn layer
        #xL2 = self.RDB2(xL1)  # 4 cnn layer
        '''
        XL2mid_1, XL2mid_2 = torch.split(xL2, [64, 64], dim=1)
        #  h1, hsp_1, w_1, wsp_1 = 1, H, W // self.sp, self.sp  # vertical
        #  h2, hsp_2, w_2, wsp_2 = H // self.sp, self.sp, 1, W  # horizontal
        XL2mid_1 = rearrange(XL2mid_1, ' b (c head) (h hsp) (w wsp) -> (b h w) head c (hsp wsp)',
                             head=self.n_head[1], h=1,  hsp=h, w=w // self.sp_size[1], wsp=self.sp_size[1])
        XL2mid_2 = rearrange(XL2mid_2, ' b (c head) (h hsp) (w wsp) -> (b h w) head c (hsp wsp)',
                             head=self.n_head[1], h=h // self.sp_size[1], hsp=self.sp_size[1], w=1, wsp=w)
        '''
        
        #xmid2_1 = XL2mid_1 @ att2[0]
        #xmid2_2 = XL2mid_2 @ att2[1]
        '''
        xmid2_1 = rearrange(xmid2_1, '(b h w) head c (hsp wsp) -> b (c head) (h hsp) (w wsp)',
                             head=self.n_head[1], h=1,  hsp=h, w=w // self.sp_size[1], wsp=self.sp_size[1])
        xmid2_2 = rearrange(xmid2_2, '(b h w) head c (hsp wsp) -> b (c head) (h hsp) (w wsp)',
                             head=self.n_head[1], h=h // self.sp_size[1], hsp=self.sp_size[1], w=1, wsp=w)
        xmid2 = torch.cat((xmid2_1, xmid2_2), dim=1)
        '''
        #xL2+=xG
        #xL3 = self.RDB3(xL2)  # 4 cnn layer
        #xL4 = self.RDB4(xL3)  # 4 cnn layer
        xG,res2,res2_p1=self.deformed_conv2(res1,res1_p1)
        #XL4mid =  rearrange(xL4, ' b (c head) (h hsp) (w wsp) -> (b h w) head c (hsp wsp)',
        #                     head=self.n_head[3], h=1,  hsp=self.sp_size[3], w=1, wsp=self.sp_size[3])

        #XL4mid = XL4mid @ att4[0]
        #XL4mid = rearrange(XL4mid, '(b h w) head c (hsp wsp) -> b (c head) (h hsp) (w wsp)',
        #                    head=self.n_head[3], h=1,  hsp=self.sp_size[3], w=1, wsp=self.sp_size[3])
        
        #xL4 = xL4+xG #+ xmid2 + XL4mid
        xG = xG #+ xmid2 + XL4mid

        #XLn = torch.cat((xL1, xL2, xL3, xL4), 1)  # concat
        #XLn = self.GFF_1x1(XLn)
        #XLn = self.GFF_3x3(XLn)
        #xL = XLn + xL  # Residual

        # x = xG + xL     # x + L + G
        #xL = self.act(self.ln_xLxG(xL))
        xG = self.act(self.ln_xLxG(xG))
        x = torch.cat((xG, xL), 1)  # concat  BS * 256 * 32 * 32

        x = self.convt1(x)
        x = self.act(x) + x4

        x = self.convt2(x)
        x = self.act(x) + x3 

        x = self.convt3(x)
        x = self.act(x) + x2

        x = self.convt4(x)
        x = self.act(x) + x1

        x = self.padt(x)
        x = self.convt5(x) # 3*512*512


        x = self.act_last(x)    # Sigmoid

        return x

