import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.kv = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False)
        self.kv_conv = DeformableConv2d(channels * 2, channels *2)#nn.Conv2d(channels * 2, channels *2, kernel_size=3, padding=1, groups=channels * 2, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x, q):
        b, c, h, w = x.shape
        k, v = self.kv_conv(self.kv(x)).chunk(2, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
        return out, q,attn

class Nested_MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(Nested_MDTA, self).__init__()
        self.pack_attention = MDTA(channels, num_heads)
        self.unpack_attention = MDTA(channels, num_heads)

    def forward(self,x, p):
        packed_context, query,packed_attn = self.pack_attention(x, p)
        unpacked_context, _,unpacked_attn = self.unpack_attention(packed_context, query)
        return unpacked_context, packed_context,packed_attn@unpacked_attn

class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GDFN, self).__init__()

        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1,
                              groups=hidden_channels * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x

class LunaTransformerEncoderLayer(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor):
        super(LunaTransformerEncoderLayer, self).__init__()
        self.luna_attention = Nested_MDTA(channels, num_heads)
        self.feed_forward = GDFN(channels, expansion_factor)
        self.packed_context_layer_norm = nn.LayerNorm(channels)
        self.unpacked_context_layer_norm = nn.LayerNorm(channels)
        # self.unpacked_context_layer_norm = nn.LayerNorm(channels)
        self.feed_forward_layer_norm = nn.LayerNorm(channels)

    def forward(self, x, p):
        b, c, h, w = x.shape
        unpacked_context, packed_context,attn = self.luna_attention(x,p)

        packed_context = self.packed_context_layer_norm((packed_context + p).reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1).contiguous().reshape(b, c, h, w)

        unpacked_context = self.unpacked_context_layer_norm((unpacked_context + x).reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1).contiguous().reshape(b, c, h, w)

        outputs = self.feed_forward(unpacked_context)

        outputs = self.feed_forward_layer_norm((outputs + unpacked_context).reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1).contiguous().reshape(b, c, h, w)

        return outputs, packed_context,attn


class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):

        super(DeformableConv2d, self).__init__()
        
        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        
        self.offset_conv = nn.Conv2d(in_channels, 
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        
        self.modulator_conv = nn.Conv2d(in_channels, 
                                     1 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)
        
        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x):
        #h, w = x.shape[2:]
        #max_offset = max(h, w)/4.

        offset = self.offset_conv(x)#.clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        
        x = torchvision.ops.deform_conv2d(input=x, 
                                          offset=offset, 
                                          weight=self.regular_conv.weight, 
                                          bias=self.regular_conv.bias, 
                                          padding=self.padding,
                                          stride=self.stride,
                                          )
        return x



class res_bottleneck(nn.Module):
    def __init__(self, in_ch, num_heads, expansion_factor = 2.21):
        super(res_bottleneck, self).__init__()


        self.res_block1 = LunaTransformerEncoderLayer(in_ch, num_heads, expansion_factor)
        #self.res_block2 = LunaTransformerEncoderLayer(in_ch, num_heads, expansion_factor)
        self.res_last = nn.Conv2d(in_ch*2, in_ch, kernel_size=1, bias=False)

    def forward(self, x,p):

        res1, res1_p1,attn1 = self.res_block1(x,p)

        #res2, res1_p2,attn2 = self.res_block2(res1, res1_p1)


        res = torch.cat([res1, res1_p1], axis = 1)
        out = self.res_last(res)

        return out,res1,res1_p1
