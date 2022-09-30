import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, d_r: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(d_r)
        self.projection = nn.Linear(emb_size, emb_size)
        self.cm = None

    def corrent_matrix(self, z):
        cm = torch.zeros((1, 1, z, z)).cuda()
        x = torch.range(0, z-1, 1, dtype=torch.float32).cuda()
        for i in range(z):
            cm[0, 0, i] = torch.exp(-torch.pow((x-i), 2)/(z/2))

        self.cm = cm


    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:    # b * (z * xy) * d
        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), "b (z xy) (d qkv) -> (qkv) b xy z d", qkv=3, xy=self.num_heads)
        queries, keys, values = qkv[0], qkv[1], qkv[2]

        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min()
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy, dim=-1, dtype=energy.dtype) / scaling
        if self.cm is None:
            self.corrent_matrix(att.shape[-1])
            self.cm = torch.tensor(self.cm, dtype=energy.dtype)
        att = att * self.cm
        att = self.att_drop(att)

        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b xy z d -> b (z xy) d")
        out = self.projection(out)

        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x



class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.Hardswish(), #nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class LayerNorm(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.Norm = nn.InstanceNorm1d(emb_size, affine=True)

    def forward(self, x, **kwargs):
        x = rearrange(x, "b h n-> b n h")
        x = self.Norm(x)
        x = rearrange(x, "b n h-> b h n")
        return x



class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 num_heads: int = 8,
                 d_r: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, **kwargs),
                nn.Dropout(d_r)
            )),
            ResidualAdd(nn.Sequential(
                LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(d_r)
            )
            ))


class TransformerEncoder2D(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])



class TransformerEncoder3D(nn.Module):
    def __init__(self, depth=3, emb_size=128, num_heads=320*360, d_r=0, **kwargs):
        super().__init__()
        self.emb_size = emb_size
        self.Transformer = TransformerEncoder2D(depth=depth, emb_size=emb_size, num_heads=num_heads, d_r=d_r, **kwargs)
        self.act = nn.ReLU()

    def forward(self, img):
        b, m, z, y, x = img.shape
        out = rearrange(img, "b m z y x -> b (z y x) m")
        out = self.Transformer(out)
        out = self.act(out)
        out = rearrange(out, "b (z y x) m -> b m z y x", z=z, y=y)

        return out




class ConvGnRelu3(nn.Module):
    """ classic combination: conv + batch normalization [+ relu]
        post-activation mode """

    def __init__(self, in_channels, out_channels, ksize, stride, padding, do_act=True, bias=True):
        super(ConvGnRelu3, self).__init__()
        if in_channels == out_channels:
            groups = out_channels
        else:
            groups = 1

        self.conv = nn.Conv3d(in_channels, out_channels, (1, ksize, ksize), stride=(1, stride, stride), padding=(0, padding, padding), groups=groups, bias=bias)
        self.do_act = do_act
        if do_act:
            self.act = nn.ReLU()

    def forward(self, input):
        out = self.conv(input)
        if self.do_act:
            out = self.act(out)
        return out


class BottConvGnRelu3(nn.Module):
    """Bottle neck structure"""

    def __init__(self, in_channels, out_channels, ksize, stride, padding, ratio, do_act=True, bias=True):
        super(BottConvGnRelu3, self).__init__()
        self.conv1 = ConvGnRelu3(in_channels, in_channels//ratio, ksize, stride, padding, do_act=True, bias=bias)
        self.conv2 = ConvGnRelu3(in_channels//ratio, in_channels//ratio, ksize, stride, padding, do_act=True, bias=bias)
        self.conv3 = ConvGnRelu3(in_channels//ratio, out_channels, ksize, stride, padding, do_act=do_act, bias=bias)

    def forward(self, input):
        out = self.conv3(self.conv2(self.conv1(input)))
        return out


class ResidualBlock3(nn.Module):
    """ residual block with variable number of convolutions """

    def __init__(self, channels, ksize, stride, padding, num_convs):
        super(ResidualBlock3, self).__init__()

        layers = []
        for i in range(num_convs):
            if i != num_convs - 1:
                layers.append(ConvGnRelu3(channels, channels, ksize, stride, padding, do_act=True))
            else:
                layers.append(ConvGnRelu3(channels, channels, ksize, stride, padding, do_act=False))

        self.ops = nn.Sequential(*layers)
        self.act = nn.ReLU()

    def forward(self, input):

        output = self.ops(input)
        output = self.act(input + output)

        return output


class BottResidualBlock3(nn.Module):
    """ block with bottle neck conv"""

    def __init__(self, channels, ksize, stride, padding, ratio, num_convs):
        super(BottResidualBlock3, self).__init__()
        layers = []
        for i in range(num_convs):
            if i != num_convs - 1:
                layers.append(BottConvGnRelu3(channels, channels, ksize, stride, padding, ratio, do_act=True))
            else:
                layers.append(BottConvGnRelu3(channels, channels, ksize, stride, padding, ratio, do_act=False))

        self.ops = nn.Sequential(*layers)
        self.act = nn.ReLU()

    def forward(self, input):
        output = self.ops(input)
        return self.act(input + output)




class double_res_conv3D(nn.Module):
    def __init__(self, in_ch, out_ch, d_r=0, activation = 'relu', num_convs=1, compression=False):
        super(double_res_conv3D, self).__init__()
        self.d_r = d_r
        self.activation = activation
        self.conv0 = nn.Conv3d(in_ch, out_ch, (1, 5, 5), padding=(0, 2, 2), dilation=(1, 1, 1), groups=out_ch, bias=False)
        if compression:
            self.rblock = BottResidualBlock3(out_ch, 3, 1, 1, 4, num_convs)
        else:
            self.rblock = ResidualBlock3(out_ch, 3, 1, 1, num_convs)

        if self.d_r!=0:
            self.DO = nn.Dropout3d(p=self.d_r)
            self.D1 = nn.Dropout3d(p=self.d_r)
        if self.activation == 'relu':
            self.AF = nn.ReLU()
        else:
            self.AF = nn.LeakyReLU(0.2)

    def forward(self, x):
        if self.d_r !=0:
            output = self.D1(self.rblock(self.AF(self.DO(self.conv0(x)))))
        else:
            output = self.rblock(self.AF(self.conv0(x)))
        return output


class transform_double_res_conv3D(nn.Module):
    def __init__(self, in_ch, out_ch, d_r=0, activation='relu', num_convs=1, compression=False, slicesize=320*360):
        super(transform_double_res_conv3D, self).__init__()
        self.d_r = d_r
        self.activation = activation
        self.conv0 = nn.Conv3d(in_ch, out_ch, (1, 5, 5), padding=(0, 2, 2), dilation=(1, 1, 1), groups=out_ch, bias=False)
        self.transform = TransformerEncoder3D(depth=1, emb_size=out_ch, num_heads=slicesize, d_r=d_r)

        if compression:
            self.rblock = BottResidualBlock3(out_ch, 3, 1, 1, 4, num_convs)
        else:
            self.rblock = ResidualBlock3(out_ch, 3, 1, 1, num_convs)

        if self.d_r != 0:
            self.DO = nn.Dropout3d(p=self.d_r)
            self.D1 = nn.Dropout3d(p=self.d_r)
        if self.activation == 'relu':
            self.AF = nn.ReLU()
        else:
            self.AF = nn.LeakyReLU(0.2)

    def forward(self, x, transform_flag=True):
        if self.d_r != 0:
            output = self.D1(self.rblock(self.AF(self.DO(self.conv0(x)))))
        else:
            output = self.rblock(self.AF(self.conv0(x)))

        if transform_flag:
            output = self.transform(output)


        return output


class inconv3D(nn.Module):
    def __init__(self, in_ch, out_ch, d_r=0, activation='relu'):
        super(inconv3D, self).__init__()
        self.d_r = d_r
        self.conv0 = nn.Conv3d(in_ch, out_ch, (1, 5, 5), padding=(0, 2, 2), dilation=(1, 1, 1), groups=1, bias=False)
        self.conv1 = nn.Conv3d(out_ch, out_ch, (1, 5, 5), padding=(0, 2, 2), dilation=(1, 1, 1), groups=out_ch, bias=False)
        self.conv2 = nn.Conv3d(out_ch, out_ch, (1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1, bias=False)
        self.BN0 = nn.InstanceNorm3d(out_ch, affine=True)
        self.BN1 = nn.InstanceNorm3d(out_ch, affine=True)
        self.activation = activation
        if self.d_r!=0:
            self.DO = nn.Dropout3d(p=self.d_r)
            self.D1 = nn.Dropout3d(p=self.d_r)
        if self.activation == 'relu':
            self.AF0 = nn.ReLU()
            self.AF1 = nn.ReLU()
        else:
            self.AF0 = nn.LeakyReLU(0.2)
            self.AF1 = nn.LeakyReLU(0.2)

    def forward(self, x):
        T = self.BN0(self.conv0(x))
        y0 = self.conv1(T)
        y1 = self.conv2(T + y0)
        y0 = self.AF0(self.BN1(y0))
        y1 = self.AF1(y1)
        if self.d_r !=0:
            y0 = self.DO(y0)
            y1 = self.D1(y1)

        return y0, y1


class outconv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv3D, self).__init__()
        self.out_ch = out_ch
        self.conv = nn.Conv3d(in_ch, out_ch, (1, 5, 5), padding=(0, 2, 2))
    def forward(self, x):
        x1 = self.conv(x)
        if self.out_ch>=2:
            output = F.softmax(x1, dim=1)
        else:
            output = torch.sigmoid(x1)
        return output


class down3D(nn.Module):
    def __init__(self, in_ch, out_ch, d_r=0, activation = 'relu', num_convs=1, compression=False, slicesize=320*360, last=False):
        super(down3D, self).__init__()
        self.d_r = d_r
        self.last = last
        self.activation = activation
        self.conv0 = nn.Sequential(
            nn.AvgPool3d((1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(in_ch, out_ch, (1, 5, 5), padding=(0, 2, 2), dilation=(1, 1, 1), bias=False)
        )
        self.BN0 = nn.InstanceNorm3d(out_ch, affine=True)
        self.BN1 = nn.InstanceNorm3d(out_ch, affine=True)

        if not self.last:
            self.conv2 = nn.Conv3d(out_ch, out_ch, (1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1, bias=False)
            self.conv1 = double_res_conv3D(out_ch, out_ch, d_r=d_r, activation=activation, num_convs=num_convs, compression=compression)

        else:
            self.conv1 = transform_double_res_conv3D(out_ch, out_ch, d_r=d_r, activation=activation, num_convs=num_convs,
                                        compression=compression, slicesize=slicesize)


        if self.d_r!=0:
            self.DO = nn.Dropout3d(p=self.d_r)
            if not self.last:
                self.D1 = nn.Dropout3d(p=self.d_r)

        if self.activation == 'relu':
            self.AF0 = nn.ReLU()
            if not self.last:
                self.AF1 = nn.ReLU()
        else:
            self.AF0 = nn.LeakyReLU(0.2)
            if not self.last:
                self.AF1 = nn.LeakyReLU(0.2)

    def forward(self, x):
        if not self.last:
            T = self.BN0(self.conv0(x))
            y0 = self.conv1(T)
            y1 = self.conv2(T + y0)
            y0 = self.AF0(self.BN1(y0))
            y1 = self.AF1(y1)
            if self.d_r != 0:
                y0 = self.DO(y0)
                y1 = self.D1(y1)
            return y0, y1
        else:
            T = self.BN0(self.conv0(x))
            y0 = self.conv1(T)
            y0 = self.AF0(self.BN1(y0))
            if self.d_r != 0:
                y0 = self.DO(y0)
            return y0



class up3D(nn.Module):
    def __init__(self, in_ch, out_ch, d_r=0, activation = 'relu', num_convs=1, compression=False, slicesize=320*360):
        super(up3D, self).__init__()
        self.upconv =  nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, (1, 2, 2), stride=(1, 2, 2)),
            nn.ReLU())
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, (1, 5, 5), padding=(0, 2, 2), dilation=(1, 1, 1), groups=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU())
        self.Bottle = BottResidualBlock3(out_ch, 3, 1, 1, 4, num_convs)
        self.transform = TransformerEncoder3D(depth=1, emb_size=out_ch, num_heads=slicesize)


    def forward(self, x1, x2, transform_flag=True):
        output = self.Bottle(self.conv(torch.cat((self.upconv(x1), x2), 1)))
        if transform_flag:
            output = self.transform(output)

        return output


class TransContextResUNet(nn.Module):
    def __init__(self, in_channel, out_channel, depth, initial_features, img_size=[], d_r =0, mutex=True, Activation  = 'Relu'):
        super(TransContextResUNet, self).__init__()
        self.name = 'TransContextResUNet'
        self.num_filters = initial_features
        num_filters = initial_features
        self.depth = depth
        self.out_channel = out_channel
        self.down_num_convs_list = [1, 2, 2]
        self.up_num_convs_list = [1, 1, 1]
        ##input layer
        d_ri = d_r * pow(2, -depth)
        self.inconv = inconv3D(in_channel, num_filters, d_r=d_ri, activation= Activation)
        self.down = nn.ModuleList()
        imgsize = img_size
        ## encoder
        for i in range(depth):
            d_ri = d_r * (pow(2, i + 1 - depth))
            num_filters = num_filters << 1
            imgsize = (imgsize[0]>>1, imgsize[1] >> 1, imgsize[2] >> 1)
            self.down.append(down3D(num_filters//2, num_filters, d_r=d_ri, activation= Activation, num_convs=self.down_num_convs_list[i], compression=False, slicesize=imgsize[1]*imgsize[2], last=(i==depth-1)))

        ## decoder
        self.up = nn.ModuleList()
        for i in range(depth):
            d_ri = d_r * (pow(2, -(i + 1)))
            num_filters = num_filters >> 1
            imgsize = (imgsize[0] <<1, imgsize[1] << 1, imgsize[2] << 1)

            self.up.append(up3D(num_filters*2, num_filters, d_r=d_ri, activation= Activation, num_convs=self.up_num_convs_list[i], compression=True, slicesize=imgsize[1]*imgsize[2]))
        ## output convolution layers
        self.outconv = outconv3D(num_filters, out_channel)


    #@autocast()
    def forward(self, x):
        down= {}
        texture = {}
        up = {}
        depth = self.depth
        down[0], texture[0] = self.inconv(x)
        for i in range(depth-1):
            down[i+1], texture[i+1] = self.down[i](down[i])

        up[0] = self.down[depth-1](down[depth-1])
        for i in range(depth):
            up[i+1] = self.up[i](up[i], texture[depth-(i+1)])

        output = self.outconv(up[depth])
        return output

    def trainer(self, x, y, process):
        down = {}
        texture = {}
        up = {}
        depth = self.depth
        down[0], texture[0] = self.inconv(x)
        for i in range(depth - 1):
            down[i + 1], texture[i + 1] = self.down[i](down[i])

        up[0] = self.down[depth - 1](down[depth - 1])
        for i in range(depth):
            up[i + 1] = self.up[i](up[i], texture[depth - (i + 1)])

        output = self.outconv(up[depth])

        loss, dice = self.loss(output, y, process=process)

        return loss, dice, output

    def valider(self, x, y):
        down = {}
        texture = {}
        up = {}
        depth = self.depth
        down[0], texture[0] = self.inconv(x)
        for i in range(depth - 1):
            down[i + 1], texture[i + 1] = self.down[i](down[i])

        up[0] = self.down[depth - 1](down[depth - 1])
        for i in range(depth):
            up[i + 1] = self.up[i](up[i], texture[depth - (i + 1)])

        output = self.outconv(up[depth])

        loss, dice = self.loss.dice(output, y)

        return loss, dice, output


    def run(self, x):
        down= {}
        texture = {}
        up = {}
        depth = self.depth
        down[0], texture[0] = self.inconv(x)
        for i in range(depth-1):
            down[i+1], texture[i+1] = self.down[i](down[i])

        down[depth] = self.down[depth-1](down[depth-1])
        up[0] = down[depth]
        for i in range(depth):
            up[i+1] = self.up[i](up[i], texture[depth-(i+1)])

        output = self.outconv(up[depth])
        return output, down

