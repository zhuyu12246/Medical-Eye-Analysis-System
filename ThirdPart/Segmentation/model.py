import torch.nn as nn
import torch
from torch import autograd
import torch.nn.functional as F
from torchvision import models
from functools import partial


nonlinearity = partial(F.relu, inplace=True)

def dwt_init(x):
    '''
    haar wavelet decomposition
    '''
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]

    ll = x1 + x2 + x3 + x4
    hl = -(-x1 - x2 + x3 + x4)
    lh = -(-x1 + x2 - x3 + x4)
    hh = -(x1 - x2 - x3 + x4)

    #Normalization
    amin, amax = ll.min(), ll.max()
    ll = (ll - amin) / (amax - amin)
    amin, amax = lh.min(), lh.max()
    lh = (lh - amin) / (amax - amin)
    amin, amax = hl.min(), hl.max()
    hl = (hl - amin) / (amax - amin)
    amin, amax = hh.min(), hh.max()
    hh = (hh - amin) / (amax - amin)

    return torch.cat((ll, lh, hl, hh), 1)

def dwt_init_N(x):
    '''
    haar wavelet decomposition
    '''
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]

    ll = x1 + x2 + x3 + x4
    hl = -(-x1 - x2 + x3 + x4)
    lh = -(-x1 + x2 - x3 + x4)
    hh = -(x1 - x2 - x3 + x4)

    #Normalization
    amin, amax = ll.min(), ll.max()
    ll = (ll - amin) / (amax - amin)
    amin, amax = lh.min(), lh.max()
    lh = (lh - amin) / (amax - amin)
    amin, amax = hl.min(), hl.max()
    hl = (hl - amin) / (amax - amin)
    amin, amax = hh.min(), hh.max()
    hh = (hh - amin) / (amax - amin)

    return torch.cat((lh, hl, hh), 1)

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)

class WBRefineBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(WBRefineBlock, self).__init__()
        self.dwt = DWT()
        self.conv1_1 = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.dwt(x)
        x = self.conv1_1(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class Our_WRB(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(Our_WRB, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        #self.firstconv = resnet.conv1
        self.firstconv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.dwt1 = WBRefineBlock(64 * 4, 64)
        self.dwt2 = WBRefineBlock(128 * 4, 128)
        self.dwt3 = WBRefineBlock(256 * 4, 256)

        self.decoder4 = DecoderBlock(512+256, filters[2])
        self.decoder3 = DecoderBlock(filters[2]+128, filters[1])
        self.decoder2 = DecoderBlock(filters[1]+64, filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        dwt1 = self.dwt1(e1)
        e2 = self.encoder2(e1)
        dwt2 = self.dwt2(e2)
        e3 = self.encoder3(e2)
        dwt3 = self.dwt3(e3)
        e4 = self.encoder4(e3)

        e4 = torch.cat([e4, dwt3], dim=1)
        d4 = torch.cat([self.decoder4(e4), dwt2], dim=1) 
        d3 = torch.cat([self.decoder3(d4), dwt1], dim=1)
        d2 = self.decoder2(d3)
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def  forward(self, x):

        return self.conv(x)

# 基本网路单元
class DoubleConv_att(nn.Module):
    def __init__(self, in_ch, out_ch, g=2, channel_att=True, spatial_att=True):
        super(DoubleConv_att, self).__init__()
        self.channel_att = channel_att
        self.spatial_att = spatial_att
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )
        self.relu_inplace = nn.ReLU(inplace=True)

        if channel_att:
            self.att_c = nn.Sequential(
                nn.Conv2d(2 * out_ch, out_ch // g, 1, 1, 0),
                nn.ReLU(),
                nn.Conv2d(out_ch // g, out_ch, 1, 1, 0),
                nn.Sigmoid()
            )
        if spatial_att:
            self.att_s = nn.Sequential(
                nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3),
                nn.Sigmoid()
            )

    def forward(self, data):
        """
        Forward function.
        :param data:
        :return: tensor
        """
        fm = self.conv1(data)

        if self.channel_att:
            # fm_pool = F.adaptive_avg_pool2d(fm, (1, 1)) + F.adaptive_max_pool2d(fm, (1, 1))
            fm_pool = torch.cat([F.adaptive_avg_pool2d(fm, (1, 1)), F.adaptive_max_pool2d(fm, (1, 1))], dim=1)
            att = self.att_c(fm_pool)
            fm = fm * att
        if self.spatial_att:
            fm_pool = torch.cat([torch.mean(fm, dim=1, keepdim=True), torch.max(fm, dim=1, keepdim=True)[0]], dim=1)
            att = self.att_s(fm_pool)
            fm = fm * att
        return fm

class Unet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet, self).__init__()
        self.conv1 = DoubleConv(in_ch,64)#输入是但通道的图片,经过一个二维卷积,输出通道数变为64,不关注尺寸变化
        self.pool1 = nn.MaxPool2d(2)#经过一个2*2的下采样,尺寸缩小一半,通道数增加一倍
        self.conv2 = DoubleConv(64, 128)#输入是64通道,输出为128通道
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        # 没有 pool5池化

        # 这里的转置卷积对应的其实是maxpool,转置卷积之后图像尺寸翻倍
        # 这里没有用maxpool对应的unpooling,而是用了反卷积，
        # 那这里的卷积核是学习到的，还是和pool公用的
        # 为什么decode的中的卷积用的是卷积，而不是反卷积，差别在哪里
        self.up6 = nn.ConvTranspose2d(1024, 512, 2,stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6,c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        out = nn.Sigmoid()(c10)
        #out = nn.Softmax(dim=0)(c10)
        return out


#重新定义一个segnet,按照编码加上解码
class Encoder(nn.Module):
    def __init__(self, input_channels):
        super(Encoder, self).__init__()

        batchNorm_momentum = 0.1
        self.enco1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=batchNorm_momentum),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=batchNorm_momentum),
            nn.ReLU()
        )
        self.enco2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=batchNorm_momentum),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=batchNorm_momentum),
            nn.ReLU()
        )
        self.enco3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=batchNorm_momentum),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=batchNorm_momentum),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=batchNorm_momentum),
            nn.ReLU()
        )
        self.enco4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=batchNorm_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=batchNorm_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=batchNorm_momentum),
            nn.ReLU()
        )
        self.enco5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=batchNorm_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=batchNorm_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=batchNorm_momentum),
            nn.ReLU()
        )

    def forward(self, x):
        id = []#输出的池化索引值形成一个列表

        x = self.enco1(x)
        x, id1 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)  # 经过最大池化后本来只输出一个值,现在还输出了它的位置
        id.append(id1)
        x = self.enco2(x)
        x, id2 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        id.append(id2)
        x = self.enco3(x)
        x, id3 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        id.append(id3)
        x = self.enco4(x)
        x, id4 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        id.append(id4)
        x = self.enco5(x)
        x, id5 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        id.append(id5)

        return x, id

class SegNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(SegNet, self).__init__()

        self.weights_new = self.state_dict()
        self.encoder = Encoder(input_channels)
        batchNorm_momentum = 0.1
        self.deco1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=batchNorm_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=batchNorm_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=batchNorm_momentum),
            nn.ReLU()
        )
        self.deco2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=batchNorm_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=batchNorm_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=batchNorm_momentum),
            nn.ReLU()
        )
        self.deco3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=batchNorm_momentum),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=batchNorm_momentum),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=batchNorm_momentum),
            nn.ReLU()
        )
        self.deco4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=batchNorm_momentum),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=batchNorm_momentum),
            nn.ReLU()
        )
        self.deco5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=batchNorm_momentum),
            nn.ReLU(),
            nn.Conv2d(64, output_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()#这里怎们说?我们的任务是二分类?
        )

    def forward(self, x):
        x, id = self.encoder(x)#调用enconder

        x = F.max_unpool2d(x, id[4], kernel_size=2, stride=2)
        x = self.deco1(x)
        x = F.max_unpool2d(x, id[3], kernel_size=2, stride=2)
        x = self.deco2(x)
        x = F.max_unpool2d(x, id[2], kernel_size=2, stride=2)
        x = self.deco3(x)
        x = F.max_unpool2d(x, id[1], kernel_size=2, stride=2)
        x = self.deco4(x)
        x = F.max_unpool2d(x, id[0], kernel_size=2, stride=2)
        x = self.deco5(x)

        return x


"构造一个gan网络"
class DNet(nn.Module):
    """
    首先定义判别器discriminator:目标是为了判断输入的图片是真图片还是假图片
    所以可以被看作是二分类网络
    """

    def __init__(self, opt):
        super(DNet, self).__init__()
        ndf = opt.ndf #判别器channel值
        self.DNet = nn.Sequential(
            # 输入 3 x 96 x 96,本实验用的是3*512*512
            # kernel_size = 5,stride = 3, padding =1
            # 按式子计算 floor((96 + 2*1 - 1*(5-1) - 1)/3 + 1) = 32
            # 是same卷积，96/32 = stride = 3
            nn.Conv2d(3, ndf, 5, 3, 1, bias=False),#ndf是判别器输出通道
            nn.LeakyReLU(0.2, inplace=True),
            # 输出 (ndf) x 32 x 32
            #kernel_size = 4,stride = 2, padding =1
            #按式子计算 floor((32 + 2*1 - 1*(4-1) - 1)/2 + 1) = 16
            #是same卷积，32/16 = stride = 2
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),#二维卷积后通道数变为2倍?
            nn.BatchNorm2d(ndf * 2),#对输出的通道数据进行归一化
            nn.LeakyReLU(0.2, inplace=True),
            # 输出 (ndf*2) x 16 x 16

            #kernel_size = 4,stride = 2, padding =1
            #按式子计算 floor((16 + 2*1 - 1*(4-1) - 1)/2 + 1) = 8
            #是same卷积，16/8 = stride = 2           
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出 (ndf*4) x 8 x 8

            #kernel_size = 4,stride = 2, padding =1
            #按式子计算 floor((8 + 2*1 - 1*(4-1) - 1)/2 + 1) = 4
            #是same卷积，8/4 = stride = 2
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出 (ndf*8) x 4 x 4

            #kernel_size = 4,stride = 1, padding =0
            #按式子计算 floor((4 + 2*0 - 1*(4-1) - 1)/1 + 1) = 1
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            #输出为1*1*1
            nn.Sigmoid()  # 返回[0,1]的值，输出一个数(作为概率值)
        )

    def forward(self, input):
        return self.DNet(input).view(-1) #输出从1*1*1变为1，得到生成器生成假图片的分数，分数高则像真图片?不用设置这个数吗?学习得到的吗?
    

class GNet(nn.Module):
    """
    生成器generator的定义,目标是尽可能地生成以假乱真的图片，让判别器以为这是真的图片,输入一个随机噪声，生成一张图片,使用的是DCGAN
    """

    def __init__(self, opt):
        super(GNet, self).__init__()
        ngf = opt.ngf  # 生成器feature map数channnel，默认为64?人家是输出为3*64*64的图片啊

        self.GNet = nn.Sequential(
            # 输入是一个nz维度(默认为100)的噪声，我们可以认为它是一个1*1*nz的feature map
            # kernel_size = 4,stride = 1, padding =0
            # 根据计算式子 (1-1)*1 - 2*0 + 4 + 0 = 4
            nn.ConvTranspose2d(opt.nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 上一步的输出形状：(ngf*8) x 4 x 4

            #kernel_size = 4,stride = 2, padding =1
            #根据计算式子 (4-1)*2 - 2*1 + 4 + 0 = 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 上一步的输出形状： (ngf*4) x 8 x 8

            #kernel_size = 4,stride = 2, padding =1
            #根据计算式子 (8-1)*2 - 2*1 + 4 + 0 = 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 上一步的输出形状： (ngf*2) x 16 x 16

            #kernel_size = 4,stride = 2, padding =1
            #根据计算式子 (16-1)*2 - 2*1 + 4 + 0 = 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 上一步的输出形状：(ngf) x 32 x 32

            # kernel_size = 5,stride = 3, padding =1
            #根据计算式子 (32-1)*3 - 2*1 + 5 + 0 = 96
            nn.ConvTranspose2d(ngf, 3, 5, 3, 1, bias=False),
            nn.Tanh()  # 输出范围 -1~1 故而采用Tanh
            # 输出形状：3 x 96 x 96
        )

    def forward(self, input):
        return self.GNet(input)
    
