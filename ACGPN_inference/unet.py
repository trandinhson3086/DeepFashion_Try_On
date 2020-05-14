import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.nn.utils.spectral_norm as spectral_norm

class VGGExtractor(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGExtractor, self).__init__()
        vgg16 = torchvision.models.vgg16(pretrained=True).eval()
        blocks = []
        blocks.append(vgg16.features[:4])
        blocks.append(vgg16.features[4:9])
        blocks.append(vgg16.features[9:16])
        blocks.append(vgg16.features[16:23])
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize

    def forward(self, input):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
        output = []
        for block in self.blocks:
            input = block(input)
            output.append(input)
        return output

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class UpSimple(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1):
        x1 = self.up(x1)
        return self.conv(x1)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return nn.Tanh()(self.conv(x))


class Discriminator(nn.Module):
    def __init__(self, ndf=64):
        super().__init__()
        self.ndf = ndf
        self.local_conv, self.local_cam, self.local_last = self.build_disc(True)
        self.global_conv, self.global_cam, self.global_last = self.build_disc(False)

    def build_disc(self, is_local):
        n_layers = 5 if is_local else 8

        model = [nn.ReflectionPad2d(1),
                 spectral_norm(nn.Conv2d(3, self.ndf, kernel_size=4, stride=2, padding=0, bias=True)),
                 nn.LeakyReLU(0.2, True)]

        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            model += [nn.ReflectionPad2d(1),
                      spectral_norm(nn.Conv2d(self.ndf * mult, self.ndf * mult * 2, kernel_size=4, stride=2, padding=0, bias=True)),
                      nn.LeakyReLU(0.2, True)]

        mult = 2 ** (n_layers - 2 - 1)
        model = nn.Sequential(*model)
        cam = CAM(self.ndf*mult, is_disc=True)
        last_conv = spectral_norm(nn.Conv2d(self.ndf*mult, 1, 4, padding=0, stride=1, bias=False))

        return model, cam, last_conv

    def get_feat(self, input, conv, cam, conv_last):
        feats = conv(input)
        cam_output, cam_logit = cam(feats)
        output = conv_last(cam_output)
        return output, cam_logit

    def forward(self, input):
        local, local_cam_logit = self.get_feat(input, self.local_conv, self.local_cam, self.local_last)
        glob, glob_cam_logit = self.get_feat(input, self.global_conv, self.global_cam, self.global_last)
        return local, local_cam_logit, glob, glob_cam_logit

class CAM(nn.Module):
    def __init__(self, in_channel, is_disc=True):
        super().__init__()

        # Class Activation Map
        self.gap_fc = nn.Linear(in_channel, 1, bias=False)
        self.gmp_fc = nn.Linear(in_channel, 1, bias=False)
        self.conv1x1 = nn.Conv2d(in_channel * 2, in_channel, kernel_size=1, stride=1, bias=True)
        if is_disc:
            self.gap_fc = spectral_norm(self.gap_fc)
            self.gmp_fc = spectral_norm(self.gmp_fc)
            self.conv1x1 = spectral_norm(self.conv1x1)
            self.activation = nn.LeakyReLU(0.2, True)
        else:
            self.activation = nn.ReLU(True)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        gap = self.avgpool(x)
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = self.maxpool(x)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        x = torch.cat([gap, gmp], 1)
        x = self.activation(self.conv1x1(x))

        return x, cam_logit
