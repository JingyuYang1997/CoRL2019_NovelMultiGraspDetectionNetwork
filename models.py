import torch.nn as nn
import torch.nn.functional as F
import torch
import pdb

class Bottleneck(nn.Module):

    def __init__(self, inplanes=256, planes=128, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out



class HourGlass(nn.Module):

    def __init__(self, block, num_classes=16):
        super(HourGlass, self).__init__()
        self.block = block
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 256, kernel_size=7, stride=2, padding=3,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)

        # preprocess the image
        self.block_pre = self.block()

        # the residual module in the master branch
        self.block_master_1 = self.block()
        self.block_master_2 = self.block()
        self.block_master_3 = self.block()
        self.block_master_4 = self.block()

        # the residual module in the minor branch
        self.block_minor_1 = self.block()
        self.block_minor_2 = self.block()
        self.block_minor_3 = self.block()
        self.block_minor_4 = self.block()

        
        # 1x1 feature conv
        self.feat_conv = nn.Conv2d(256, self.num_classes, 1)


        self.pool = nn.MaxPool2d(2, stride=2)

        # upconv
        self.upconv1 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv2 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv3 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv4 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.block_master_5 = self.block()
        self.block_master_6 = self.block()
        self.block_master_7 = self.block()
        self.block_master_8 = self.block()
    
    def forward(self, x):
        
        # preprocess downsample to 64x64 
        x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        x = self.block_pre(x)
        x = self.pool(x)

        # everytime before pooling will have a minor branch
        # downsample to 32x32 
        x_minor_1 = x
        x_pooling1 = self.pool(x)
        x = self.block_master_1(x_pooling1)


        # downsample to 16x16 
        x_minor_2 = x
        x_pooling2 = self.pool(x)
        x = self.block_master_2(x_pooling2)

        # downsample to 8x8 
        x_minor_3 = x
        x_pooling3 = self.pool(x)
        x = self.block_master_3(x_pooling3)

        # downsample to 4x4 (lowest) 
        x_minor_4 = x
        x_pooling4 = self.pool(x)
        x = self.block_master_4(x_pooling4)

        # the minor branch to upsample and skip connection
        x_minor_1 = self.block_minor_1(x_minor_1)
        x_minor_2 = self.block_minor_2(x_minor_2)
        x_minor_3 = self.block_minor_3(x_minor_3)
        x_minor_4 = self.block_minor_4(x_minor_4)


        x_merge1 = self.upconv1(x)
        x_merge1 = self.block_master_5(x_merge1)
        x_merge1 += x_minor_4

        x_merge2 = self.upconv2(x_merge1)
        x_merge2 = self.block_master_6(x_merge2)
        x_merge2 += x_minor_3

        x_merge3 = self.upconv3(x_merge2)
        x_merge3 = self.block_master_7(x_merge3)
        x_merge3 += x_minor_2

        x_merge4 = self.upconv4(x_merge3)
        x_merge4 = self.block_master_8(x_merge4)
        x_merge4 += x_minor_1
        
        x = self.feat_conv(x_merge4)

        
        return x
    
        
class HourGlassModule(nn.Module):

    def __init__(self, block, num_classes=16):
        """
        HourGlassModule
        input Nx256x64x64
        output Nx256x64x64
        """
        super(HourGlassModule, self).__init__()
        self.block = block
        self.num_classes = num_classes


        # preprocess the image
        self.block_pre = self.block()

        # the residual module in the master branch
        self.block_master_1 = self.block()
        self.block_master_2 = self.block()
        self.block_master_3 = self.block()
        self.block_master_4 = self.block()

        # the residual module in the minor branch
        self.block_minor_1 = self.block()
        self.block_minor_2 = self.block()
        self.block_minor_3 = self.block()
        self.block_minor_4 = self.block()

        


        self.pool = nn.MaxPool2d(2, stride=2)

        # upconv
        self.upconv1 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv2 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv3 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv4 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.block_master_5 = self.block()
        self.block_master_6 = self.block()
        self.block_master_7 = self.block()
        self.block_master_8 = self.block()
    
    def forward(self, x):
        
        # preprocess downsample to 64x64 
        # x = self.bn1(x)
        # x = self.relu(x)
        x = self.block_pre(x)
        # x = self.pool(x)

        # everytime before pooling will have a minor branch
        # downsample to 32x32 
        x_minor_1 = x
        x_pooling1 = self.pool(x)
        x = self.block_master_1(x_pooling1)


        # downsample to 16x16 
        x_minor_2 = x
        x_pooling2 = self.pool(x)
        x = self.block_master_2(x_pooling2)

        # downsample to 8x8 
        x_minor_3 = x
        x_pooling3 = self.pool(x)
        x = self.block_master_3(x_pooling3)

        # downsample to 4x4 (lowest) 
        x_minor_4 = x
        x_pooling4 = self.pool(x)
        x = self.block_master_4(x_pooling4)

        # the minor branch to upsample and skip connection
        x_minor_1 = self.block_minor_1(x_minor_1)
        x_minor_2 = self.block_minor_2(x_minor_2)
        x_minor_3 = self.block_minor_3(x_minor_3)
        x_minor_4 = self.block_minor_4(x_minor_4)


        x_merge1 = self.upconv1(x)
        x_merge1 = self.block_master_5(x_merge1)
        x_merge1 += x_minor_4

        x_merge2 = self.upconv2(x_merge1)
        x_merge2 = self.block_master_6(x_merge2)
        x_merge2 += x_minor_3

        x_merge3 = self.upconv3(x_merge2)
        x_merge3 = self.block_master_7(x_merge3)
        x_merge3 += x_minor_2

        x_merge4 = self.upconv4(x_merge3)
        x_merge4 = self.block_master_8(x_merge4)
        x_merge4 += x_minor_1

        return torch.sigmoid(x_merge4, biased=False)

class HourGlassV2(nn.Module):

    def __init__(self, block, num_classes=15, num_stack=2):
        super(HourGlassV2, self).__init__()
    
        for i in range(num_stack):
            exec('self.hg{}=HourGlassModule(block, num_classes)'.format(i+1))
            exec('self.feat_conv{}=nn.Conv2d(256, num_classes, 1)'.format(i+1))
        for i in range(num_stack-1):
            exec('self.upfeat_conv{}=nn.Conv2d(num_classes, 256, 1)'.format(i+1))

        # pre conv before hourglass module
        self.conv1 = nn.Conv2d(1, 256, kernel_size=3, stride=1, padding=1,
                               bias=True)

        self.num_stack = num_stack
        self.num_classes = num_classes

    def forward(self, x):
        x = self.conv1(x)
        output = []
        
        for i in range(self.num_stack-1):
            x = eval('self.hg{}(x)'.format(i+1))
            x_feat = eval('self.feat_conv{}(x)'.format(i+1))
            x_upfeat = eval('self.upfeat_conv{}(x_feat)'.format(i+1))
            output.append(x_feat)
            x = x + x_upfeat

        x_feat_final = eval('self.feat_conv{}(x)'.format(self.num_stack))
        output.append(x_feat_final)
        return output
        
            

if __name__ == "__main__":
    net = HourGlass(Bottleneck, num_classes=10)
    x = torch.FloatTensor(1, 3, 256, 256)
    y = net(x)
