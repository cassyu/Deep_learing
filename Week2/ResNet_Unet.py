import torch
import torch.nn as nn
import torch.nn.functional as F

#下采样block
class UNetConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []  # 新建空的列表，要往里面填东西

        block.append(nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())

        # 检测有没有BN
        if batch_norm:
            block.append(nn.BatchNorm2d(out_chans))

        block.append(nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_chans))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out

#上采样block
class UNetUpBlock(nn.Module):
    def __init__(self,in_channel,out_channel,up_mode,padding,batch_norm):
        super(UNetUpBlock,self).__init__()
        if up_mode == 'upconv':#转置卷积
            self.up = nn.ConvTranspose2d(in_channel,out_channel,kernel_size=2,stride=2)
        elif up_mode == 'upsample':#传统网络的用法
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear',scale_factor=2),
                nn.Conv2d(in_channel,out_channel,kernel_size=1)
            ,)

        self.conv_block = UNetConvBlock(in_channel,out_channel,padding,batch_norm)

    #对featuremap  裁剪
    def center_crop(self,layer,target_size):
        _,_,layer_height,layer_width = layer.size()
        diff_y = (layer_height - target_size[0])//2
        diff_x = (layer_width - target_size[1])//2
        return layer[:,:,diff_y:diff_y +target_size[0],diff_x :diff_x + target_size[1]]

    def forward(self,x,bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge,up.shape[2:])
        out = torch.cat([up,crop1],1)
        out = self.conv_block(out)
        return out


class UNet(nn.Module):
    def __init__(self,in_channels=1,
                 n_classs = 2,
                 depth =5,
                 wf =6,
                 padding = False,
                 batch_norm=False,
                 up_mode= 'upconv'):
        super(UNet,self).__init__()
        assert up_mode in ('upconv','upsample')
        self.padding = padding
        self.depth = depth

        prev_channels = in_channels

        self.down_path = nn.ModuleList()

        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels,2**(wf+i),padding,batch_norm))
            prev_channels = 2**(wf+i)


        self.up_path = nn.ModuleList()

        for i in reversed(range(depth -1)):
            self.up_path.append(UNetUpBlock(prev_channels,2**(wf+i),up_mode,padding,batch_norm))
            prev_channels = 2**(wf+i)

        self.last = nn.Conv2d(prev_channels,n_classs,kernel_size=1)

    def forward(self,x):
        blocks = []

        for i,down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) -1:
                blocks.append(x)
                x = F.max_pool2d(x,2)
        for i,up in enumerate(self.up_path):
            x = up(x,blocks[-i-1])
        return self.last(x)

# x = torch.randn((1,1,572,572))#batch_size 是1 ，通道是1，输入572 572
# unet = UNet()
# unet.eval()
# y_unet = unet(x)


class UNetEncode(nn.Module):
    def __init__(self,in_channel=1,depth=5,wf=6,padding=False,batch_norm=False):
        super(UNetEncode, self).__init__()
        self.padding = padding
        self.depth = depth
        prev_channels = in_channel

        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels,2**(wf+i),padding,batch_norm))
            prev_channels = 2**(wf + i)


    def forward(self,x):
        blocks = []
        for i ,down in enumerate(self.down_path):
            if i != len(self.down_path) -1:
                blocks.append(x)
                x = F.max_pool2d(x,2)
        blocks.append(x)

        return blocks

# class UNet(nn.Module):
#     def __init__(self,n_classed=2,depth=5,wf=6,padding=False,batch_norm=False,up_mode='upconv'):
#         super(UNet,self).__init__()

#         assert up_mode in ('upconv','upsample')
#         self.padding = padding
#         self.depth = depth

#         prev_channels = 2**(wf+ depth-1)

#         self.encode = UNetEncode()
#         self.up_path = nn.ModuleList()

#         for i in reversed(range(depth -1)):
#             self.up_path.append(UNetUpBlock(prev_channels,2**(wf+i),up_mode,padding,batch_norm))
#             prev_channels = 2**(wf +i)

#         self.last = nn.Conv2d(prev_channels,n_classed,kernel_size=1)


#     def forward(self,x):
#         blocks = self.encode(x)
#         x = blocks[-1]
#         for i ,up in enumerate(self.up_path):
#             x = up(x,blocks[-i-2])
#         return self.last(x)


# Resnet 101

class Block(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,padding=1,stride=1):
        super(Block,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=padding,stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)


    def forward(self,x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out

class ResBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,padding=1,stride=1):
        super(ResBlock, self).__init__()
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=padding,stride=stride)

    def forward(self,x):
        out = self.conv1(self.relu1(self.bn1(x)))
        return out


class Bottleneck(nn.Module):
    expansion =4
    def __init__(self,in_channels,out_channels,):
        super(Bottleneck, self).__init__()
        assert out_channels%4 ==0

        self.block1 = ResBlock(in_channels,int(out_channels/4),kernel_size=1,padding=0)
        self.block2 = ResBlock(int(out_channels/4),int(out_channels/4),kernel_size=3,padding=1)
        self.block3 = ResBlock(int(out_channels/4),out_channels,kernel_size=1,padding=0)

    def forward(self,x):
        identity =x
        out  = self.block1(x)
        out  = self.block2(out)
        out = self.block3(out)
        out +=identity
        return out

class DownBottleneck(nn.Module):
    expansion = 4
    def __init__(self,in_channel,out_channel,stride=2):
        super(DownBottleneck, self).__init__()
        assert out_channel %4 ==0

        self.block1 = ResBlock(in_channel,int(out_channel/4),kernel_size=1,padding=0)
        self.conv1 = nn.Conv2d(in_channel,out_channel,kernel_size=1,padding=0,stride=stride)

        self.block2 = ResBlock(int(out_channel/4),int(out_channel/4),kernel_size=3,padding=1)
        self.block3 = ResBlock(int(out_channel/4),out_channel,kernel_size=1,padding=0)


    def forward(self,x):
        identity = self.conv1(x)
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)

        out +=identity

        return out

def make_layers(in_channels,layer_list,name='vgg'):
    layers =[]
    if name == 'vgg':
        for v in layer_list:
            layers +=[Block(in_channels,v)]
            in_channels = v
    elif name == 'resnet':
        layers += [DownBottleneck(in_channels,layer_list[0])]
        in_channels = layer_list[0]

        for v in layer_list[1:]:
            layers += [Bottleneck(in_channels,v)]
            in_channels = v
    return nn.Sequential(*layers)

class Layer(nn.Module):
    def __init__(self,in_channels,layer_list,net_name):
        super(Layer,self).__init__()

        self.layer = make_layers(in_channels,layer_list,name= net_name)
    def forward(self,x):
        out = self.layer(x)
        return out

class ResNet101(nn.Module):

    def __init__(self):
        super(ResNet101, self).__init__()

        self.conv1 = Block(3,64,7,3,2)
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=True)

        self.conv2_1 = DownBottleneck(64,256,stride=1)
        self.conv2_2 = Bottleneck(256,256)
        self.conv2_3 = Bottleneck(256,256)

        self.layer3 = Layer(256,[512] * 2,'resnet')
        self.layer4 = Layer(512,[1024]*23,'resnet')
        self.layer5 = Layer(1024,[2048]*3,'resnet')

    def forward(self,x):
        f1 = self.conv1(x)
        f2 = self.conv2_3(self.conv2_2(self.conv2_3(self.pool1(f1))))

        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        f5 = self.layer5(f4)

        return [f2,f3,f4,f5]

class ResNetUNet(nn.Module):
    def __init__(self,n_classes=2,depth=5,wf=6,padding=1,batch_norm=False,up_mode='upconv'):
        super(ResNetUNet,self).__init__()
        assert up_mode in ('upconv','upsample')
        self.padding = padding 
        self.depth = depth 
        prev_channels = 2**(wf + depth)

        self.encode = ResNet101()

        self.up_path = nn.ModuleList()
        for i in reversed(range(2,depth)):
            self.up_path.append(
                UNetUpBlock(prev_channels,2**(wf + i),up_mode,padding,batch_norm)
            )
            prev_channels = 2**(wf + i)

        self.last = nn.Conv2d(prev_channels,n_classes,kernel_size=1)
    
    def forward(self,x):
        blocks = self.encode(x)
        x = blocks[-1]
        for i, up in enumerate(self.up_path):
            x = up(x,blocks[-i-2])

        return self.last(x)


x = torch.randn((1,3, 256,256))
unet = ResNetUNet()
unet.eval()
y_unet = unet(x)


