import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
import math

if __name__ == "__main__":
    import common
else:
    from models import common

class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True))
        
        self.conv_residual = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        x = self.conv(x) + self.conv_residual(x)
        return x

class U_Net(nn.Module):
    def __init__(self, in_ch=4, out_ch=3, ps=True):
        super(U_Net, self).__init__()

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.ps = ps
        if not self.ps:
            out_ch = 4
        
        self.Down1 = nn.Conv2d(filters[0], filters[0], kernel_size=4, stride=2, padding=1, bias=True)
        self.Down2 = nn.Conv2d(filters[1], filters[1], kernel_size=4, stride=2, padding=1, bias=True)
        self.Down3 = nn.Conv2d(filters[2], filters[2], kernel_size=4, stride=2, padding=1, bias=True)
        self.Down4 = nn.Conv2d(filters[3], filters[3], kernel_size=4, stride=2, padding=1, bias=True)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=2, stride=2, padding=0, bias=True)
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2, padding=0, bias=True)
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2, padding=0, bias=True)
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2, padding=0, bias=True)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        if self.ps:
            self.Conv = nn.Conv2d(filters[0], out_ch*4, kernel_size=1, stride=1, padding=0)
            self.Up = nn.PixelShuffle(2)
        else:
            self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Down1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Down2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Down3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Down4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        if self.ps:
            out = self.Conv(d2)
            out = self.Up(out)
        else:
            out = self.Conv(d2)

        #d1 = self.active(out)

        return out

class Concat(nn.Module):
    def __init__(self, x_ch, y_ch):
        super(Concat, self).__init__()
        self.conv = nn.Conv2d(x_ch+y_ch, x_ch, 1, 1, 0)
    
    def forward(self, x, y):
        out = torch.cat((x, y), dim=1)
        out = self.conv(out)
        return out

class LocalAttention(nn.Module):
    def __init__(self, x_ch, y_ch, feat, kernel_size=5):
        super(LocalAttention, self).__init__()
        self.kernel_size = kernel_size

        self.unfold = nn.Unfold(kernel_size=[kernel_size, kernel_size], padding=kernel_size//2, stride=1)
        self.softmax = nn.Softmax(dim=1)

        self.conv_x = nn.Conv2d(x_ch, feat, 1, 1, 0)
        self.conv_y = nn.Conv2d(y_ch, feat, 1, 1, 0)

        self.conv = nn.Conv2d(feat, x_ch, 1, 1, 0)

    def forward(self, x, y):
        batch, channel, height, width = y.size()

        K = self.unfold(self.conv_y(y)).view(batch, -1, self.kernel_size**2, height, width)
        Q = self.conv_y(y).view(batch, -1, 1, height, width)
        V = self.unfold(self.conv_x(x)).view(batch, -1, self.kernel_size**2, height, width)

        A = torch.einsum('bckhw,bcqhw -> bkqhw', K, Q)
        # A = self.softmax(A/math.sqrt(channel))
        A = self.softmax(A)

        out = torch.einsum('bkqhw,bckhw -> bcqhw', A, V).view(batch, -1, height, width)

        out = self.conv(out)

        return out+x
        
class conv_block_G(nn.Module):
    def __init__(self, in_ch, out_ch, in_ch_G, out_ch_G, guidance=True):
        super(conv_block_G, self).__init__()
        self.guidance = guidance
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True))
        
        self.conv_G = nn.Sequential(
            nn.Conv2d(in_ch_G, out_ch_G, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(out_ch_G, out_ch_G, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True))
        
        self.conv_residual = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=True)
        self.conv_residual_G = nn.Conv2d(in_ch_G, out_ch_G, kernel_size=1, stride=1, bias=True)

        if guidance:
            self.guidance = LocalAttention(out_ch, out_ch_G, out_ch_G, kernel_size=5)
            # self.guidance = Concat(out_ch, out_ch_G)

    def forward(self, x, y):
        x = self.conv(x) + self.conv_residual(x)
        y = self.conv_G(y) + self.conv_residual_G(y)
        if self.guidance:
            x = self.guidance(x, y)
        return x, y

class Down(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, in_ch_G=1, out_ch_G=1, kernel_size=4, stride=2):
        super(Down, self).__init__()
        self.conv_G = nn.Conv2d(in_ch_G, out_ch_G, kernel_size=kernel_size, stride=stride, padding=(kernel_size-stride)//2, bias=True)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=(kernel_size-stride)//2, bias=True)
    
    def forward(self, x, y):
        x = self.conv(x)
        y = self.conv_G(y)
        return x, y

class Up(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, in_ch_G=1, out_ch_G=1, kernel_size=2, stride=1):
        super(Up, self).__init__()
        # self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv_G = nn.ConvTranspose2d(in_ch_G, out_ch_G, kernel_size=kernel_size, stride=stride, padding=(kernel_size-stride)//2, bias=True)
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=(kernel_size-stride)//2, bias=True)
    
    def forward(self, x, y):
        x = self.conv(x)
        y = self.conv_G(y)
        return x, y

class U_Net_G(nn.Module):
    def __init__(self, in_ch=4, out_ch=3, ps=True, pre=False, guidance='R'):
        super(U_Net_G, self).__init__()

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        filters_G = [n1//2, n1, n1 * 2, n1 * 4, n1 * 8]
        self.ps = ps
        self.pre = pre
        self.guidance = guidance
        if not self.ps:
            out_ch = 4
        if self.pre:
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        
        self.Down1 = Down(filters[0], filters[0], filters_G[0], filters_G[0], kernel_size=4, stride=2)
        self.Down2 = Down(filters[1], filters[1], filters_G[1], filters_G[1], kernel_size=4, stride=2)
        self.Down3 = Down(filters[2], filters[2], filters_G[2], filters_G[2], kernel_size=4, stride=2)
        self.Down4 = Down(filters[3], filters[3], filters_G[3], filters_G[3], kernel_size=4, stride=2)

        self.Up5 = Up(filters[4], filters[3], filters_G[4], filters_G[3], kernel_size=2, stride=2)
        self.Up4 = Up(filters[3], filters[2], filters_G[3], filters_G[2], kernel_size=2, stride=2)
        self.Up3 = Up(filters[2], filters[1], filters_G[2], filters_G[1], kernel_size=2, stride=2)
        self.Up2 = Up(filters[1], filters[0], filters_G[1], filters_G[0], kernel_size=2, stride=2)

        if guidance == 'G':
            self.Conv1 = conv_block_G(in_ch, filters[0], 2, filters_G[0])
        else:
            self.Conv1 = conv_block_G(in_ch, filters[0], 1, filters_G[0])
        self.Conv2 = conv_block_G(filters[0], filters[1], filters_G[0], filters_G[1])
        self.Conv3 = conv_block_G(filters[1], filters[2], filters_G[1], filters_G[2])
        self.Conv4 = conv_block_G(filters[2], filters[3], filters_G[2], filters_G[3])
        self.Conv5 = conv_block_G(filters[3], filters[4], filters_G[3], filters_G[4])

        self.Up_conv5 = conv_block_G(filters[4], filters[3], filters_G[4], filters_G[3])
        self.Up_conv4 = conv_block_G(filters[3], filters[2], filters_G[3], filters_G[2])
        self.Up_conv3 = conv_block_G(filters[2], filters[1], filters_G[2], filters_G[1])
        self.Up_conv2 = conv_block_G(filters[1], filters[0], filters_G[1], filters_G[0], guidance=True)

        if self.ps and not self.pre:
            self.Conv = nn.Conv2d(filters[0], out_ch*4, kernel_size=1, stride=1, padding=0)
            self.Conv_G = nn.Conv2d(filters_G[0], out_ch//2*4, kernel_size=1, stride=1, padding=0)
            self.Up = nn.PixelShuffle(2)
        else:
            self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
            if guidance == 'G':
                self.Conv_G = nn.Conv2d(filters_G[0], 2, kernel_size=1, stride=1, padding=0)
            else:
                self.Conv_G = nn.Conv2d(filters_G[0], 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x, y=None):
        if y is None:
            if self.guidance == 'G':
                y = torch.cat((x[:,1:2], x[:,3:]), dim=1)
            elif self.guidance == 'R':
                y = x[:,:1]
            elif self.guidance == 'B':
                y = x[:,2:3]
            elif self.guidance == 'W':
                y = x[:,3:]
        
        if self.ps and self.pre:
            x = self.upsample(x)
            y = self.upsample(y)

        e1, ge1 = self.Conv1(x, y)

        e2, ge2 = self.Down1(e1, ge1)
        e2, ge2 = self.Conv2(e2, ge2)

        e3, ge3 = self.Down2(e2, ge2)
        e3, ge3 = self.Conv3(e3, ge3)

        e4, ge4 = self.Down3(e3, ge3)
        e4, ge4 = self.Conv4(e4, ge4)

        e5, ge5 = self.Down4(e4, ge4)
        e5, ge5 = self.Conv5(e5, ge5)

        d5, gd5 = self.Up5(e5, ge5)
        d5, gd5 = torch.cat((e4, d5), dim=1), torch.cat((ge4, gd5), dim=1)
        d5, gd5 = self.Up_conv5(d5, gd5)

        d4, gd4 = self.Up4(d5, gd5)
        d4, gd4 = torch.cat((e3, d4), dim=1), torch.cat((ge3, gd4), dim=1)
        d4, gd4 = self.Up_conv4(d4, gd4)

        d3, gd3 = self.Up3(d4, gd4)
        d3, gd3 = torch.cat((e2, d3), dim=1), torch.cat((ge2, gd3), dim=1)
        d3, gd3 = self.Up_conv3(d3, gd3)

        d2, gd2 = self.Up2(d3, gd3)
        d2, gd2 = torch.cat((e1, d2), dim=1), torch.cat((ge1, gd2), dim=1)
        d2, gd2 = self.Up_conv2(d2, gd2)

        if self.ps and not self.pre:
            out = self.Conv(d2)
            out = self.Up(out)
            out_G = self.Conv_G(gd2)
            out_G = self.Up(out_G)
        else:
            out = self.Conv(d2)
            out_G = self.Conv_G(gd2)

        #d1 = self.active(out)

        return out, out_G

if __name__ == "__main__":
    x = torch.ones((1,3,256*scale,256*scale)).cuda()
    
    import pdb; pdb.set_trace()