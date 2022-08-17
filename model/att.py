import torch
from torch import nn

class ST_Joint_Att(nn.Module):
    def __init__(self, channel, reduct_ratio, bias, **kwargs):
        super(ST_Joint_Att, self).__init__()

        inner_channel = channel // reduct_ratio

        self.fcn = nn.Sequential(
            nn.Conv2d(channel, inner_channel, kernel_size=1, bias=bias),
            nn.BatchNorm2d(inner_channel),
            # nn.Hardswish(),
            # nn.Sigmoid(),
            HardSwish(),
        )
        self.conv_t = nn.Conv2d(inner_channel, channel, kernel_size=1)
        self.conv_v = nn.Conv2d(inner_channel, channel, kernel_size=1)

    def forward(self, x):
        N, C, T, V = x.size()
        x_t = x.mean(3, keepdims=True)
        x_v = x.mean(2, keepdims=True).transpose(2, 3)
        x_att = self.fcn(torch.cat([x_t, x_v], dim=2))
        x_t, x_v = torch.split(x_att, [T, V], dim=2)
        x_t_att = self.conv_t(x_t).sigmoid()
        x_v_att = self.conv_v(x_v.transpose(2, 3)).sigmoid()
        # x_att1 = x_t_att - x_v_att
        # x_att2 = x_v_att - x_t_att
        # x_little_att = x_t_att * x_v_att
        # x_att = x_v_att - x_t_att
        # x_att = x_t_att * x_v_att + x_v_att * x_v_att
        x_att = x_t_att * x_v_att
        # print(x_t_att)
        return x_att


class ST_Joint_Att33(nn.Module):
    def __init__(self, channel, reduct_ratio, bias, **kwargs):
        super(ST_Joint_Att33, self).__init__()

        inner_channel = channel // reduct_ratio

        self.fcn = nn.Sequential(
            nn.Conv2d(channel, inner_channel, kernel_size=1, bias=bias),
            nn.BatchNorm2d(inner_channel),
            # nn.Hardswish(),
            # nn.Sigmoid(),
            HardSwish(),
        )
        self.conv_t = nn.Conv2d(inner_channel, channel, kernel_size=1)
        self.conv_v = nn.Conv2d(inner_channel, channel, kernel_size=1)

        self.fcn1 = nn.Sequential(
            nn.Conv2d(25, 25, kernel_size=1, bias=bias),
            nn.BatchNorm2d(25),
            # nn.Hardswish(),
            # nn.Sigmoid(),
            HardSwish(),
        )
        self.conv_t1 = nn.Conv2d(25, 25, kernel_size=1)
        self.conv_v1 = nn.Conv2d(25, 25, kernel_size=1)

        # self.fcn2 = nn.Sequential(
        #     nn.Conv2d(channel, inner_channel, kernel_size=1, bias=bias),
        #     nn.BatchNorm2d(inner_channel),
        #     # nn.Hardswish(),
        #     # nn.Sigmoid(),
        #     HardSwish(),
        # )
        # self.conv_t2 = nn.Conv2d(inner_channel, channel, kernel_size=1)
        # self.conv_v2 = nn.Conv2d(inner_channel, channel, kernel_size=1)

    def forward(self, x):
        N, C, T, V = x.size()
        x_t = x.mean(3, keepdims=True)
        x_v = x.mean(2, keepdims=True).transpose(2, 3)
        x_att = self.fcn(torch.cat([x_t, x_v], dim=2))
        x_t, x_v = torch.split(x_att, [T, V], dim=2)
        x_t_att = self.conv_t(x_t).sigmoid()
        x_v_att = self.conv_v(x_v.transpose(2, 3)).sigmoid()
        x_att1 = x_t_att * x_v_att

        # print(x_att1.size())
        x_c = x.mean(1, keepdims=True).permute(0, 3, 2, 1).contiguous() # N V T 1
        x_t = x.mean(2, keepdims=True).permute(0, 3, 1, 2).contiguous() # N V C 1
        x_att = self.fcn1(torch.cat([x_c, x_t], dim=2))
        
        x_c, x_t = torch.split(x_att, [T, C], dim=2)
        x_t_att = self.conv_t1(x_c).sigmoid()
        x_C_att = self.conv_v1(x_t.transpose(2, 3)).sigmoid()
        x_att2 = x_t_att * x_C_att
        x_att2 = x_att2.permute(0, 3, 2, 1).contiguous()
        # print(x_att2.size())
        # x_c = x.mean(1, keepdims=True).permute(0, 2, 3, 1).contiguous() # N T V 1
        # x_v = x.mean(3, keepdims=True).permute(0, 2, 1, 3).contiguous() # N T C 1
        # x_att = self.fcn2(torch.cat([x_c, x_v], dim=2))
        # x_c, x_v = torch.split(x_att, [V, C], dim=2)
        # x_t_att = self.conv_t2(x_c).sigmoid()
        # x_C_att = self.conv_v2(x_v.transpose(2, 3)).sigmoid()
        # x_att3 = x_t_att * x_C_att
        # x_att3 = x_att3.permute(0, 3, 1, 2).contiguous()

        x_att = (x_att1 + x_att2) / 2
        return x_att

class ST_Joint_Att_3channel(nn.Module):
    def __init__(self, channel, reduct_ratio, bias, **kwargs):
        super(ST_Joint_Att_3channel, self).__init__()

        inner_channel = channel // reduct_ratio

        self.fcn = nn.Sequential(
            nn.Conv2d(inner_channel, channel, kernel_size=1, bias=bias),
            nn.BatchNorm2d(channel),
            # nn.Hardswish(),
            # nn.Sigmoid(),
            HardSwish(),
        )

        # self.fcn1 = nn.Sequential(
        #     nn.Conv2d(channel, inner_channel, kernel_size=1, bias=bias),
        #     nn.BatchNorm2d(inner_channel),
        #     # nn.Hardswish(),
        #     # nn.Sigmoid(),
        #     HardSwish(),
        # )

        # self.fcn2 = nn.Sequential(
        #     nn.Conv2d(channel, inner_channel, kernel_size=1, bias=bias),
        #     nn.BatchNorm2d(inner_channel),
        #     # nn.Hardswish(),
        #     # nn.Sigmoid(),
        #     HardSwish(),
        # )

        self.conv_t = nn.Conv2d(channel, inner_channel, kernel_size=1)
        self.conv_v = nn.Conv2d(channel, inner_channel, kernel_size=1)

    def forward(self, x):
        N, C, T, V = x.size()
        x_t = x.mean(3, keepdims=True)
        x_v = x.mean(2, keepdims=True)
        # x_att = self.fcn(torch.cat([x_t, x_v], dim=2))
        x_att1 = self.conv_t(x_t).sigmoid()
        x_att2 = self.conv_v(x_v).sigmoid()
        x_att = x_att1 * x_att2
        x_att = self.fcn(x_att)
        # x_t, x_v = torch.split(x_att, [T, V], dim=2)
        # x_t_att = self.conv_t(x_att1).sigmoid()
        # x_v_att = self.conv_v(x_att2.transpose(2, 3)).sigmoid()
        # x_att1 = x_t_att - x_v_att
        # x_att2 = x_v_att - x_t_att
        # x_little_att = x_t_att * x_v_att
        # x_att = x_v_att - x_t_att
        # x_att = x_t_att * x_v_att + x_v_att * x_v_att
        # x_att = x_t_att * x_v_att
        # print(x_t_att)
        return x_att

class ST_Joint_Att_chengfa(nn.Module):
    def __init__(self, channel, reduct_ratio, bias, **kwargs):
        super(ST_Joint_Att_chengfa, self).__init__()

        inner_channel = channel // reduct_ratio
        # 均值增强
        self.fcn = nn.Sequential(
            nn.Conv2d(inner_channel, channel, kernel_size=1, bias=bias),
            nn.BatchNorm2d(channel),
            # nn.Hardswish(),
            # nn.Sigmoid(),
            HardSwish(),
        )
        # self.conv_t = nn.Conv2d(inner_channel, channel, kernel_size=1)
        self.conv1 = nn.Conv2d(channel, inner_channel, kernel_size=1)
        self.conv2 = nn.Conv2d(channel, inner_channel, kernel_size=1)


    def forward(self, x):
        N, C, T, V = x.size()
        # 均值增强
        # x_v = x.mean(2, keepdims=True)
        # x_t = x.mean(3, keepdims=True)
        # x_att_v = x_v * x # / x_v
        # x_att_v_t = x.mean(3, keepdims=True)

        # x_att_t = x_t * x
        # x_att_t_v = x.mean(2, keepdims=True).transpose(2, 3)

        # x_att = self.fcn(torch.cat([x_att_v_t, x_att_t_v], dim=2))
        # x_t2, x_v2 = torch.split(x_att, [T, V], dim=2)
        # x_t_att2 = self.conv_t(x_t2).sigmoid()
        # x_v_att2 = self.conv_v(x_v2.transpose(2, 3)).sigmoid()

        # x_att3 = x_t_att2 * x_v_att2
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_att = x_1 - x_2
        x_att = self.fcn(x_att)

        # x_att3 = x_v_att * x_t_att
        return x_att
        # x_choose = x - x_v
        # x_chengfa = (x + x_v) / (2 * x) 
        # x_choose[x_choose > 0] = 1
        # x_choose[x_choose <= 0] = 0
        # x_chengfa = x_chengfa * x_choose
        # print(x_chengfa.size())

        # x_choose = x_choose - 1
        # x_choose[x_choose < 0] = 1
        # x_choose = x_choose + x_chengfa
        # x_choose = self.fcn(x_choose)
        # return self.conv(x_choose)

class ST_Joint_Att_pusu(nn.Module):
    def __init__(self, channel, reduct_ratio, bias, **kwargs):
        super(ST_Joint_Att_pusu, self).__init__()

        inner_channel = channel // reduct_ratio

        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(channel, inner_channel, kernel_size=1, bias=bias),
        #     nn.BatchNorm2d(inner_channel),
        #     # nn.Hardswish(),
        #     # nn.Sigmoid(),
        #     HardSwish(),
        # )
        self.conv1 = nn.Conv2d(channel, inner_channel, kernel_size=1, bias=bias),
        self.conv2 = nn.Conv2d(channel, inner_channel, kernel_size=1, bias=bias),
        # self.conv3 = nn.Conv2d(channel, inner_channel, kernel_size=1)

        self.conv_t = nn.Conv2d(inner_channel, channel, kernel_size=1)
        self.conv_v = nn.Conv2d(inner_channel, channel, kernel_size=1)
        self.tanh = nn.Tanh()

    # def forward(self, x):
    #     N, C, T, V = x.size()
    #     x_t = x.mean(3, keepdims=True)
    #     x_v = x.mean(2, keepdims=True).transpose(2, 3)
    #     x = torch.cat([x_t, x_v], dim=2)
    #     x_mix1 = self.conv1(x)
    #     x_mix2 = self.conv2(x).transpose(2, 3)
    #     x_mix = self.tanh(x_mix1 - x_mix2)
    #     x = self.conv3(x)
    #     x = torch.einsum('ncuv,nctv->nctu', x_mix, x.transpose(2, 3)).transpose(2, 3)
    #     x_t, x_v = torch.split(x, [T, V], dim=2)
    #     x_t_att = self.conv_t(x_t).sigmoid()
    #     x_v_att = self.conv_v(x_v.transpose(2, 3)).sigmoid()
    #     x_att = x_t_att * x_v_att
    #     return x_att

    def forward(self, x):
        N, C, T, V = x.size()
        x_t = x.mean(3, keepdims=True)
        x_v = x.mean(2, keepdims=True).transpose(2, 3)
        x = torch.cat([x_t, x_v], dim=2)
        x_mix1 = self.conv1(x)
        x_mix2 = self.conv2(x).transpose(2, 3)
        x_mix = self.tanh(x_mix1 - x_mix2)
        x = x_mix.mean(3, keepdims=True)

        x_t, x_v = torch.split(x, [T, V], dim=2)
        x_t_att = self.conv_t(x_t).sigmoid()
        x_v_att = self.conv_v(x_v.transpose(2, 3)).sigmoid()
        x_att = x_t_att * x_v_att
        return x_att

class Channel_Att(nn.Module):
    def __init__(self, channel, **kwargs):
        super(Channel_Att, self).__init__()

        self.fcn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel//4, kernel_size=1),
            nn.BatchNorm2d(channel//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//4, channel, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.fcn(x)

class Channel_tv(nn.Module):
    def __init__(self, channel, **kwargs):
        super(Channel_tv, self).__init__()

        self.fcn1 = nn.Sequential(
            nn.Conv2d(channel, channel//4, kernel_size=1),
            nn.BatchNorm2d(channel//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//4, channel, kernel_size=1),
            nn.Sigmoid(),
        )

        self.fcn2 = nn.Sequential(
            nn.Conv2d(channel, channel//4, kernel_size=1),
            nn.BatchNorm2d(channel//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//4, channel, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_1 = x.mean(2, keepdims=True)
        x_2 = x.mean(2, keepdims=True).transpose(2, 3)

        x_1 = self.fcn1(x_1)
        x_2 = self.fcn2(x_2)

        x_att = torch.matmul(x_1, x_2)
        return x

class Channel_Att_me(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(Channel_Att_me, self).__init__()
        # self.inter_channel = in_channels // 4
        self.in_channels = in_channels
        self.conv_a = nn.Conv2d(in_channels, in_channels, 1)
        self.conv_b = nn.Conv2d(in_channels, in_channels, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels, 1)
        self.soft = nn.Softmax(-2)  # softmax倒数第二个维度进行softmax
        self.relu = nn.ReLU()   
    def forward(self, x):
        # print("ru", x.size())
        N, C, T, V = x.size()
        x_1 = self.conv_a(x).view(N, self.in_channels, T*V)
        x_2 = self.conv_b(x).view(N, self.in_channels, T*V).permute(0, 2, 1)
        att_c = self.soft(torch.matmul(x_1, x_2) / x_1.size(-1)) # N C C
        # att_c = self.soft(torch.matmul(x_1, x_2)) # N C C
        x = x.permute(0, 2, 3, 1).view(N, T*V, C)
        x = self.conv_c(torch.matmul(x, att_c).permute(0, 2, 1).view(N, C, T, V))
        # print("wan", x.size())
        return self.relu(x)
        # return x

class HardSwish(nn.Module):
    def __init__(self, inplace=False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        inner = nn.functional.relu6(x + 3.).div_(6.)
        return x.mul_(inner) if self.inplace else x.mul(inner)