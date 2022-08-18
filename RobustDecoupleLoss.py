import torch
import torch.nn as nn

class RDL(nn.Module):
    """
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=3, knearest=4, use_gpu=True,output_device = 0,learnable=True):
        super(RDL, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.knearest = knearest
        self.output_device=output_device
        # if self.use_gpu:
        #     self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        # else:
        #     self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
        if learnable==True:
            if self.use_gpu:
                self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda(output_device))
            else:
                self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
        else:
            if self.use_gpu:
                self.centers = torch.autograd.Variable(torch.randn(self.num_classes, self.feat_dim).cuda(output_device),requires_grad=False)
            else:
                self.centers =torch.autograd.Variable(torch.randn(self.num_classes, self.feat_dim),requires_grad=False)


    def forward(self, x, labels, epoch=0):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """

        batch_size = x.size(0) # n * 10

        # print(torch.norm(x,dim=1))
        # print("label", labels.size())
        cur_centers = self.centers[labels]
        # print("cur_centers",  cur_centers.size())
        X_c = torch.sqrt(torch.pow(cur_centers, 2).sum(dim = 1)).cuda(self.output_device)
        # C_2 = torch.sqrt(torch.pow(self.centers, 2).sum(dim = 1)).cuda(self.output_device)  # C
        
        x_2 = torch.sqrt(torch.pow(x, 2).sum(dim = 1).cuda(self.output_device))  # N
        x_2L = torch.mul(x_2, torch.pow(X_c, -1))

        C_2 = torch.pow(X_c, -1)
        C_2 = torch.diag(C_2)

        x_2 = torch.pow(x_2, -1)
        x_2 = torch.diag(x_2)

        x = torch.mm(x.t(), x_2).t() #n * d      
        C = torch.mm(cur_centers.t(), C_2).t() # n * d

        dist = torch.mm(x, C.t())
        dist = 1 - dist

        # classes = torch.arange(self.num_classes).long()
        # if self.use_gpu: classes = classes.cuda()

        # labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        # mask = labels.eq(classes.expand(batch_size, self.num_classes))
        # dist = dist * mask.float()

        dist = torch.mul(dist, dist)

        dist_in = torch.diag(dist)
        dist_out = dist - dist_in
        dist_in = torch.mul(dist_in, dist_in)
        dist_out=torch.mul(dist_out,dist_out)
        lossA_in = dist_in.clamp(min=1e-12, max=1e+12).sum() / batch_size
        lossA_out = dist_out.clamp(min=1e-12, max=1e+12).sum() / (batch_size * (batch_size - 1))
        #print(x)
        # print("size", x.size(), x_2.size())
        # print(torch.mul(x_2, torch.pow(X_c, -1)).size())
        dist_L = torch.pow(1 - x_2L, 2)
        lossL = dist_L.clamp(min=1e-12, max=1e+12).sum() / batch_size
        # loss=lossA_in - lossA_out + lossL*0.01
        loss =  lossA_in- lossA_out*0.01+lossL*0.01
        # torch.save(self.centers, './test_center.pt')
        # exit(0)
        # print(torch.norm(self.centers,dim=1))

        return loss
