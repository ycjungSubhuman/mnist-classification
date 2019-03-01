'''
Model definition
'''
import torchvision as vision
import torch.nn as nn
import torch.nn.functional as F
import util

# convert module kind to nn.module instance
def get_module(kind):
    if 'conv-relu-bn' == kind:
        def conv_relu_bn(*args):
            conv = nn.Conv2d(*args)
            relu = nn.modules.activation.ReLU()
            bn = nn.BatchNorm2d(args[1])
            return nn.Sequential(conv, relu, bn)
        return conv_relu_bn

    elif 'maxpool' == kind:
        def maxpool(*args):
            return nn.MaxPool2d(*args)
        return maxpool

    elif 'fc-relu' == kind:
        def fc(*args):
            lin = nn.Linear(*args)
            relu = nn.modules.activation.ReLU()
            return nn.Sequential(lin, relu)
        return fc
    elif 'fc-softmax' == kind:
        def fc(*args):
            lin = nn.Linear(*args)
            sm = nn.modules.activation.LogSoftmax(dim=1)
            return nn.Sequential(lin, sm)
        return fc

    else:
        raise RuntimeException('Not a valid module name : {}'.format(kind))

# Gets 64 by 64 images. outputs (10)-sized one-hot vector
class MnistClassifier(nn.Module):
    def __init__(self, mod_opt):
        super().__init__()
        self.mod_opt = mod_opt

        f_opt = mod_opt['feature-extraction']
        c_opt = mod_opt['classification']

        def process_opt(li, opt):
            for i, (kind, params, expect_in, expect_out) in enumerate(opt):
                m = get_module(kind)(*params)
                util.check_in_out_dim(m, expect_in, expect_out)
                li.append(m)
                self.add_module('{}-{}'.format(i,kind), m)

        # build up feature extractor
        self.f = []
        process_opt(self.f, f_opt)

        # build up classifier
        self.c = []
        process_opt(self.c, c_opt)

        # build sequential modules
        self.f_seq = nn.Sequential(*self.f)
        self.c_seq = nn.Sequential(*self.c)

    def forward(self, in_data):
        f = self.f_seq(in_data)
        assert(4 == f.dim())
        assert(1 == f.size()[2] and 1 == f.size()[3])

        # rotate vector from (N x C x 1 x 1) to (N x C)
        f_dim1 = f.squeeze() 
        assert(2 == f_dim1.dim())

        c = self.c_seq(f_dim1)
        assert(2 == c.dim())
        assert(10 == c.size()[1])
        return c
