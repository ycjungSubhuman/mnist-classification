import torch
import torch.nn as nn
import torchvision as vision
import os
import config
from model import MnistClassifier

'''
Get torchvision.transforms for MSNIT raw image data
'''
def get_mnist_transform():
    return vision.transforms.Compose([
        vision.transforms.ToTensor(),
        vision.transforms.Normalize((0.1307,), (0.3081,)),
        ])
'''
Download and get MNIST dataset
'''
def get_mnist():
    train = vision.datasets.MNIST(
            'dataset/', download=True, transform=get_mnist_transform())
    test = vision.datasets.MNIST(
            'dataset/', download=True, transform=get_mnist_transform())

    return (train, test)
'''
Get loss module according to current configuration
'''
def get_loss(conf):
    return nn.NLLLoss()

'''
Get optimizer builder according to current configuration

An optimizer builder takes one argument
(pytorch tensor parameters. i.e. *.parameters()) and 
returns pytorch optimizer object
'''
def get_optim_builder(conf):
    def build_optim(params):
        return torch.optim.Adam(params, conf['learning-rate'])
    return build_optim

'''
Get objects for training (training suite)

(conf : dict, classifier : nn.Module, loss : nn.Module, optim : torch.optim.*)
'''
def build_train_suite():
    conf = config.get()
    classifier = MnistClassifier(conf['model']).cuda()
    loss = get_loss(conf).cuda()
    optim = get_optim_builder(conf)(classifier.parameters())

    return {
        'conf': conf,
        'classifier': classifier,
        'loss': loss,
        'optim': optim,}

'''
Get objects for testing (test suite)

(conf : dict, classifier : nn.Module)
'''
def build_test_suite():
    conf = config.get()
    classifier = MnistClassifier(conf['model']).cuda()
    return { 'conf': conf, 'classifier': classifier }

def get_checkpoint_name(var_map):
    return '{}00{}.chk'.format(var_map['epoch'], var_map['i'])

def save_checkpoint(conf, var_map):
    if not os.path.exists(conf['checkpoints']):
        os.makedirs(conf['checkpoints'])
    torch.save(var_map, 
            os.path.join(conf['checkpoints'], get_checkpoint_name(var_map)))

'''
Get latest checkpoint file
'''
def get_latest_chk(conf):
    li = [d for d in os.listdir(conf['checkpoints']) if d.endswith('.chk')]
    return os.path.join(
            conf['checkpoints'],
            sorted(li, key=lambda d: int(os.path.splitext(d)[0]))[-1])

'''
Load checkpoint data into training suite
'''
def load_checkpoint_train(conf, suite):
    db = torch.load(get_latest_chk(conf))
    suite['classifier'].load_state_dict(db['classifier'])
    suite['loss'].load_state_dict(db['loss'])
    suite['optim'].load_state_dict(db['optim'])


'''
Load checkpoint data into test suite
'''
def load_checkpoint_test(conf, suite):
    db = torch.load(get_latest_chk(conf))
    suite['classifier'].load_state_dict(db['classifier'])

