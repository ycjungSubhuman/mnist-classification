'''
Main training code
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import test
import train_test_tool as tool
from torch.utils.data.dataloader import DataLoader

if '__main__' == __name__:
    data_train, data_test = tool.get_mnist()

    suite = tool.build_train_suite()
    conf = suite['conf']
    classifier = suite['classifier']
    loss = suite['loss']
    optim = suite['optim']

    var_map = {}
    test_accuracy = []
    loss_progress = []
    data_train = DataLoader(data_train, batch_size=conf['batch-size'])

    for epoch in range(conf['max-epoch']):
        for i, (image, target) in enumerate(data_train):

            # update network paramters
            def forward_backward():
                optim.zero_grad()
                c = classifier(image.cuda())
                l = loss(c, target.cuda())
                l.backward()
                optim.step()
                loss_progress.append(l.item())

            def build_var_map():
                var_map = {}
                var_map['classifier'] = classifier.state_dict()
                var_map['optim'] = optim.state_dict()
                var_map['epoch'] = epoch
                var_map['i'] = i
                var_map['loss_progress'] = loss_progress
                var_map['test_accuracy'] = test_accuracy

                return var_map

            # callback every n iteration
            def handle_callback_every_n(n, func, *args):
                if (i % n) == (n - 1):
                    func(*args)

            def print_running_loss(ran):
                s = sum(loss_progress[-1-ran:])
                avg = s / ran
                print('epoch: {} / i: {} / avg_loss: {}'.format(epoch, i, avg))

            forward_backward()
            handle_callback_every_n(50, print_running_loss, 50)
            var_map = build_var_map()
            handle_callback_every_n(500, tool.save_checkpoint, conf, var_map)

        print ('epoch {} fisnished'.format(epoch))
        var_map = build_var_map()
        tool.save_checkpoint(conf, var_map)
        test_accuracy.append(test.test(data_test))

            
                
    
