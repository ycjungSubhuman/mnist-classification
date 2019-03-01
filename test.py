'''
Test existing network
'''
import torch
import train_test_tool as tool
import config
from torch.utils.data.dataloader import DataLoader

def test(data_test):
    suite = tool.build_test_suite()
    tool.load_checkpoint_test(suite['conf'], suite)
    data_test = DataLoader(data_test, batch_size=suite['conf']['batch-size'])
    
    classifier = suite['classifier']
    tot = 0
    correct = 0
    for i, (image, target) in enumerate(data_test):
        est = classifier(image.cuda()) # (N x 10)
        max_values, est_labels = torch.max(est, 1)
        tot += target.size()[0]
        correct += (target.cuda() == est_labels).sum().item()

    print ('Accuracy : {}. {}/{}'.format(correct/tot, correct, tot))
    return correct/tot

if '__main__' == __name__:
    data_train, data_test = tool.get_mnist()
    test(data_test)

