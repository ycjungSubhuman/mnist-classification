'''
Training/Testing configuration
'''
import sys

default_config = {
    'learning-rate': 0.001,
    'batch-size': 32,
    'max-epoch': 100,
    'checkpoints': 'checkpoints/',
    'model': {
        #('layer name', layer arguments, expected input dim, expected output dim)
        'feature-extraction': [
            ('conv-relu-bn', [1, 32, 5, 1, 2], [1, 28, 28], [32, 28, 28]),
            ('conv-relu-bn', [32, 64, 3, 1, 1], [32, 28, 28], [64, 28, 28]), 
            ('maxpool', [2], [64, 28, 28], [64, 14, 14]),
            ('conv-relu-bn', [64, 128, 4, 2, 2], [64, 14, 14], [128, 8, 8]),
            ('maxpool', [2], [128, 8, 8], [128, 4, 4]),
            ('conv-relu-bn', [128, 256, 4, 1, 0], [128, 4, 4], [256, 1, 1]),
        ],
        'classification': [
            ('fc-relu', [256, 128], [256], [128]),
            ('fc-softmax', [128, 10], [128], [10]), # outputs one-hot vector
            ]
    },
    }

# TODO: provide a way to override default config 
def get():
    return default_config
