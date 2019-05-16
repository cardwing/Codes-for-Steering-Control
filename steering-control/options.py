import argparse
parser = argparse.ArgumentParser(description="Required parameters of steering control models")

parser.add_argument('--flag', default='test', type=str, help='flag indicating training or testing')
parser.add_argument('--path_pre_trained', default='/home/houyuenan/houyn/pre-trained-models/models/ResNet-L50', type=str, help='path to pre-trained model')
parser.add_argument('--path_trained', default='/home/houyuenan/houyn/3d_resnet_lstm/all', type=str, help='path to trained model')
parser.add_argument('--path_save', default='/home/houyuenan/houyn/3d_resnet_lstm/saved/all', type=str, help='path to save model')

