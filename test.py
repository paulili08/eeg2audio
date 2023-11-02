import torch
from torchvision import transforms
from torch.autograd import Variable
from dataset import DatasetFromFolder, EEGDataset
from GAN_model import Generator
from sklearn.model_selection import KFold
import numpy as np
import utils
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='eeg2audio', help='input dataset')#########
parser.add_argument('--num_batch', type=int, default=468, help='num of batches whole dataset divided')
parser.add_argument('--batch_size', type=int, default=1, help='test batch size')
parser.add_argument('--test_size', type=float, default=0.1, help='Proportion of data for testing')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--num_resnet', type=int, default=9, help='number of resnet blocks in generator')
parser.add_argument('--input_size', type=int, default=256, help='input size')
parser.add_argument('--nfolds', type=int, default=12, help='K-fold')
params = parser.parse_args()
print(params)

# Directories for loading data and saving results
feat_path = r'./features'
pts = ['sub-%02d'%i for i in range(1,11)]
data_dir = '../Data/' + params.dataset + '/'
save_dir = params.dataset + '_test_results/'
model_dir = params.dataset + '_model/'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

# K-fold
kf = KFold(params.nfolds, shuffle=False)
"""
# Data pre-processing
transform = transforms.Compose([transforms.Scale(params.input_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
"""
for pNr, pt in enumerate(pts):
    #Load the data
    train_data_U = np.load(os.path.join(feat_path, f'{pt}_feat.npy')) #eeg
    train_data_V = np.load(os.path.join(feat_path, f'{pt}_spec.npy')) #audio
    train_data_U = np.array_split(train_data_U, params.num_batch, axis=0)
    train_data_V = np.array_split(train_data_V, params.num_batch, axis=0)

    # Test data
    test_data_U = train_data_U[:int(params.num_batch * params.test_size)]
    test_data_U = EEGDataset(test_data_U)
    test_data_U = torch.utils.data.DataLoader(dataset=test_data_U,
                                               batch_size=params.batch_size,
                                               shuffle=True)
    test_data_V = train_data_V[:int(params.num_batch * params.test_size)]
    test_data_V = EEGDataset(test_data_V)
    test_data_V = torch.utils.data.DataLoader(dataset=test_data_V,
                                               batch_size=params.batch_size,
                                               shuffle=True)

    # Load model
    sub_model_dir = model_dir + f'{pt}'
    sub_save_dir = save_dir + f'{pt}'
    G_A = Generator(1, params.ngf, 1)
    G_B = Generator(1, params.ngf, 1)
    G_C = Generator(1, params.ngf, 1)
    G_D = Generator(1, params.ngf, 1)
    G_A.cuda()
    G_B.cuda()
    G_C.cuda()
    G_D.cuda()
    G_A.load_state_dict(torch.load(sub_model_dir + f'{pt}generator_A_param.pkl'))
    G_B.load_state_dict(torch.load(sub_model_dir + f'{pt}generator_B_param.pkl'))
    G_C.load_state_dict(torch.load(sub_model_dir + f'{pt}generator_C_param.pkl'))
    G_D.load_state_dict(torch.load(sub_model_dir + f'{pt}generator_D_param.pkl'))

    # Test
    for i, real_U in enumerate(test_data_U):
        # input eeg data
        real_U = real_U.reshape((params.batch_size, params.num_channel, real_U.shape[1], real_U.shape[2]))
        real_U = Variable(real_U.cuda())

        # U -> O -> V
        fake_O = G_A(real_U)
        fake_V = G_D(fake_O)



        # Show result for test data
        utils.plot_test_result(real_A, fake_B, recon_A, i, save=True, save_dir=sub_save_dir + 'AtoB/')

        print('%d images are generated.' % (i + 1))

"""
    for i, real_B in enumerate(test_data_loader_B):
        # input audio data
        real_B = Variable(real_B.cuda())

        # B -> A -> B
        fake_A = G_B(real_B)
        recon_B = G_A(fake_A)

        # Show result for test data
        utils.plot_test_result(real_B, fake_A, recon_B, i, save=True, save_dir=sub_save_dir + 'BtoA/')

        print('%d images are generated.' % (i + 1))
"""