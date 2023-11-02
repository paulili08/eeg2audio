import torch
from torch.autograd import Variable
from dataset import EEGDataset
from GAN_model import Generator, Discriminator
import utils
import argparse
import os, itertools
from logger import Logger
import numpy as np

"""
PARAMETERS
Cascade Proportion = 0.6
batch_size = 1
The number of critic iterations per generator iteration: num_iter_G = 5
learning rate: lrD = 0.0002  lrG = 0.0002
lambda: U = 500  O = 500  V = 500
"""
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='eeg2audio', help='input dataset')
parser.add_argument('--batch_size', type=int, default=1, help='train batch size')
parser.add_argument('--num_batch', type=int, default=429, help='num of batches whole dataset divided')
parser.add_argument('--nfolds', type=int, default=12, help='K-fold')
parser.add_argument('--ngf', type=int, default=64, help='num of filter in generator')
parser.add_argument('--ndf', type=int, default=64, help='num of filter in discriminator')
parser.add_argument('--cas_prop', type=float, default=0.6, help='cascade proportion')
parser.add_argument('--num_channel', type=int, default=1, help='number of channels for input image')
parser.add_argument('--fliplr', type=bool, default=True, help='random fliplr True of False')
parser.add_argument('--num_epochs', type=int, default=30, help='number of train epochs')
parser.add_argument('--num_iter_G', type=int, default=5, help='number of iterations for training generator')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate for generator, default=0.0002')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate for discriminator, default=0.0002')
parser.add_argument('--decay', type=float, default=0.1, help='weight decay for RMSProp optimizer')
parser.add_argument('--lambdaU', type=float, default=500, help='lambdaU for L1 loss')
parser.add_argument('--lambdaO', type=float, default=500, help='lambdaO for L1 loss')
parser.add_argument('--lambdaV', type=float, default=500, help='lambdaV for L1 loss')
params = parser.parse_args()
print(params)

# Directories for loading data and saving results
feat_path = r'./features'
pts = ['sub-%02d'%i for i in range(1,11)]
data_dir = '../Data/' + params.dataset + '/'
save_dir = params.dataset + '_results/'
model_dir = params.dataset + '_model/'
test_dir = params.dataset + '_testset/'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
if not os.path.exists(test_dir):
    os.mkdir(test_dir)


test = np.arange(2496)
train = np.arange(2496,29952)

for pNr, pt in enumerate(pts):
    #Load the data
    data_U = np.load(os.path.join(feat_path, f'{pt}_feat.npy')) #eeg
    data_V = np.load(os.path.join(feat_path, f'{pt}_spec.npy')) #audio
    data_len = data_V.shape[0]
    pts = ['sub-%02d'%i for i in range(1,11)]

    # Directories for saving results
    sub_save_dir = save_dir + f'{pt}' + '/'
    sub_model_dir = model_dir + f'{pt}' + '/'
    sub_test_dir = test_dir + f'{pt}' + '/'

    if not os.path.exists(sub_save_dir):
        os.mkdir(sub_save_dir)
    if not os.path.exists(sub_model_dir):
        os.mkdir(sub_model_dir)
    if not os.path.exists(sub_test_dir):
        os.mkdir(sub_test_dir)

    # Data pre-processing
    # Proportional cascade
    eeg_part = data_U[0:int(data_len*params.cas_prop), :]
    audio_part = data_V[int(data_len*params.cas_prop):, :]
    data_O = np.vstack((eeg_part,audio_part))

    # Z-Normalize with mean and std from the training data
    mu = np.mean(data_U[train, :], axis=0)
    std = np.std(data_U[train, :], axis=0)
    train_data_U = (data_U[train, :] - mu) / std
    test_data_U = (data_U[test, :] - mu) / std

    mu = np.mean(data_V[train, :], axis=0)
    std = np.std(data_V[train, :], axis=0)
    train_data_V = (data_V[train, :] - mu) / std
    test_data_V = (data_V[test, :] - mu) / std

    mu = np.mean(data_O[train, :], axis=0)
    std = np.std(data_O[train, :], axis=0)
    train_data_O = (data_O[train, :] - mu) / std
    test_data_O = (data_O[test, :] - mu) / std

    # save test data
    np.save(os.path.join(sub_test_dir, 'test_data_U'), test_data_U)
    np.save(os.path.join(sub_test_dir, 'test_data_O'), test_data_O)
    np.save(os.path.join(sub_test_dir, 'test_data_V'), test_data_V)

    # divide into several batches
    train_data_U = np.array_split(train_data_U, params.num_batch, axis=0)
    train_data_O = np.array_split(train_data_O, params.num_batch, axis=0)
    train_data_V = np.array_split(train_data_V, params.num_batch, axis=0)

    # Train data
    train_data_U = EEGDataset(train_data_U)
    train_data_U = torch.utils.data.DataLoader(dataset=train_data_U,
                                                      batch_size=params.batch_size,
                                                      shuffle=True)
    train_data_V = EEGDataset(train_data_V)
    train_data_V = torch.utils.data.DataLoader(dataset=train_data_V,
                                                      batch_size=params.batch_size,
                                                      shuffle=True)
    train_data_O = EEGDataset(train_data_O)
    train_data_O = torch.utils.data.DataLoader(dataset=train_data_O,
                                                      batch_size=params.batch_size,
                                                      shuffle=True)
    """
    # Test data
    test_data_U = DatasetFromFolder(data_dir, subfolder='val/U', transform=transform, is_color=False)
    test_data_loader_U = torch.utils.data.DataLoader(dataset=test_data_U,
                                                     batch_size=params.batch_size,
                                                     shuffle=False)
    test_data_V = DatasetFromFolder(data_dir, subfolder='val/V', transform=transform, is_color=False)
    test_data_loader_V = torch.utils.data.DataLoader(dataset=test_data_V,
                                                     batch_size=params.batch_size,
                                                     shuffle=False)
    test_data_O = DatasetFromFolder(data_dir, subfolder='val/O', transform=transform, is_color=False)
    test_data_loader_O = torch.utils.data.DataLoader(dataset=test_data_O,
                                                     batch_size=params.batch_size,
                                                     shuffle=False)

    # Get specific test signals ???
    test_real_U_data = test_data_U.__getitem__(0).unsqueeze(0)  # Convert to 4d tensor (BxNxHxW)
    test_real_V_data = test_data_V.__getitem__(0).unsqueeze(0)
    
    """

    # Models
    G_A = Generator(params.num_channel, params.ngf, params.num_channel)
    G_B = Generator(params.num_channel, params.ngf, params.num_channel)
    G_C = Generator(params.num_channel, params.ngf, params.num_channel)
    G_D = Generator(params.num_channel, params.ngf, params.num_channel)
    D_A = Discriminator(params.num_channel, params.ndf, 1)
    D_B = Discriminator(params.num_channel, params.ndf, 1)
    D_C = Discriminator(params.num_channel, params.ndf, 1)
    D_D = Discriminator(params.num_channel, params.ndf, 1)
    G_A.normal_weight_init(mean=0.0, std=0.02)
    G_B.normal_weight_init(mean=0.0, std=0.02)
    G_C.normal_weight_init(mean=0.0, std=0.02)
    G_D.normal_weight_init(mean=0.0, std=0.02)
    D_A.normal_weight_init(mean=0.0, std=0.02)
    D_B.normal_weight_init(mean=0.0, std=0.02)
    D_C.normal_weight_init(mean=0.0, std=0.02)
    D_D.normal_weight_init(mean=0.0, std=0.02)
    G_A.cuda()
    G_B.cuda()
    G_C.cuda()
    G_D.cuda()
    D_A.cuda()
    D_B.cuda()
    D_C.cuda()
    D_D.cuda()

    # Set the logger
    D_A_log_dir = sub_save_dir + 'D_A_logs'
    D_B_log_dir = sub_save_dir + 'D_B_logs'
    D_C_log_dir = sub_save_dir + 'D_C_logs'
    D_D_log_dir = sub_save_dir + 'D_D_logs'

    if not os.path.exists(D_A_log_dir):
        os.mkdir(D_A_log_dir)
    D_A_logger = Logger(D_A_log_dir)
    if not os.path.exists(D_B_log_dir):
        os.mkdir(D_B_log_dir)
    D_B_logger = Logger(D_B_log_dir)
    if not os.path.exists(D_C_log_dir):
        os.mkdir(D_C_log_dir)
    D_C_logger = Logger(D_C_log_dir)
    if not os.path.exists(D_D_log_dir):
        os.mkdir(D_D_log_dir)
    D_D_logger = Logger(D_D_log_dir)

    G_A_log_dir = sub_save_dir + 'G_A_logs'
    G_B_log_dir = sub_save_dir + 'G_B_logs'
    G_C_log_dir = sub_save_dir + 'G_C_logs'
    G_D_log_dir = sub_save_dir + 'G_D_logs'
    if not os.path.exists(G_A_log_dir):
        os.mkdir(G_A_log_dir)
    G_A_logger = Logger(G_A_log_dir)
    if not os.path.exists(G_B_log_dir):
        os.mkdir(G_B_log_dir)
    G_B_logger = Logger(G_B_log_dir)
    if not os.path.exists(G_C_log_dir):
        os.mkdir(G_C_log_dir)
    G_C_logger = Logger(G_C_log_dir)
    if not os.path.exists(G_D_log_dir):
        os.mkdir(G_D_log_dir)
    G_D_logger = Logger(G_D_log_dir)


    L1_A_log_dir = sub_save_dir + 'L1_A_logs'
    L1_B_log_dir = sub_save_dir + 'L1_B_logs'
    L1_C_log_dir = sub_save_dir + 'L1_C_logs'
    L1_D_log_dir = sub_save_dir + 'L1_D_logs'

    if not os.path.exists(L1_A_log_dir):
        os.mkdir(L1_A_log_dir)
    L1_A_logger = Logger(L1_A_log_dir)
    if not os.path.exists(L1_B_log_dir):
        os.mkdir(L1_B_log_dir)
    L1_B_logger = Logger(L1_B_log_dir)
    if not os.path.exists(L1_C_log_dir):
        os.mkdir(L1_C_log_dir)
    L1_C_logger = Logger(L1_C_log_dir)
    if not os.path.exists(L1_D_log_dir):
        os.mkdir(L1_D_log_dir)
    L1_D_logger = Logger(L1_D_log_dir)

    """
    img_log_dir = sub_save_dir + 'img_logs'
    if not os.path.exists(img_log_dir):
        os.mkdir(img_log_dir)
    img_logger = Logger(img_log_dir)
    """

    # Loss function
    BCE_loss = torch.nn.BCELoss().cuda()
    L1_loss = torch.nn.L1Loss().cuda()

    # optimizers
    G_AB_optimizer = torch.optim.RMSprop(
        itertools.chain(G_A.parameters(), G_B.parameters(), G_C.parameters(), G_D.parameters()), lr=params.lrG,
        weight_decay=params.decay)
    G_CD_optimizer = torch.optim.RMSprop(
        itertools.chain(G_A.parameters(), G_B.parameters(), G_C.parameters(), G_D.parameters()), lr=params.lrG,
        weight_decay=params.decay)
    D_A_optimizer = torch.optim.RMSprop(D_A.parameters(), lr=params.lrD, weight_decay=params.decay)
    D_B_optimizer = torch.optim.RMSprop(D_B.parameters(), lr=params.lrD, weight_decay=params.decay)
    D_C_optimizer = torch.optim.RMSprop(D_C.parameters(), lr=params.lrD, weight_decay=params.decay)
    D_D_optimizer = torch.optim.RMSprop(D_D.parameters(), lr=params.lrD, weight_decay=params.decay)

    # Training GAN
    """
    D_A_avg_losses = []
    D_B_avg_losses = []
    D_C_avg_losses = []
    D_D_avg_losses = []
    G_A_avg_losses = []
    G_B_avg_losses = []
    G_C_avg_losses = []
    G_D_avg_losses = []
    L1_A_avg_losses = []
    L1_B_avg_losses = []
    L1_C_avg_losses = []
    L1_D_avg_losses = []
    """
    step = 0
    for epoch in range(params.num_epochs):
        D_A_losses = []
        D_B_losses = []
        D_C_losses = []
        D_D_losses = []
        G_A_losses = []
        G_B_losses = []
        G_C_losses = []
        G_D_losses = []
        L1_A_losses = []
        L1_B_losses = []
        L1_C_losses = []
        L1_D_losses = []

        # training
        for i, (real_U, real_V, real_O) in enumerate(zip(train_data_U, train_data_V, train_data_O)):

            # input signal
            real_U = real_U.reshape((params.batch_size, params.num_channel, real_U.shape[1], real_U.shape[2]))
            real_U = Variable(real_U.cuda())
            real_V = real_V.reshape((params.batch_size, params.num_channel, real_V.shape[1], real_V.shape[2]))
            real_V = Variable(real_V.cuda())
            real_O = real_O.reshape((params.batch_size, params.num_channel, real_O.shape[1], real_O.shape[2]))
            real_O = Variable(real_O.cuda())

            for _ in range(params.num_iter_G):
                # Train generator G
                # U -> O
                fake_O = G_A(real_U)
                D_B_fake_decision = D_B(fake_O)
                G_A_loss = BCE_loss(D_B_fake_decision, Variable(torch.ones(D_B_fake_decision.size()).cuda()))

                # O -> U
                # a = fake_O.cpu().detach().numpy()
                fake_U = G_B(real_O)
                D_A_fake_decision = D_A(fake_U)
                G_B_loss = BCE_loss(D_A_fake_decision, Variable(torch.ones(D_A_fake_decision.size()).cuda()))

                # forward L1 Loss (reconstruction error)*lambda
                recon_U = G_B(fake_O)
                L1_A_loss = L1_loss(recon_U, real_U) * params.lambdaU

                # backward L1 Loss (reconstruction error)*lambda
                recon_O = G_A(fake_U)
                L1_B_loss = L1_loss(recon_O, real_O) * params.lambdaO

                # Back propagation
                G_AB_loss = G_A_loss + G_B_loss + L1_A_loss + L1_B_loss
                G_AB_optimizer.zero_grad()
                G_AB_loss.backward(retain_graph=True)
                G_AB_optimizer.step()

                # O -> V
                fake_V = G_D(recon_O)
                D_C_fake_decision = D_C(fake_V)
                G_C_loss = BCE_loss(D_C_fake_decision, Variable(torch.ones(D_C_fake_decision.size()).cuda()))

                # V -> O
                fake_O_ = G_C(real_V)
                D_D_fake_decision = D_D(fake_O_)
                G_D_loss = BCE_loss(D_D_fake_decision, Variable(torch.ones(D_D_fake_decision.size()).cuda()))

                # forward L1 Loss (reconstruction error)*lambda
                recon_O_ = G_C(fake_V)
                L1_D_loss = L1_loss(recon_O_, recon_O) * params.lambdaO

                # backward L1 Loss (reconstruction error)*lambda
                recon_V = G_D(fake_O_)
                L1_C_loss = L1_loss(recon_V, real_V) * params.lambdaV

                # Back propagation
                G_CD_loss = G_C_loss + G_D_loss + L1_C_loss + L1_D_loss
                G_CD_optimizer.zero_grad()
                G_CD_loss.backward(retain_graph=True)
                G_CD_optimizer.step()

            # Train discriminator D_A
            D_A_real_decision = D_A(real_U)
            D_A_real_loss = BCE_loss(D_A_real_decision, Variable(torch.ones(D_A_real_decision.size()).cuda()))
            D_A_fake_decision = D_A(fake_U)
            D_A_fake_loss = BCE_loss(D_A_fake_decision, Variable(torch.zeros(D_A_fake_decision.size()).cuda()))

            # Back propagation
            D_A_loss = D_A_real_loss + D_A_fake_loss
            D_A_optimizer.zero_grad()
            D_A_loss.backward(retain_graph=True)
            D_A_optimizer.step()

            # Train discriminator D_B
            D_B_real_decision = D_B(real_O)
            D_B_real_loss = BCE_loss(D_B_real_decision, Variable(torch.ones(D_B_real_decision.size()).cuda()))
            D_B_fake_decision = D_B(fake_O)
            D_B_fake_loss = BCE_loss(D_B_fake_decision, Variable(torch.zeros(D_B_fake_decision.size()).cuda()))

            # Back propagation
            D_B_loss = D_B_real_loss + D_B_fake_loss
            D_B_optimizer.zero_grad()
            D_B_loss.backward(retain_graph=True)
            D_B_optimizer.step()

            # Train discriminator D_D
            D_D_real_decision = D_D(recon_O)
            D_D_real_loss = BCE_loss(D_A_real_decision, Variable(torch.ones(D_D_real_decision.size()).cuda()))
            D_D_fake_decision = D_A(fake_O_)
            D_D_fake_loss = BCE_loss(D_D_fake_decision, Variable(torch.zeros(D_D_fake_decision.size()).cuda()))

            # Back propagation
            D_D_loss = D_D_real_loss + D_D_fake_loss
            D_D_optimizer.zero_grad()
            D_D_loss.backward()
            D_D_optimizer.step()

            # Train discriminator D_C
            D_C_real_decision = D_C(real_V)
            D_C_real_loss = BCE_loss(D_C_real_decision, Variable(torch.ones(D_C_real_decision.size()).cuda()))
            D_C_fake_decision = D_C(fake_V)
            D_C_fake_loss = BCE_loss(D_C_fake_decision, Variable(torch.zeros(D_C_fake_decision.size()).cuda()))

            # Back propagation
            D_C_loss = D_C_real_loss + D_C_fake_loss
            D_C_optimizer.zero_grad()
            D_C_loss.backward()
            D_C_optimizer.step()

            # loss values
            D_A_losses.append(D_A_loss.item())
            D_B_losses.append(D_B_loss.item())
            G_A_losses.append(G_A_loss.item())
            G_B_losses.append(G_B_loss.item())
            L1_A_losses.append(L1_A_loss.item())
            L1_B_losses.append(L1_B_loss.item())

            D_D_losses.append(D_D_loss.item())
            D_C_losses.append(D_C_loss.item())
            G_D_losses.append(G_D_loss.item())
            G_C_losses.append(G_C_loss.item())
            L1_D_losses.append(L1_D_loss.item())
            L1_C_losses.append(L1_C_loss.item())

            print('Epoch [%d/%d], Step [%d/%d], D_A_loss: %.4f, D_B_loss: %.4f, G_A_loss: %.4f, G_B_loss: %.4f'
                  % (epoch + 1, params.num_epochs, i + 1, len(train_data_U), D_A_loss.item(), D_B_loss.item(),
                     G_A_loss.item(), G_B_loss.item()))

            print('Epoch [%d/%d], Step [%d/%d], D_C_loss: %.4f, D_D_loss: %.4f, G_C_loss: %.4f, G_D_loss: %.4f'
                  % (epoch + 1, params.num_epochs, i + 1, len(train_data_V), D_C_loss.item(), D_D_loss.item(),
                     G_C_loss.item(), G_D_loss.item()))

            # ============ TensorBoard logging ============#
            D_A_logger.scalar_summary('losses', D_A_loss.item(), step + 1)
            D_B_logger.scalar_summary('losses', D_B_loss.item(), step + 1)
            G_A_logger.scalar_summary('losses', G_A_loss.item(), step + 1)
            G_B_logger.scalar_summary('losses', G_B_loss.item(), step + 1)
            L1_A_logger.scalar_summary('losses', L1_A_loss.item(), step + 1)
            L1_B_logger.scalar_summary('losses', L1_B_loss.item(), step + 1)

            D_C_logger.scalar_summary('losses', D_C_loss.item(), step + 1)
            D_D_logger.scalar_summary('losses', D_D_loss.item(), step + 1)
            G_C_logger.scalar_summary('losses', G_C_loss.item(), step + 1)
            G_D_logger.scalar_summary('losses', G_D_loss.item(), step + 1)
            L1_C_logger.scalar_summary('losses', L1_C_loss.item(), step + 1)
            L1_D_logger.scalar_summary('losses', L1_D_loss.item(), step + 1)
            step += 1

        """
        D_A_avg_loss = torch.mean(torch.FloatTensor(D_A_losses))
        D_B_avg_loss = torch.mean(torch.FloatTensor(D_B_losses))
        G_A_avg_loss = torch.mean(torch.FloatTensor(G_A_losses))
        G_B_avg_loss = torch.mean(torch.FloatTensor(G_B_losses))
        L1_A_avg_loss = torch.mean(torch.FloatTensor(L1_A_losses))
        L1_B_avg_loss = torch.mean(torch.FloatTensor(L1_B_losses))
        D_C_avg_loss = torch.mean(torch.FloatTensor(D_C_losses))
        D_D_avg_loss = torch.mean(torch.FloatTensor(D_D_losses))
        G_C_avg_loss = torch.mean(torch.FloatTensor(G_C_losses))
        G_D_avg_loss = torch.mean(torch.FloatTensor(G_D_losses))
        L1_C_avg_loss = torch.mean(torch.FloatTensor(L1_C_losses))
        L1_D_avg_loss = torch.mean(torch.FloatTensor(L1_D_losses))
        """

        torch.save(G_A.state_dict(), sub_model_dir + f'{pt}generator_A_param.pkl')
        torch.save(G_B.state_dict(), sub_model_dir + f'{pt}generator_B_param.pkl')
        torch.save(D_A.state_dict(), sub_model_dir + f'{pt}discriminator_A_param.pkl')
        torch.save(D_B.state_dict(), sub_model_dir + f'{pt}discriminator_B_param.pkl')
        torch.save(G_D.state_dict(), sub_model_dir + f'{pt}generator_D_param.pkl')
        torch.save(G_C.state_dict(), sub_model_dir + f'{pt}generator_C_param.pkl')
        torch.save(D_D.state_dict(), sub_model_dir + f'{pt}discriminator_D_param.pkl')
        torch.save(D_C.state_dict(), sub_model_dir + f'{pt}discriminator_C_param.pkl')
