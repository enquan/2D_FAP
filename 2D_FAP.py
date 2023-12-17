import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.distributed as dist
import torchsummary
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from thop import profile
from scipy.stats import norm
from models.MobileNetv2 import *
from models.MobileNetv2_modified import *
from models.ResNet import *
from models.VGG import *
from models.AttNet import *
from models.MobileNetv3 import *
from models.HMTNet import *

parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, default=1)
parser.add_argument('--alpha', type=int, default=1)
parser.add_argument('--beta', type=int, default=1)
parser.add_argument('--gamma', type=int, default=1)
parser.add_argument('--dataset', type=int, default=5500)
parser.add_argument('--aligned', type=bool, default=True)
parser.add_argument('--MTCNN', type=bool, default=False)
parser.add_argument('--sample', type=str, default='L')
parser.add_argument('--loss1', type=str, default='ED')
parser.add_argument('--loss1_option',type=str, default='mean')
parser.add_argument('--loss3', type=str, default='3')
parser.add_argument('--loss3_option',type=str, default='sum')
parser.add_argument('--losses', type=str, default='123')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--network', type=str, default='mobilenet')
parser.add_argument('--local_rank', type=int, default=-1)
parser.add_argument('--batch', type=int, default=256)
parser.add_argument('--count', type=bool, default=False)
parser.add_argument('--interval', type=float, default=0.1)
parser.add_argument('--min_score', type=int, default=1)
parser.add_argument('--max_score', type=int, default=5)
parser.add_argument('--device', type=str, default='0')
global args
args = parser.parse_args()

# random seed
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# setup_seed(20)

# device configuration
if args.network == 'vgg16' or args.network == 'vgg19':
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda:"+args.device, local_rank)
else:
    device = torch.device("cuda:"+args.device) if torch.cuda.is_available() else 'cpu'


# hyper-parameter
fold = args.fold
num_epochs = 90
batch_size = args.batch
learning_rate = args.lr  # 0.001

# transform
transform = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    'train2': transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    'test2': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
}

def cdf(x, mean, std):
    if args.sample == 'L':
        b = std / np.sqrt(2)
        return 0.5*(1+np.sign(x-mean)*(1-np.exp(-np.abs(x-mean)/b)))
    if args.sample == 'G':
        return norm.cdf(x, mean, std)

def sampling(x, mean, std):
    if args.sample == 'G':
        # std = 2
        return np.exp(-(np.power(x-mean,2))/(2*np.power(std,2)))/(std*np.sqrt(2*np.pi))
    elif args.sample == 'L':
        # b = np.sqrt(2)
        b = std / np.sqrt(2)
        return np.exp(-np.abs(x-mean)/b)/(2*b)



class SCUT(Dataset):
    def __init__(self, dataset, img_dir, datafile, transform):
        '''
        :param img_dir: 图片目录
        :param datafile: txt格式的数据文件
        :param transform: 图片变换
        '''
        self.dataset = dataset
        self.img_dir = img_dir
        self.datafile = datafile
        self.transform = transform
        self.img_label_list = self.read_img(img_dir=self.img_dir, datafile=self.datafile)
        self.len = len(self.img_label_list)

    def __getitem__(self, i):
        index = i % self.len
        imgpath, label, score, level = self.img_label_list[index]
        img = Image.open(imgpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label, score, level

    def __len__(self):
        return len(self.img_label_list)

    def read_img(self, img_dir, datafile):
        # Data loading
        output = []
        with open(datafile, 'r') as f:
            lines = f.readlines()
        for line in lines:
            linesplit = line.split('\n')[0].split(' ')
            addr = linesplit[0]
            score = float(linesplit[1])
            std = float(linesplit[2])
            level = [float(linesplit[l]) for l in range(3, 4 + args.max_score - args.min_score)]
            level = torch.Tensor(level)
            if self.dataset == 5500 or self.dataset == 500:
                level = level / torch.sum(level, dim=0)
            
            x = np.arange(args.min_score, args.max_score + args.interval, args.interval)
            target = [cdf(x[i+1],score,std)-cdf(x[i],score,std) for i in range(x.shape[0] - 1)]
            target = [j if j > 1e-15 else 1e-15 for j in target]
            target = torch.Tensor(target)
            # target = torch.nn.functional.softmax(target, dim=0)
            target = torch.sigmoid(target)
            target = torch.nn.functional.normalize(target, p=1, dim=0)
            imgpath = os.path.join(img_dir, addr)
            output.append((imgpath, target, score, level))
        return output

# loading the train/test data
if args.dataset == 5500:
    if args.aligned == True:
        if args.MTCNN == False:
            # SCUT-FBP5500 aligned
            img_dir = './data/SCUT-FBP5500/SCUT-FBP5500_v2/aligned'
            trainfile = './data/SCUT-FBP5500/SCUT-FBP5500_v2/train_test_files/5_folders_cross_validations_files/cross_validation_'+str(fold)+'/train_'+str(fold)+'_aligned_ext.txt'
            testfile = './data/SCUT-FBP5500/SCUT-FBP5500_v2/train_test_files/5_folders_cross_validations_files/cross_validation_'+str(fold)+'/test_'+str(fold)+'_aligned_ext.txt'
            train_data = SCUT(dataset=args.dataset, img_dir=img_dir, datafile=trainfile, transform=transform['train'])
            test_data = SCUT(dataset=args.dataset, img_dir=img_dir, datafile=testfile, transform=transform['test'])
        elif args.MTCNN == True:
            img_dir = './data/SCUT-FBP5500/SCUT-FBP5500_v2/MTCNN'
            trainfile = './data/SCUT-FBP5500/SCUT-FBP5500_v2/train_test_files/5_folders_cross_validations_files/cross_validation_'+str(fold)+'/train_'+str(fold)+'_mtcnn.txt'
            testfile = './data/SCUT-FBP5500/SCUT-FBP5500_v2/train_test_files/5_folders_cross_validations_files/cross_validation_'+str(fold)+'/test_'+str(fold)+'_mtcnn.txt'
            train_data = SCUT(dataset=args.dataset, img_dir=img_dir, datafile=trainfile, transform=transform['train2'])
            test_data = SCUT(dataset=args.dataset, img_dir=img_dir, datafile=testfile, transform=transform['test2'])
    elif args.aligned == False:
        # SCUT-FBP5500 unaligned
        img_dir = './data/SCUT-FBP5500/SCUT-FBP5500_v2/Images'
        trainfile = './data/SCUT-FBP5500/SCUT-FBP5500_v2/train_test_files/5_folders_cross_validations_files/cross_validation_'+str(fold)+'/train_'+str(fold)+'.txt'
        testfile = './data/SCUT-FBP5500/SCUT-FBP5500_v2/train_test_files/5_folders_cross_validations_files/cross_validation_'+str(fold)+'/test_'+str(fold)+'.txt'
        train_data = SCUT(dataset=args.dataset, img_dir=img_dir, datafile=trainfile, transform=transform['train'])
        test_data = SCUT(dataset=args.dataset, img_dir=img_dir, datafile=testfile, transform=transform['test'])
    if args.network == 'vgg16' or args.network == 'vgg19':
        batch_size = int(batch_size / 2)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, shuffle=True)
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, num_workers=8, pin_memory=True, sampler=train_sampler)
        test_loader = DataLoader(dataset=test_data, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=False)
    else:
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        test_loader = DataLoader(dataset=test_data, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=False)
elif args.dataset == 500:
    # SCUT-FBP
    train_img_dir = './data/SCUT-FBP/MTCNN/train'
    test_img_dir = './data/SCUT-FBP/MTCNN/test'
    trainfile = './data/SCUT-FBP/train.txt'
    testfile = './data/SCUT-FBP/test.txt'
    train_data = SCUT(dataset=args.dataset, img_dir=train_img_dir, datafile=trainfile, transform=transform['train2'])
    test_data = SCUT(dataset=args.dataset, img_dir=test_img_dir, datafile=testfile, transform=transform['test2'])
    if args.network == 'vgg16' or args.network == 'vgg19':
        batch_size = int(batch_size / 2)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, shuffle=True)
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, num_workers=8, pin_memory=True, sampler=train_sampler)
        test_loader = DataLoader(dataset=test_data, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=False)
    else:
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        test_loader = DataLoader(dataset=test_data, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=False)


# net definition
if args.network == 'mobilenet':
    net = mobilenet_v2(pretrained=True)
    in_channel = net.classifier[1].in_features
    net.classifier[1] = nn.Linear(in_channel, (int)((args.max_score-args.min_score)/args.interval))

if args.network == 'mobilenet_m':
    net = mobilenet_v2_m(pretrained=False)
    pretrained_model = torch.load('./models/mobilenet_v2-b0353104.pth')
    model_dict = net.state_dict()
    state_dict = {k:v for k, v in pretrained_model.items() if (k in model_dict.keys() and v.size()==model_dict[k].size())}
    model_dict.update(state_dict)
    net.load_state_dict(model_dict)
    in_channel = net.classifier[1].in_features
    net.classifier[1] = nn.Linear(in_channel, (int)((args.max_score-args.min_score)/args.interval))

if args.network == 'mobilenetv3_large':
    net = mobilenet_v3_large(pretrained=True)
    in_channel = net.classifier[3].in_features
    net.classifier[3] = nn.Linear(in_channel, (int)((args.max_score-args.min_score)/args.interval))

if args.network == 'mobilenetv3_small':
    net = mobilenet_v3_small(pretrained=True)
    in_channel = net.classifier[3].in_features
    net.classifier[3] = nn.Linear(in_channel, (int)((args.max_score-args.min_score)/args.interval))

if args.network == 'resnet18':
    net = resnet18(pretrained=True)
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, (int)((args.max_score-args.min_score)/args.interval))

if args.network == 'resnet50':
    net = resnet50(pretrained=True)
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, (int)((args.max_score-args.min_score)/args.interval))

if args.network == 'vgg16':
    net = vgg16_bn(pretrained=False)
    pretrained_model = torch.load('./models/vgg16_bn-6c64b313.pth')
    model_dict = net.state_dict()
    state_dict = {k:v for k, v in pretrained_model.items() if (k in model_dict.keys() and v.size()==model_dict[k].size())}
    model_dict.update(state_dict)
    net.load_state_dict(model_dict)

if args.network == 'vgg19':
    net = vgg19_bn(pretrained=False)
    pretrained_model = torch.load('./models/vgg19_bn-c79401a0.pth')
    model_dict = net.state_dict()
    state_dict = {k:v for k, v in pretrained_model.items() if (k in model_dict.keys() and v.size()==model_dict[k].size())}
    model_dict.update(state_dict)
    net.load_state_dict(model_dict)

if args.network == 'attnet_m':
    net = ThinAttNet_m()
    net.fc[0] = nn.Linear(512,(int)((args.max_score-args.min_score)/args.interval))

if args.network == 'hmt':
    net = HMTNet()

net = net.to(device)

if args.network == 'vgg16' or args.network == 'vgg19':
    net = DDP(net, device_ids=[local_rank], output_device=local_rank)

if args.count == True:
    print(net)
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    torchsummary.summary(net, (3, 224, 224))
    print('{}, #Params: {}'.format(args.network, count_parameters(net)))

    input = torch.randn(1, 3, 224, 224).to(device)
    macs, params = profile(net, inputs=(input, ), verbose=False)
    print("{}, #Params: {}, MAdds: {}".format(args.network, params, macs))
    exit()

# loss and optimizer
def kl_loss(inputs, labels):
    criterion = nn.KLDivLoss(reduction='batchmean')  # reduce=False
    outputs = torch.log(inputs+1e-15)  # sys.float_info.min
    loss = criterion(outputs, labels)
#     loss = loss.sum() / loss.shape[0]
    return loss


def L1_loss(inputs, labels):
    criterion = nn.L1Loss(reduction='mean')
    loss = criterion(inputs, labels.float())
    return loss

def L1_dis(inputs, labels, args):
    loss = torch.sum(torch.abs(inputs-labels),dim=1)
    if args.loss1_option == 'mean':
        return loss.mean()
    elif args.loss1_option =='sum':
        return loss.sum()

def Euclidean_dis(inputs, labels, args, weight=None):
    if weight == None:
        loss = torch.pow(torch.sum(torch.pow((inputs-labels),2),dim=1),0.5)  # 开根号
    # loss = torch.sum(torch.pow((inputs-labels),2),dim=1)
    else:
        loss = torch.pow(torch.sum(weight*torch.pow((inputs-labels),2),dim=1),0.5)  # 开根号
    if args.loss1_option == 'mean':
        return loss.mean()
    elif args.loss1_option =='sum':
        return loss.sum()

L1 = nn.L1Loss()
L2 = nn.MSELoss()
SmoothL1 = nn.SmoothL1Loss()
optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, amsgrad=True)

def my_loss(outputs, target, args):
    if args.loss3 == '1':
        loss = torch.log(0.5*(torch.exp(outputs-target)+torch.exp(target-outputs)))
    elif args.loss3 == '2':
        loss = torch.log(1+torch.abs(outputs-target))
    elif args.loss3 == '3':
        loss = torch.exp(torch.abs(outputs-target))-1
    elif args.loss3 == '4':
        loss = torch.log(torch.abs(outputs-target)+torch.pow((1+torch.pow(outputs-target,2)),0.5))
    elif args.loss3 == '5':
        return 0
    if args.loss3_option == 'mean':
        return loss.mean()
    elif args.loss3_option == 'sum':
        return loss.sum()

# learning rate update
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# test
def test():
    net.eval()
    label = np.array([])
    pred = np.array([])
    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            img, target, score, level = inputs  # target: (n, 40), score: (n)
            img = img.to(device)
            output = net(img)  # (n, 40)
            pred_score = torch.sum(output*rank, dim=1)  # (n)
            label = np.append(label, score.cpu().numpy())
            pred = np.append(pred, pred_score.cpu().numpy())
        print(label.shape)
        print(pred.shape)
        correlation = np.corrcoef(label, pred)[0][1]
        mae = np.mean(np.abs(label - pred))
        rmse = np.sqrt(np.mean(np.square(label - pred)))

    print('Pearson Correlation: {:.4f}, MAE: {:.4f}, RMSE: {:.4f}'.format(correlation, mae, rmse))
    # if correlation >= 0.931:
    #     torch.save(net, str(correlation)+'-'+str(mae)+'-'+str(rmse)+'.pth')


# train
def train(epoch, rank, alpha, beta, gamma):
    net.train()
    if args.network == 'vgg16' or args.network == 'vgg19':
        train_loader.sampler.set_epoch(epoch)
    for i, inputs in enumerate(train_loader):
        img, target, score, level = inputs
        img = img.to(device)
        target = target.to(device)
        score = score.to(device)
        level = level.to(device)
        optimizer.zero_grad()
        outputs = net(img) #(n, 40)

        plevel = torch.zeros(outputs.shape[0], args.max_score - args.min_score + 1, dtype=torch.float).to(device)
        cnt = torch.tensor([10] * (args.max_score - args.min_score + 1)).to(device)
        cnt[0] -= 5
        cnt[-1] -= 5
        idx = [i.repeat(times) for i, times in zip(torch.arange(len(cnt)),cnt)]
        idx = torch.cat(idx).to(device)
        plevel.index_add_(dim=1, index=idx, source=outputs)

        # scores = torch.sum(plevel*rank, dim=1)  # (n)
        scores = torch.sum(outputs*rank, dim=1)  # (n)
        if args.loss1 == 'L1':
            loss1 = L1_dis(outputs, target, args)
        elif args.loss1 == 'ED':
            loss1 = Euclidean_dis(outputs, target, args)
        elif args.loss1 == 'KL':
            loss1 = kl_loss(outputs, target)
        loss2 = Euclidean_dis(plevel, level, args, weight=None)

        if args.loss3 == 'L1':
            loss3 = L1(scores, score)
        else:
            loss3 = my_loss(scores, score, args)
        
        total_loss = 0
        if '1' in args.losses:
            total_loss += alpha * loss1
        if '2' in args.losses:
            total_loss += beta * loss2
        if '3' in args.losses:
            total_loss += gamma * loss3
        total_loss.backward()
        optimizer.step()

        if (i + 1) == total_step:
            print("Epoch [{}/{}], Step [{}/{}], Loss 1: {:.4f}, Loss 2: {:.4f}, Loss 3: {:.4f}, Loss: {:.4f}".format(epoch + 1, num_epochs, i + 1, total_step, loss1, loss2, loss3, total_loss), flush=True)
    return total_loss.item()


total_step = len(train_loader)
curr_lr = learning_rate
rank = torch.Tensor([i for i in np.arange(args.min_score + 0.5 * args.interval, args.max_score + 0.5 * args.interval, args.interval)]).to(device)

for epoch in range(num_epochs):
    _ = train(epoch=epoch,rank=rank, alpha=args.alpha, beta=args.beta, gamma=args.gamma)

    if (epoch + 1) % 30 == 0:
        curr_lr /= 10
        update_lr(optimizer, curr_lr)

test()
