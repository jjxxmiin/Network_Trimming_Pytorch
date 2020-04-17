import os
import sys
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import argparse
from torchvision import datasets, transforms
from PIL import ImageFile

from vgg import vgg16
from helper import train, valid
from converter import conv_post_mask, linear_mask, linear_pre_mask

# setting
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser(description='Network Trimming')
parser.add_argument('--data_path', type=str, default='/home/ubuntu/datasets/imagenet',
                    help='Path to root dataset folder ')
parser.add_argument('--save_path', type=str, default='./apoz_fine_tune_model.pth.tar',
                    help='Path to model save')
parser.add_argument('--prune_path', '-p', type=str,
                    default='./apoz_prune_model.pth.tar')
parser.add_argument('--epoch', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--device', '-d', type=str, default='cuda',
                    help='select [cpu / cuda]')
args = parser.parse_args()

# train/valid dataset
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

valid_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_path = os.path.join(args.data_path, 'train')
valid_path = os.path.join(args.data_path, 'val')

train_dataset = datasets.ImageFolder(train_path,
                                     transform=train_transform)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           pin_memory=True)

valid_dataset = datasets.ImageFolder(valid_path,
                                     transform=valid_transform)

valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=False,
                                           pin_memory=True)


# load prune model
checkpoint = torch.load(args.prune_path)

model = vgg16(pretrained=True).to(args.device)

mask = checkpoint['mask']

# Conv 5-3 [output]
model.features[-3] = conv_post_mask(model.features[-3], mask[0])
# FC 6 [input, output]
model.classifier[0] = linear_mask(model.classifier[0], mask[0], mask[1])
# FC 7 [input]
model.classifier[3] = linear_pre_mask(model.classifier[3], mask[1])

model.load_state_dict(checkpoint['state_dict'])

criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(),
                            args.lr,
                            weight_decay=1e-4)

best_top1 = 0

for e in range(args.epoch):
    train(model,
          train_loader,
          criterion,
          optimizer,
          f"EPOCH : [{e + 1} / {args.epoch}]")

    top1, top5 = valid(model,
                       valid_loader,
                       criterion)

    print(f"top1 : {top1} / top5 : {top5}")

    if top1 > best_top1:
        best_top1 = top1

        torch.save({'state_dict': model.state_dict()},
                   args.save_path + '.tar')

