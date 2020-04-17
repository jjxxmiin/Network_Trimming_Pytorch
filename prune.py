import os
import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from vgg import vgg16
from apoz import APoZ
from helper import save_pkl, load_pkl, valid
from converter import conv_post_mask, linear_mask, linear_pre_mask

parser = argparse.ArgumentParser(description='Network Trimming')
parser.add_argument('--data_path', type=str, default='/home/ubuntu/datasets/imagenet',
                    help='Path to root dataset folder ')
parser.add_argument('--save_path', type=str, default='./apoz_prune_model.pth.tar',
                    help='Path to model save')
parser.add_argument('--select_rate', type=int, default=0,
                    help='0 : (488, 3477) \n'
                         '1 : (451, 2937) \n'
                         '2 : (430, 2479) \n'
                         '3 : (420, 2121) \n'
                         '4 : (400, 1787) \n'
                         '5 : (390, 1513)')
parser.add_argument('--apoz_path', type=str, default='./vgg_apoz_fc.pkl',
                    help='Path to apoz pkl')
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--device', '-d', type=str, default='cuda',
                    help='select [cpu / cuda]')
args = parser.parse_args()

module_name = ['Conv 1-1', 'Conv 1-2', 'Conv 2-1', 'Conv 2-2', 'Conv 3-1',
               'Conv 3-2', 'Conv 3-3', 'Conv 4-1', 'Conv 4-2', 'Conv 4-3',
               'Conv 5-1', 'Conv 5-2', 'Conv 5-3', 'FC 6', 'FC 7']

rate = [(488, 3477),
        (451, 2937),
        (430, 2479),
        (420, 2121),
        (400, 1787),
        (390, 1513)]

select_rate = rate[args.select_rate]

# train/valid dataset
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

valid_dataset = datasets.ImageFolder(os.path.join(args.data_path, 'val'),
                                     transform=val_transform)

valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                           batch_size=args.batch_size,
                                           pin_memory=True)

criterion = nn.CrossEntropyLoss().cuda()

model = vgg16(pretrained=True).to(args.device)

show_summary(model)

# save apoz pkl
if not os.path.exists(args.apoz_path):
    apoz = APoZ(model).get_apoz(valid_loader, criterion)
    save_pkl(apoz, args.apoz_path)

else:
    apoz = load_pkl(args.apoz_path)

# info apoz
print("Average Percentage Of Zero Mean")
for n, p in zip(module_name, apoz):
    print(f"{n} : {p.mean() * 100 : .2f}%")

# Masking
mask = []

for i, p in enumerate(apoz[-3:-1]):
    sorted_arg = np.argsort(p)
    mask.append(sorted_arg < select_rate[i])

# Conv 5-3 [output]
model.features[-3] = conv_post_mask(model.features[-3], mask[0])
# FC 6 [input, output]
model.classifier[0] = linear_mask(model.classifier[0], mask[0], mask[1])
# FC 7 [input]
model.classifier[3] = linear_pre_mask(model.classifier[3], mask[1])

torch.save({'cfg': ['Conv 5-3', 'FC 6'],
            'mask': mask,
            'state_dict': model.state_dict()},
             args.save_path)

# valid
acc_top1, acc_top5 = valid(model, valid_loader, criterion)

print(f"Acc@1: {acc_top1} \n"
      f"Acc@5: {acc_top5} \n")
