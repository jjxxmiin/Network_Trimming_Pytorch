import numpy as np
import torch.nn as nn
from vgg import feature_cfgs, classifier_cfgs
from helper import valid

class APoZ:
    def __init__(self, model):
        self.model = model
        self.idx = 0
        self.num_layer = 0
        self.apoz = []

        for c in feature_cfgs + classifier_cfgs:
            if c is 'M':
                continue

            self.apoz.append([0] * c)
            self.num_layer += 1

        self.apoz = np.array(self.apoz)

        self.register()

        print(f"Layer(ReLU + Linear) {self.num_layer} module register")

    def get_zero_percent_hook(self, module, input, output):
        if output.dim() == 4:
            p_zero = (output == 0).sum(dim=(2, 3)).float() / (output.size(2) * output.size(3))
            self.apoz[self.idx] += p_zero.mean(dim=0).cpu().numpy()
        elif output.dim() == 2:
            p_zero = (output == 0).sum(dim=0).float() / output.size(0)
            self.apoz[self.idx] += p_zero.cpu().numpy()
        else:
            raise ValueError(f"{output.dim()} dimension is Not Supported")

        self.idx += 1

        if self.idx == self.num_layer:
            self.idx = 0

    def register(self):
        for module in self.model.features.modules():
            if type(module) == nn.ReLU:
                module.register_forward_hook(self.get_zero_percent_hook)

        for module in self.model.classifier.modules():
            if type(module) == nn.ReLU:
                module.register_forward_hook(self.get_zero_percent_hook)

    def get_apoz(self, loader, criterion):
        top1, top5 = valid(self.model,
                           loader,
                           criterion)

        print(f"top1 : {top1} top5 : {top5}")

        return self.apoz / len(loader)
