# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 11:44:05 2021

@author: nilah
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as tmodels
from functools import partial
import collections

# dummy data: 10 batches of images with batch size 16
dataset = [torch.rand(16,3,224,224).cuda() for _ in range(10)]

# network: a resnet50
net = tmodels.resnet50(pretrained=True).cuda()

# a dictionary that keeps saving the activations as they come
activations = collections.defaultdict(list)
def save_activation(name, mod, inp, out):
	activations[name].append(out.cpu())

# Registering hooks for all the Conv2d layers
# Note: Hooks are called EVERY TIME the module performs a forward pass. For modules that are
# called repeatedly at different stages of the forward pass (like RELUs), this will save different
# activations. Editing the forward pass code to save activations is the way to go for these cases.
for name, m in net.named_modules():
	if type(m)==nn.Conv2d:
		# partial to assign the layer name to each hook
		m.register_forward_hook(partial(save_activation, name))

# forward pass through the full dataset
for batch in dataset:
	out = net(batch)

# concatenate all the outputs we saved to get the the activations for each layer for the whole dataset
activations = {name: torch.cat(outputs, 0) for name, outputs in activations.items()}

# just print out the sizes of the saved activations as a sanity check
for k,v in activations.items():
	print (k, v.size())
    
'''    
def hook( m, i, o):
    print( m._get_name() )

for ( mo ) in model.modules():
    mo.register_forward_hook(hook)
'''
def forward(self, input, explain=False, rule="epsilon", **kwargs):
        if not explain: return super(Conv2d, self).forward(input)
        return self._conv_forward_explain(input, self.weight, conv2d[rule], **kwargs)
    
def _conv_forward_explain(self, input, weight, conv2d_fn, **kwargs):
        if self.padding_mode != 'zeros':
            return conv2d_fn(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups, **kwargs)

        p = kwargs.get('pattern')
        if p is not None: 
            return conv2d_fn(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups, p)
        else: return conv2d_fn(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
class Linear(torch.nn.Linear):
    def forward(self, input, explain=False, rule="epsilon", **kwargs):
        if not explain: return super(Linear, self).forward(input)

        p = kwargs.get('pattern')
        if p is not None: return linear[rule](input, self.weight, self.bias, p)
        else: return linear[rule](input, self.weight, self.bias)

    @classmethod
    def from_torch(cls, lin):
        bias = lin.bias is not None
        module = cls(in_features=lin.in_features, out_features=lin.out_features, bias=bias)
        module.load_state_dict(lin.state_dict())
