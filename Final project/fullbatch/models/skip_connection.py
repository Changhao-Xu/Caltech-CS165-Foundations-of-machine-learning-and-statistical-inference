import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function


class SkipConnection(nn.Module):

    def __init__(self, scale=1):
        super(SkipConnection, self).__init__()
        self.scale = scale
    def _shortcut(self, input):
        #needs to be implemented

        return input

    def forward(self, x):
        # with torch.no_grad():
        identity = self._shortcut(x)
        return identity * self.scale

class DeterChannelDropoutSkip(SkipConnection):

    def __init__(self, num_remain_channels, scale=1):
        super(DeterChannelDropoutSkip, self).__init__(scale)
        self.num_remain_channels = num_remain_channels
    
    def _shortcut(self, input):
        # input is (N, C, H, M)
        # and return is (N, C-num_reduce_channels, H, M)
        
        return input[:,0:self.num_remain_channels,:,:]        

class ChannelPaddingSkip(SkipConnection):

    def __init__(self, num_expand_channels_left, num_expand_channels_right, scale=1):
        super(ChannelPaddingSkip, self).__init__(scale)
        self.num_expand_channels_left = num_expand_channels_left
        self.num_expand_channels_right = num_expand_channels_right
    
    def _shortcut(self, input):
        # input is (N, C, H, M)
        # and return is (N, C + num_left + num_right, H, M)
        
        return F.pad(input, (0, 0, 0, 0, self.num_expand_channels_left, self.num_expand_channels_right) , "constant", 0) 

class ChannelRandomPaddingSkip(SkipConnection):

    def __init__(self, input_num, output_num, scale=1):
        super(ChannelRandomPaddingSkip, self).__init__(scale)
        self.input_num = input_num
        self.output_num = output_num

        assert self.output_num % self.input_num == 0
        
        self.num_replica = int(self.output_num / self.input_num)
        self.weight = 1/self.num_replica
    
    def _shortcut(self, input):
        # input is (N, C, H, M)
        # and return is (N, C + (output_num-input_num), H, M)


        input_size = input.size()
        output = torch.zeros(input_size[0],self.output_num,*input_size[2:]).type_as(input)

        index=0
        for i in range(self.num_replica):
            perm_int = torch.randperm(self.input_num)
            for k in perm_int:
                output[:,index] = self.weight * input[:,k]
                index +=1
        
        return output

class RandomProjectionSkip(SkipConnection):

    def __init__(self, input_num, output_num, scale=1):
        super(RandomProjectionSkip, self).__init__(scale)
        self.scale = scale
        self.input_num = input_num
        self.output_num = output_num

    def _shortcut(self, input):
        # skip connection for linear layer with random projection
        # input: (N, input_num)
        # output: (N, output_num)
        perm_int = torch.randperm(self.input_num)
        choice_int = perm_int[:self.output_num]

        return input[:,choice_int]

class IterateProjectionSkip(SkipConnection):

    def __init__(self, input_num, output_num, switch_iter=1, scale=1):
        super(IterateProjectionSkip, self).__init__(scale)
        self.scale = scale
        self.input_num = input_num
        self.output_num = output_num
        if self.input_num % self.output_num !=0:
            raise RuntimeError('output_dim should be mutiples of input_dim')
        self.last_index = 0
        self.iterates = 0
        self.switch_iter = switch_iter

    def _shortcut(self, input):
        # skip connection for linear layer with random projection
        # input: (N, input_num)
        # output: (N, output_num)
        output = input[:,self.last_index:(self.last_index+self.output_num)]

        self.iterates += 1
        if self.iterates == self.switch_iter:
            self.iterates = 0
            self.last_index += self.output_num
            if self.last_index == self.input_num:
                self.last_index = 0
        return output

class IterateProjectionExpandingSkip(SkipConnection):

    def __init__(self, input_num, output_num, scale=1):
        super(IterateProjectionExpandingSkip, self).__init__(scale)
        self.scale = scale
        self.input_num = input_num
        self.output_num = output_num
        if self.output_num % self.input_num !=0:
            raise RuntimeError('input_dim should be mutiples of output_dim')
        self.last_index = 0
        self.iterates = 0

    def _shortcut(self, input):
        # skip connection for linear layer with random projection
        # input: (N, output_num)
        # output: (N, input_num)
        input_size = input.size()
        output = torch.zeros(input_size[0],self.output_num,*input_size[2:]).type_as(input)
        output[:,self.last_index:(self.last_index+self.input_num)] = input

        self.last_index += self.input_num
        if self.last_index == self.output_num:
            self.last_index = 0
        
        return output