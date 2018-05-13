import torch.nn as nn

from utils import *
import numpy as np

class unet(nn.Module):

    def __init__(self, no_input=1, feature_scale=4, n_classes=3, is_deconv=True, in_channels=3, is_batchnorm=True, gpu_ids=[]):
        super(unet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
	self.no_input = no_input
	self.gpu_ids = gpu_ids
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
	self.conv1= []
	self.conv2 = []
	for i in range(self.no_input):
	        self.conv1.append(unetConv2(self.in_channels, filters[0], self.is_batchnorm))
		# 128
        	self.conv2.append(unetConv2(filters[0], filters[1], self.is_batchnorm))
	# 64
        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
	# 32
        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
	# 16
        self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm)
	# 8
        self.conv6 = unetConv2(filters[4], filters[4], self.is_batchnorm)
	# 4
        self.conv7 = unetConv2(filters[4], filters[4], self.is_batchnorm)
	# 2
        self.conv8 = unetConv2(filters[4], filters[4], self.is_batchnorm)
	#1

        # upsampling
        self.up_concat8 = unetUp(filters[4], filters[4], self.is_deconv)
	#2
        self.up_concat7 = unetUp(filters[4], filters[4], self.is_deconv)
	#4
        self.up_concat6 = unetUp(filters[4], filters[4], self.is_deconv)
	#8
        self.up_concat5 = unetUp(filters[4], filters[3], self.is_deconv)
	#16
        self.up_concat4 = unetUp(filters[3], filters[2], self.is_deconv)
	#32
        self.up_concat3 =nn.Conv2d(filters[2], filters[1], 1)
	#64
        self.up_concat2 = nn.Conv2d(filters[1], filters[0], 1)
	#128
        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

    def forward(self, inputs1, inputs2):
	print(inputs1.size())
	print(inputs2.size())

	conv10=self.conv1[0](inputs1)
	conv11=self.conv1[1](inputs2)
	conv20=self.conv2[0](conv10)
	conv21=self.conv2[1](conv20)
	conv3_ = torch.concat([conv20, conv21])
        conv3 = self.conv3(conv3_)        
	conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        conv7 = self.conv7(conv6)        
	conv8 = self.conv8(conv7)
	up7 = self.up_concat7(conv7, conv8)
        up6 = self.up_concat6(conv6, up7)
	up5 = self.up_concat5(conv5, up6)
        up4 = self.up_concat4(conv4, up5)
	up3 = self.up_concat3(conv3, up4)
	up2 = self.up_concat2(up3)
        up1 = self.up_concat1(up2)
        final = self.final(up1)

        return final
