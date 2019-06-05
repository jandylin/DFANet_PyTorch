from model.dfanet import DFANet
import numpy as np
import cv2
import torch


class DFANetPlugin(object):
    def __init__(self, im_height, im_width, use_cuda):
        super().__init__()
        self.name = "DFANet"
        self.im_height = im_height
        self.im_width = im_width
        self.use_cuda = use_cuda

        self.model = DFANet(pretrained=True)
        self.model.eval()
        if use_cuda:
            self.model.cuda()

    def process(self, image):
        x = cv2.resize(image, dsize=(1024, 1024))
        x = torch.from_numpy(x)
        # transpose to [1, 3, 1024, 1024] ?
        if self.use_cuda:
            x = x.cuda()
        mask = self.model(x).argmax(dim=1)
        if self.use_cuda:
            mask = mask.cpu()

        # TODO:
        # -
        raise ValueError("Process function of plugin {0} not yet implemented!".format(self.__class__.__name__))

    def release(self):
        """ To be implemented by the inheriting plugin.
            This function gets called when the plugin is turned off (e.g. switching to another plugin)
            Release your resources here! (e.g. free memory, close files ... etc.)
        """
        raise ValueError("Release function of plugin {0} not yet implemented!".format(self.__class__.__name__))