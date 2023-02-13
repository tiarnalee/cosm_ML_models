import torch
import torchvision.transforms as T

##########################################################################
#COSMETRICS 

class PermuteIm(object):
    def __call__(self, sample):
            img = sample['image']
            mask = sample['label']
                
            img=img.permute(2,0,1).float()/255
            mask=mask.permute(2,0,1).float()/255
            return {'image': img, 'label': mask}
    

class CenterCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):

        img = sample['image']
        mask = sample['label']

        img=T.CenterCrop(size=self.crop_size)(img)
        mask=T.CenterCrop(size=self.crop_size)(mask)

        return {'image': img,  'label': mask}

class Normalise(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std=std
        
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        img=T.Normalize(mean=self.mean, std=self.std)(img) #98, 80

        return {'image': img, 'label': mask}
