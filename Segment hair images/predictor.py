import numpy as np
import cv2
from models.deeplab import *
from  datagen_utils import denormalize_image
import torch
from my_transforms1 import *
from torchvision import transforms 

class Predictor():
    def __init__(self, config,  checkpoint_path='./snapshots/best.pth.tar'):
        self.config = config
        self.crop_size=self.config['image']['crop_size']
        self.checkpoint_path = checkpoint_path
        self.model = self.load_model()


    def load_model(self):
        model = DeepLab(num_classes=self.config['network']['num_classes'], backbone=self.config['network']['backbone'],
                        output_stride=self.config['image']['out_stride'], sync_bn=False, freeze_bn=True)

        if self.config['network']['use_cuda']:
            checkpoint = torch.load(self.checkpoint_path)
        else:
            checkpoint = torch.load(self.checkpoint_path, map_location={'cuda:0': 'cpu'})

        model = torch.nn.DataParallel(model)

        model.load_state_dict(checkpoint['state_dict'])

        return model

    def preprocess(self, sample, crop_size, mean, std):
        composed_transforms = transforms.Compose([
            PermuteIm(),
            CenterCrop(crop_size),
            Normalise(mean, std),
            ])
        return composed_transforms(sample)

    def segment_image(self, filename):
    
            img = torch.from_numpy(cv2.imread(filename)).float()
    
            sample = {'image': img, 'label': img}
    
            sample = self.preprocess(sample, self.config['image']['crop_size'], mean = (0.3387, 0.3794, 0.4425),std = (0.3025, 0.3124, 0.3339))
            image = sample['image']
            image = image.unsqueeze(0)

            with torch.no_grad():
                image=image.float()

                prediction = self.model(image)
    
            image = image.squeeze(0).numpy()
            image = denormalize_image(np.transpose(image, (1, 2, 0)))
            image *= 255.
    
            prediction = prediction.squeeze(0).cpu().numpy()
    
            prediction = np.argmax(prediction, axis=0)
    
            return image, prediction