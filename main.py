import torch
from torchvision import transforms
from vgg19 import vgg19
from PIL import Image
import argparse


def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device)

parser = argparse.ArgumentParser()

parser.add_argument("--image_path", type = str)
parser.add_argument("--feature_layer", type = str)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

normalization_mean = [0.485, 0.456, 0.406]
normalization_std = [0.229, 0.224, 0.225]

loader  = transforms.Compose([transforms.ToTensor(), 
                                 transforms.Normalize(mean = normalization_mean, std = normalization_std)])

vgg = vgg19().to(device)
img = image_loader(args.image_path)
vgg_features = vgg(img)
feature = getattr(vgg_features, args.feature_layer)
