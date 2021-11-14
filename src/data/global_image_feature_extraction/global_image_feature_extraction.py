
import os, sys
from tqdm import tqdm
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import h5py, time

from PIL import Image
import src.commons.globals as glb

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'


def create_model(name):
    if name == 'vgg':
        model = models.vgg16(pretrained=True).features[:29]
    elif name == 'resnet':
        backbone = models.resnet152(pretrained=True)
        model = nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
                backbone.layer2,
                backbone.layer3,
                backbone.layer4[0],
                backbone.layer4[1],
                backbone.layer4[2].conv1,
                backbone.layer4[2].bn1,
                backbone.layer4[2].conv2,
                backbone.layer4[2].bn2,
                backbone.layer4[2].conv3,
                backbone.layer4[2].bn3,
        )
    return model


def extract_feature(model, images_path, image_feature_path):
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    print("image path: ",images_path)
    print("image_feature_path: ",image_feature_path)
    img_feature = h5py.File(image_feature_path, 'w')
    images_files = [file for file in os.listdir(images_path) if file.endswith('.jpg')]

    print(images_files[:10])

    start = time.time()
    print("Processing {} images".format(len(images_files)))

    processed_count = 0
    for item in tqdm(images_files):
        try:
            img_path = images_path + item
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).view(1, 3, 224, 224)
        except:
            print(img_path)
        result = model(tensor)
        result = result.data.clone().cpu().numpy()
        img_feature.create_dataset(name=item, data=result[0])
        processed_count += 1
    
    print("Processed images: {} / {}".format(processed_count, len(images_files)))
    print("Feature extraction time: {:.1f}s".format(time.time() - start))


if __name__ == "__main__":

    model = create_model('vgg')    
    extract_feature(model, glb._AAAI_IMAGE_TRAIN, glb._AAAI_VGG_IMAGE_FEATURE_TRAIN)
    extract_feature(model, glb._AAAI_IMAGE_DEV, glb._AAAI_VGG_IMAGE_FEATURE_DEV)
    extract_feature(model, glb._AAAI_IMAGE_TEST, glb._AAAI_VGG_IMAGE_FEATURE_TEST)

    extract_feature(model, glb._TWITTER_IMAGE_TRAIN, glb._TWITTER_VGG_IMAGE_FEATURE_TRAIN)
    extract_feature(model, glb._TWITTER_IMAGE_DEV, glb._TWITTER_VGG_IMAGE_FEATURE_DEV)
    extract_feature(model, glb._TWITTER_IMAGE_TEST, glb._TWITTER_VGG_IMAGE_FEATURE_TEST)

    model = create_model('resnet')    
    extract_feature(model, glb._AAAI_IMAGE_TRAIN, glb._AAAI_RESNET_IMAGE_FEATURE_TRAIN)
    extract_feature(model, glb._AAAI_IMAGE_DEV, glb._AAAI_RESNET_IMAGE_FEATURE_DEV)
    extract_feature(model, glb._AAAI_IMAGE_TEST, glb._AAAI_RESNET_IMAGE_FEATURE_TEST)

    extract_feature(model, glb._TWITTER_IMAGE_TRAIN, glb._TWITTER_RESNET_IMAGE_FEATURE_TRAIN)
    extract_feature(model, glb._TWITTER_IMAGE_DEV, glb._TWITTER_RESNET_IMAGE_FEATURE_DEV)
    extract_feature(model, glb._TWITTER_IMAGE_TEST, glb._TWITTER_RESNET_IMAGE_FEATURE_TEST)
    
    