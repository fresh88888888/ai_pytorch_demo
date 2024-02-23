"""Run ImageNet-pretrained ResNet18 on arbitrary image"""
import json
import torchvision.models as models
import torchvision.transforms as transforms
import torch
import sys
from PIL import Image

def get_idx_to_label():
    with open("imagenet_idx_to_label.json", encoding='utf-8') as f:
        return json.load(f)


def get_image_transform():
    transform = transforms.Compose([
        transforms.Resize(224),  # resize smaller side of image to 224
        transforms.CenterCrop(224),  # take center 224x224 crop
        # convert from image object to PyTorch tensor, which our PyTorch model needs
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # normalize images, according to https://pytorch.org/docs/stable/torchvision/models.html
                             std=[0.229, 0.224, 0.225])
    ])
    return transform


def load_image():
    assert len(sys.argv) > 1, 'Need to pass path to image'
    image = Image.open(sys.argv[1]).convert('RGB')

    # transform image into correct format
    transform = get_image_transform()
    image = transform(image)[None]
    return image

def predict(image):
    # load pretrained ResNet18 model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()  # set model in 'evaluation' mode

    # evaluate model on image
    out = model(image)

    # translate output into human-readable text
    _, pred = torch.max(out, 1)  # get class with highest probability
    idx_to_label = get_idx_to_label()  # get mapping from index to name
    cls = idx_to_label[str(int(pred))]  # get name, using index
    return cls


def main():
    'main function for entry'
    x = load_image()
    print(f'Prediction: {predict(x)}')


if __name__ == '__main__':
    main()
