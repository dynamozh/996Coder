# import the necessary packages
"""
auth: 周昱程
create date: 2020.7.12
update date: 2020.7.12
discrip:
后端直接使用results = run(img)函数获取结果
results是检测出来的信息，形式如：
[{'label': 'Eagle Talon Hatchback 1998', 'color': 'Green', 'prob': '0.9912'}]
"""

import numpy as np
import scipy.io
import torch
from PIL import Image
import matplotlib.pyplot as plt

# from .data.data_gen import data_transforms
# from .models import CarRecognitionModel
# from .colorClassifier import color_Classifier
# Yu-cheng Computer import path
from data.data_gen import data_transforms
from models import CarRecognitionModel
from colorClassifier import color_Classifier


def run(img):
    """
            Recognition car
            :param: JpegImageFile
            :return: List[Dict[str, str, str]]
    """
    filename = 'weights/car_recognition.pt'
    model = CarRecognitionModel()
    model.load_state_dict(torch.load(filename, map_location='cpu'))
    model.eval()

    transformer = data_transforms['valid']

    cars_meta = scipy.io.loadmat('data/devkit/cars_meta')
    class_names = cars_meta['class_names']  # shape=(1, 196)
    class_names = np.transpose(class_names)

    results = []
    color = color_Classifier(img)

    img = transformer(img)
    imgs = img.unsqueeze(dim=0)

    with torch.no_grad():
        preds = model(imgs)

    preds = preds.cpu().numpy()[0]
    prob = np.max(preds)
    class_id = np.argmax(preds)
    results.append({'label': class_names[class_id][0][0], 'color': color, 'prob': '{:.4}'.format(prob)})

    return results

if __name__ == '__main__':
    img = Image.open('./images/samples/00033.jpg')

    res = run(img)
    print(res)
