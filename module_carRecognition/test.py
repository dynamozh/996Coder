import os
import time

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .data.data_gen import data_transforms
from .models import CarRecognitionModel

if __name__ == '__main__':
    filename = 'weights/car_recognition.pt'
    model = CarRecognitionModel()
    model.load_state_dict(torch.load(filename, map_location='cpu'))
    # model = model.to(device)
    model.eval()

    transformer = data_transforms['valid']

    print('Predicting test data')
    num_samples = 1
    start = time.time()
    out = open('result.txt', 'w')
    for i in tqdm(range(num_samples)):
        full_path = os.path.join('data/test', '%05d.jpg' % (i + 1))  # data/test/00010.jpg
        img = Image.open(full_path)
        img.save('images/{}_out.png'.format(i))
        img = transformer(img)
        imgs = img.unsqueeze(dim=0)
        # imgs = imgs.to(device)

        with torch.no_grad():
            preds = model(imgs)

        preds = preds.cpu().numpy()[0]
        prob = np.max(preds)
        class_id = np.argmax(preds)
        out.write('{}\n'.format(str(class_id + 1)))

    end = time.time()
    seconds = end - start
    print('avg fps: {}'.format(str(num_samples / seconds)))

    out.close()
