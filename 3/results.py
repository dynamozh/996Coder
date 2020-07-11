'''
auth:夏沁菡
create date:2020.7.11
update date:2020.7.11
discrip:
后端直接使用objects,outimg = car_quantity(img_BGR, CountArea)函数获取结果，img_BGR是通过cv2读取成BGR的图片
（渐本文件最后main函数），objects是检测出来的信息，形式如：
[['car', 0.539247453212738, [1290, 538, 74, 42]], ['car', 0.6105919480323792, [2034, 438, 65, 43]]]
在car_quantity(img_BGR, CountArea)函数体中，对加了框的图片进行保存
（文件名固定，所以每测试一次,记得将out_imgs中生成的图片删掉。）
'''

import predict
from config import *
import cv2
from models import *
from utils.utils import *
from utils.datasets import *

import matplotlib.pyplot as plt

data_config = "config/coco.data"
weights_path = "weights/yolov3.weights"
model_def = "config/yolov3.cfg"
permission = names
colorDict = color_dict
CountArea = np.array([[241, 182], [2408, 147], [2454, 1326], [124, 1322]])
data_config = parse_data_config(data_config)
yolo_class_names = load_classes(data_config["names"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_model = Darknet(model_def).to(device)
num_dict = {"bicycle": 0,"bus":0 , "car": 0,"motorbike":0 ,"truck":0}


def initModel():
    num_dict = {"bicycle": 0, "bus": 0, "car": 0, "motorbike": 0, "truck": 0}
    print("Loading model ...")
    if weights_path.endswith(".weights"):
        # Load darknet weights
        yolo_model.load_darknet_weights(weights_path)
    else:
        # Load checkpoint weights
        yolo_model.load_state_dict(torch.load(weights_path))

def pointInCountArea(painting, AreaBound, point):
    h,w = painting.shape[:2]
    point = np.array(point)
    point = point - AreaBound[:2]
    if point[0] < 0 or point[1] < 0 or point[0] >= w or point[1] >= h:
        return 0
    else:
        return painting[point[1],point[0]]

def filiter_out_repeat(objects):
    objects = sorted(objects,key=lambda x: x[1])
    l = len(objects)
    new_objects = []
    if l > 1:
        for i in range(l-1):
            flag = 0
            for j in range(i+1,l):
                x_i, y_i, w_i, h_i = objects[i][2]
                x_j, y_j, w_j, h_j = objects[j][2]
                box1 = [int(x_i - w_i / 2), int(y_i - h_i / 2), int(x_i + w_i / 2), int(y_i + h_i / 2)]
                box2 = [int(x_j - w_j / 2), int(y_j - h_j / 2), int(x_j + w_j / 2), int(y_j + h_j / 2)]
                if cal_iou(box1,box2) >= 0.7:
                    flag = 1
                    break
            #if no repeat
            if not flag:
                new_objects.append(objects[i])
        #add the last one
        new_objects.append(objects[-1])
    else:
        return objects

    return list(tuple(new_objects))


def cal_iou(box1,box2):
    x1 = max(box1[0],box2[0])
    y1 = max(box1[1],box2[1])
    x2 = min(box1[2],box2[2])
    y2 = min(box1[3],box2[3])
    i = max(0,(x2-x1))*max(0,(y2-y1))
    u = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) -  i
    iou = float(i)/float(u)
    return iou


def get_results(model,device, class_names,permission, frame, CountArea,colorDict):
    # painting area
    AreaBound = [min(CountArea[:, 0]), min(CountArea[:, 1]), max(CountArea[:, 0]), max(CountArea[:, 1])]
    painting = np.zeros((AreaBound[3] - AreaBound[1], AreaBound[2] - AreaBound[0]), dtype=np.uint8)
    CountArea_mini = CountArea - AreaBound[0:2]
    cv2.fillConvexPoly(painting, CountArea_mini, (1,))

    objects = predict.yolo_prediction(model, device, frame, class_names)
    objects = filter(lambda x: x[0] in permission, objects)
    objects = filter(lambda x: x[1] > 0.5, objects)
    objects = list(
        filter(lambda x: pointInCountArea(painting, AreaBound, [int(x[2][0]), int(x[2][1] + x[2][3] / 2)]), objects))

    # filter out repeat bbox
    objects = filiter_out_repeat(objects)

    # for item in objects:
    #     objectName = item[0]
    #     x=item[2][0]
    #     y = item[2][1]
    #     w = item[2][2]
    #     h = item[2][3]
    #     x1 = x- w//2
    #     y1 = y + h//2
    #     x2 = x + w // 2
    #     y2 = y - h // 2
    #     boxColor = colorDict[objectName]
    #     num_dict[objectName]= num_dict[objectName]+1
    #     outimg = cv2.rectangle(frame, (x1, y1), (x2, y2), boxColor, thickness=2)
    #     cv2.putText(frame, str(num_dict[objectName]) + "_" + objectName, (x1 - 1, y1 - 3), cv2.FONT_HERSHEY_COMPLEX, 0.7,
    #                 boxColor,
    #                 thickness=2)
    #     #save_file_path = os.path.join('out_imgs/', '%s.jpg' % (image_name))  # 本次保存图片jpg路径
    #     cv2.imwrite('out_imgs/outimg.jpg', outimg)

    return objects

def car_quantity(image,CountArea):
    initModel()
    objects = get_results(yolo_model, device, yolo_class_names, permission, image, CountArea,colorDict )
    print(objects)
    for item in objects:
        print(item)
        objectName = item[0]
        x = item[2][0]
        y = item[2][1]
        w = item[2][2]
        h = item[2][3]
        x1 = x - w // 2
        y1 = y + h // 2
        x2 = x + w // 2
        y2 = y - h // 2
        boxColor = colorDict[objectName]
        num_dict[objectName] = num_dict[objectName] + 1
        outimg = cv2.rectangle(image, (x1, y1), (x2, y2), boxColor, thickness=2)
        cv2.putText(image, str(num_dict[objectName]) + "_" + objectName, (x1 - 1, y1 - 3), cv2.FONT_HERSHEY_COMPLEX,
                    0.7, boxColor,thickness=2)
        # save_file_path = os.path.join('out_imgs/', '%s.jpg' % (image_name))  # 本次保存图片jpg路径
    cv2.imwrite('out_imgs/outimg.jpg', outimg)

    return objects,outimg


if __name__ == '__main__':
    img_BGR = cv2.imread('imgs/test1.jpg')
    # plt.subplot(2, 2, 1)
    # plt.imshow(img_BGR)
    # plt.axis('off')
    # plt.title('BGR')
    # plt.show()
    objects,outimg = car_quantity(img_BGR, CountArea)