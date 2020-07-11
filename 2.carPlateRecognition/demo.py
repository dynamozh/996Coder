"""
auth: 周昱程
create date: 2020.7.11
update date: 2020.7.11
discrip:
后端直接使用result = run(img)函数获取结果
result是检测出来的信息，形式如：[['皖AQ4025', 0.96576008626392906, [296, 340, 449, 401]]]
"""

# 导入包
from hyperlpr import *
from PIL import Image
import numpy as np


def run(img):
    """
            Identify car plate
            :param: JpegImageFile
            :return: List[List[Union[str, float, list]]]
    """
    data = np.array(img)
    result = HyperLPR_plate_recognition(data)
    return result


if __name__ == '__main__':
    image = Image.open("")
    result = run(image)
    print(result)