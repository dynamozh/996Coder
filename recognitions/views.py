from django.shortcuts import render
from recognitions.models import PREIMG
from recognitions.models import IMG
from django.conf import settings
from PIL import Image
import module_carDetection.demo as d1
import module_carPlateRecognition.demo as d2
import os             # 创建文件夹需要的包
import cv2
import numpy

# Create your views here.


def index(request):
    return render(request, 'index.html')

# def cardetection(request):
#     if request.method == 'POST':
#         new_img = PREIMG(img=request.FILES.get('img'),name="default")
#         new_img.save()
#         old_img = PREIMG.objects.get(name="default")
#         content = {'img' : old_img}
#         return render(request,'cardetection.html',content)
#     else:
#         return render(request, 'cardetection.html')


def cardetection(request):
    if request.method == 'POST':
        img = request.FILES.get('img')
        path = settings.MEDIA_ROOT
        file = 'pre_images'
        file2 = 'images'
        pic_path = path + '/' + file
        pic_path2 = path + '/' + file2

        #上传图片部分
        isExists = os.path.exists(pic_path)
        if isExists:
            print("目录已存在")
        else:
            os.mkdir(pic_path)
            print("创建成功")
        img_url = pic_path + '/' + "inimg.jpg"
        with open(img_url, 'wb') as f:      #将上传图片以二进制形式写入
            for data in img.chunks():
                f.write(data)
        pic_data = 'http://127.0.0.1:8000/media' + '/' + file + '/' + "inimg.jpg"   #将路径转化一下，转为href形式，然后存入数据库，发到后端
        #PREIMG.objects.create(img=pic_data)             #写入数据库

        # 图像识别处理部分
        isExists2 = os.path.exists(pic_path2)
        if isExists2:
            print("目录已存在")
        else:
            os.mkdir(pic_path2)
            print("创建成功")
        img_BGR = cv2.imread(img_url)
        CountArea = numpy.array([[241, 182], [2408, 147], [2454, 1326], [124, 1322]])          #识别区域
        the_objects, the_outimg, the_num_dict = d1.run(img_BGR, CountArea)
        img_url2 = pic_path2 + '/' + "outimg.jpg"
        pic_data2 = 'http://127.0.0.1:8000/media' + '/' + file2 + '/' + "outimg.jpg"   #将路径转化一下，转为href形式，然后存入数据库，发到后端
        #IMG.objects.create(img=pic_data2)             #写入数据库

        #识别结果返回网页
        return render(request, 'cardetection.html', {'img_url' : pic_data2 ,'objects':the_objects,'dic':the_num_dict,'area':CountArea})
    else:
        return render(request,'cardetection.html',{'img_url' : '/static/desert-mountain-road-sunset.jpg'})


def carplaterecog(request):
    if request.method == 'POST':
        img = request.FILES.get('img')
        path = settings.MEDIA_ROOT
        file = 'pre_images'
        pic_path = path + '/' + file

        #上传图片部分
        isExists = os.path.exists(pic_path)
        if isExists:
            print("目录已存在")
        else:
            os.mkdir(pic_path)
            print("创建成功")
        img_url = pic_path + '/' + "inimg.jpg"          #取同名以覆盖上传
        with open(img_url, 'wb') as f:      #将上传图片以二进制形式写入
            for data in img.chunks():
                f.write(data)
        pic_data = 'http://127.0.0.1:8000/media' + '/' + file + '/' + "inimg.jpg"   #将路径转化一下，转为href形式，然后存入数据库，发到后端
        #PREIMG.objects.create(img=pic_data)             #写入数据库

        # 图像识别处理部分
        pre_img = Image.open(img_url)
        result = d2.run(pre_img)

        #识别结果返回网页
        return render(request, 'carplaterecog.html',{'resultList' : result,'img_url' : pic_data})
    else:
        return render(request,'carplaterecog.html',{'img_url' : "/static/desert-mountain-road-sunset.jpg"})