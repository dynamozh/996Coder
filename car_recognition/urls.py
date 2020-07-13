"""car_recognition URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, re_path
from recognitions import views as rec_views
from django.views.static import serve        # 图片显示
from pathlib import Path

base_path = Path(__file__).parent

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', rec_views.index),
    path('index/', rec_views.index, name="index"),
    path('index/cardetection/', rec_views.cardetection, name="cardetection"),
    path('index/carplaterecog/',rec_views.carplaterecog, name="carplaterecog"),
    re_path('media/(?P<path>.*)$', serve, {'document_root': str((base_path / "/media").resolve())}),
    # 这句意思是将访问的图片href由“http://127.0.0.1:0000/media/图片存储文件夹/字母哥.jpg”转为本地访问D:\workspace\upload_pic\media的形式
]
