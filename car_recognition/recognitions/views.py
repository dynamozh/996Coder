from django.shortcuts import render

# Create your views here.

def index(request):
    return render(request,'index.html')

def cardetection(request):
    return render(request, 'cardetection.html')
