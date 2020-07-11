#996Coder

##2.Car Plate Rrecognition

### 安装包

`python -m pip install hyperlpr`

### Dependencies

- Keras (>2.0.0)
- Theano(>0.9) or Tensorflow(>1.1.x)
- Numpy (>1.10)
- Scipy (0.19.1)
- OpenCV(>3.0)
- Scikit-image (0.13.0)
- PIL
- Opencv (>3.4)

### Linux/Mac 编译

- 仅需要的依赖OpenCV 3.4 (需要DNN框架)

```bash
cd Prj-Linux
mkdir build 
cd build
cmake ../
sudo make -j 
```

##3.Car Identification

### Dependencies

- ubuntu/windows
- cuda>=10.0
- python>=3.6
- `pip3 install -r requirements.txt`