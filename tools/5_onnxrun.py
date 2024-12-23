#encoding=gbk
import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2
from imutils import paths
import os

#路径配置
onnx_path=r'./onnx/best-smi.onnx'
inputpath=r'./imgs'
outputpath=r'./results'

if not os.path.exists(outputpath):
    os.makedirs(outputpath)


imgpaths=list(paths.list_images(inputpath))


w,h=512,512

#调色板配置
palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)
palette[0]=[255,255,255]
palette[1]=[0,255,0]
palette[2]=[0,0,255]
palette[3]=[255,0,0]
palette[4]=[255,255,0]
palette[5]=[255,0,255]
palette[6]=[171,130,255]
palette[7]=[155,211,255]
palette[8]=[0,255,255]

mean=(120,114,104)
std=(70,69,73)

#onnx模型载入
model = onnx.load(onnx_path)
onnx.checker.check_model(model)
session = ort.InferenceSession(onnx_path,providers=['CPUExecutionProvider'])



for pic_path in imgpaths:
    img_input=cv2.imread(pic_path,1)
    basename=os.path.basename(pic_path)
    o_H,o_W = img_input.shape[0],img_input.shape[1]
    imgbak=img_input.copy()
    img_input=img_input[:,:,::-1]
    img_input=cv2.resize(img_input,(w,h)).astype(np.float32)
    img_input-=mean                             #减均值
    img_input/=std                              #除方差
    img_input=np.array([np.transpose(img_input,(2,0,1))])

    #模型推理
    out = session.run(None,input_feed = { 'input' : img_input })
    # out=np.argmax(out[0],axis=1)
    out=out[0].astype('uint8')
    pred= palette[out].squeeze()
    pred=cv2.resize(pred,(o_W,o_H),interpolation=cv2.INTER_NEAREST)


    #保存图像
    n=0
    # cv2.imwrite('./onnx/mask_{}.jpg'.format(n), pred)

    img1=np.array(imgbak)
    # img1=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)

    img=cv2.addWeighted(img1,0.3,pred,0.7,0)
    cv2.imwrite(os.path.join(outputpath,basename),img)


