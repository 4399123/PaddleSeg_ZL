#encoding=gbk
import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2
from imutils import paths
import os
from tqdm import tqdm

#路径配置
onnx_path=r'./onnx/best-smi.onnx'
# inputpath=r'./imgs'
inputpath=r'../../BlueFaceDataX2_PP/images'
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



for pic_path in tqdm(imgpaths):
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
    pred= out.squeeze()
    pred=cv2.resize(pred,(o_W,o_H),interpolation=cv2.INTER_NEAREST)
    predcpy=pred.copy()
    pred[pred>0]=1

    if (len(np.unique(out)) == 1): cv2.imwrite(os.path.join(outputpath,basename),imgbak)

    contours, hierarchy = cv2.findContours(pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for k in range(len(contours)):
        point = contours[k][0][0]
        x = point[0]
        y = point[1]
        id = int(predcpy[y][x])
        color = palette[id]
        line = np.array([contours[k]])
        cv2.drawContours(imgbak, line, -1, color=(int(color[0]), int(color[1]), int(color[2])), thickness=3)

    cv2.imwrite(os.path.join(outputpath, basename), imgbak)



