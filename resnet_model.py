import cv2
import numpy as np
import torch
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
import json
from ultralytics import YOLO


man_class_name = ['bea_man', 'din_man', 'dog_man', 'rab_man', 'cat_man']
woman_class_name = ['dog_woman', 'fox_woman', 'rab_woman', 'cat_woman', 'dee_woman']

woman_model = torch.load('woman85.pth', map_location=torch.device('cpu'))
man_model = torch.load('man85.pth', map_location=torch.device('cpu'))
woman_model.eval()
man_model.eval()

face_model= YOLO('yolov8n-face.pt')

app = Flask(__name__)

def detect(image):
    box = face_model(image)[0].boxes
    if len(box.conf) == 0:
	return 0 ,0
    xyxy = box.xyxy[0].tolist()
    image = image[int(xyxy[1]): int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
    
    return image, len(box.conf)
    
def transform(image):
    my_transforms = transforms.Compose([transforms.ToPILImage(),
                                      transforms.Resize((512, 512)),
                                        transforms.ToTensor()])
    return my_transforms(image).unsqueeze(0)

def get_woman_prediction(image):
    tensor = transform(image=image)
    outputs = woman_model(tensor)
    _, predicted = torch.max(outputs.data, 1)
    
    return predicted

def get_man_prediction(image):
    tensor = transform(image=image)
    outputs = man_model(tensor)
    _, predicted = torch.max(outputs.data, 1)
    
    return predicted


@app.route('/man', methods=["POST"])
def man():
    file = request.files['image']
    img_byte = file.read()
    encoded_img = np.fromstring(img_byte, dtype = np.uint8)
    img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    face, num = detect(image = img)
    
    if num != 1:
        return jsonify({'class_name': None})
    
    class_id = get_man_prediction(image=face)
    class_id = class_id.tolist()
    return jsonify({'class_name': man_class_name[class_id[0]]})
 
    
@app.route('/woman', methods=["POST"])
def woman():
    file = request.files['image']
    img_byte = file.read()
    encoded_img = np.fromstring(img_byte, dtype = np.uint8)
    img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    
    face, num = detect(image = img)
    
    if num != 1:
        return jsonify({'class_name': None})

    class_id = get_woman_prediction(image=face)
    class_id = class_id.tolist()
    return jsonify({'class_name': woman_class_name[class_id[0]]})
 
@app.route('/check', methods=["POST"])
def check():
    for i in range(1,len(request.files) + 1):
        file = request.files['image' + str(i)]
        print(i)
        img_byte = file.read()
        encoded_img = np.fromstring(img_byte, dtype = np.uint8)
        img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
        face, num = detect(image = img)
        if num == 0:
            return jsonify({'class_name': None})
    
    return jsonify({'class_name' : "true"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000)