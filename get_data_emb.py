#pytorch
import torch
from PIL import Image
from torchvision import transforms
import torchvision

#other lib
import sys
import numpy as np
import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pickle

# model_embedded_face (insightface)
from insightface.insight_face import iresnet100
from yolov5_face.models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, scale_coords

sys.path.insert(0, "yolov5_face")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_emb = insight_face(path="insightface/ckpt_epoch_50.pth", device=device, train=True)
weight = torch.load("insightface/16_backbone.pth", map_location = device)
model_emb = iresnet100()
model_emb.load_state_dict(weight)
model_emb.to(device)
model_emb.eval()

face_preprocess = transforms.Compose([
                                    transforms.ToTensor(), # input PIL => (3,56,56), /255.0
                                    transforms.Resize((112, 112)),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                    ])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = attempt_load("yolov5_face/yolov5m-face.pt", map_location=device)

def resize_image(img0, img_size):
    h0, w0 = img0.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size
    img = letterbox(img0, new_shape=imgsz)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    
    return img
# data = {"names": name_images, "emb_vec": emb_images}

# 1. Đẩy toàn bộ ảnh trong database vô model InsightFace => database vectors
def inference_database(root_path = "faces_database"):
    
    images_name = []
    images_emb = []
    
    for folder in os.listdir(root_path):
        print(folder)
        if os.path.isdir(root_path + "/"+ folder):
            for name in os.listdir(root_path + "/" + folder):
                if name.endswith(("png", 'jpg', 'jpeg')):
                    path = f"{root_path}/{folder}/{name}"
                    
                    img_face = face_preprocess(Image.open(path).convert("RGB")).to(device)

                    with torch.no_grad():
                        emb_img_face = model_emb(img_face[None, :])[0].cpu().numpy()
                    
                    images_emb.append(emb_img_face)
                    images_name.append(name.split('.')[0])

    images_emb = np.array(images_emb)
    images_name = np.array(images_name)
    
    return images_name, images_emb/np.linalg.norm(images_emb, axis=1, keepdims=True)

name_images, emb_images = inference_database("faces_database")

with open('faces_emb_data.pkl', 'wb') as f:
    pickle.dump((name_images, emb_images), f)

# def get_user_name(model, model_emb):
# 	cap = cv2.VideoCapture(0)
# 	size_convert = 640 
# 	conf_thres = 0.4
# 	iou_thres = 0.5
# 	caption= "UN_KNOWN"

# 	while cap.isOpened():
# 		isSuccess, frame = cap.read()
# 		if isSuccess:
# 			img = resize_image(frame.copy(), size_convert)
# 			with torch.no_grad():
# 				pred = model(img[None, :])[0]

# 			# Apply NMS
# 			det = non_max_suppression_face(pred, conf_thres, iou_thres)[0]
# 			bboxs = np.int32(scale_coords(img.shape[1:], det[:, :4], frame.shape).round().cpu().numpy())
		    
# 			if len(bboxs) <= 0:
# 				continue

# 			faces = torch.zeros((len(bboxs), 3, 112, 112), dtype=torch.float32)
# 			for i in range(len(bboxs)):
# 				x1, y1, x2, y2 = bboxs[i]
# 				roi = frame[y1:y2, x1:x2]
# 				faces[i] = face_preprocess(Image.fromarray(roi))

# 			with torch.no_grad():
# 				emb_query = model_emb(faces.to(device)).cpu().numpy()
# 				emb_query = emb_query/np.linalg.norm(emb_query)

# 			# scores = np.linalg.norm(emb_query[:, None] - emb_images[None, :], axis=-1)
# 			# idxs = np.argmin(scores, axis=-1)
# 			# scores = np.sum(emb_query[:, None] * emb_images[None, :], axis=-1)
# 			scores = (emb_query @ emb_images.T)
# 			idxs = np.argmax(scores, axis=-1)


# 			for i in range(len(bboxs)):
# 				score = scores[i, idxs[i]] 
# 				name = name_images[idxs[i]]
# 				if score <= 0.15:
# 				# if score >= 1.3:
# 				    caption= "UN_KNOWN"
# 				else:
# 				    caption = name
		
# 		if caption != "UN_KNOWN": 
# 			print(caption)
# 			break

# 	cap.release()
# 	cv2.destroyAllWindows()
# 	return caption
