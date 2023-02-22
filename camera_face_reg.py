#pytorch
import torch
#other lib
import numpy as np
import cv2
from PIL import Image
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, scale_coords
import time

def resize_image(img0, img_size, model, device):
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

def get_user_name(model, model_emb, face_preprocess, device, name_images, emb_images):
	cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
	size_convert = 640 
	conf_thres = 0.4
	iou_thres = 0.5
	caption= "UN_KNOWN"
	pT = 0
	cT = 0
	while cap.isOpened():
		isSuccess, frame = cap.read()
		if isSuccess:
			start = time.time()
			img = resize_image(frame.copy(), size_convert, model, device)
			with torch.no_grad():
				pred = model(img[None, :])[0]

			# Apply NMS
			det = non_max_suppression_face(pred, conf_thres, iou_thres)[0]
			bboxs = np.int32(scale_coords(img.shape[1:], det[:, :4], frame.shape).round().cpu().numpy())

			if len(bboxs) <= 0:
				continue

			faces = torch.zeros((len(bboxs), 3, 112, 112), dtype=torch.float32)
			for i in range(len(bboxs)):
				x1, y1, x2, y2 = bboxs[i]
				roi = frame[y1:y2, x1:x2]
				faces[i] = face_preprocess(Image.fromarray(roi))

			with torch.no_grad():
				emb_query = model_emb(faces.to(device)).cpu().numpy()
				emb_query = emb_query/np.linalg.norm(emb_query)
			scores = (emb_query @ emb_images.T)
			idxs = np.argmax(scores, axis=-1)

			for i in range(len(bboxs)):
				score = scores[i, idxs[i]] 
				name = name_images[idxs[i]]
				if score <= 0.4:
				# if score >= 1.3:
				    caption= "UN_KNOWN"
				else:
				    caption = name.split("_")[0]
		cT = time.time()
		fps = int(1/(cT-pT))
		pT = cT
		cv2.putText(frame, str(fps), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                   1, (255, 0, 0), 2, cv2.LINE_AA)
		if caption != "UN_KNOWN":
			print(caption)
			break
		cv2.imshow('frame', frame)
		cv2.waitKey(1)
	cap.release()
	cv2.destroyAllWindows()
	return caption
