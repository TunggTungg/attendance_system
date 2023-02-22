import cv2
import time
import glob, os
from facenet_pytorch import MTCNN
import torch

def main():
    device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(thresholds= [0.7, 0.7, 0.8] ,keep_all=True, device = device)

    for file in glob.glob("./data/*.mp4"):
        path = './faces_database/' + file[7:-4] + '/'
        print(file[7:-4])
        os.mkdir(path)
        cap = cv2.VideoCapture(file)
        # Resolution 640*480
        time.sleep(1)
        if cap is None or not cap.isOpened():
            print('Khong the mo file video')
            return
        cv2.namedWindow('Image', cv2.WINDOW_AUTOSIZE);
        n = 0
        while True:
            [success, img] = cap.read()
            ch = cv2.waitKey(30)
            if success:
                boxes, _ = mtcnn.detect(img)
                try: 
                    img = img[int(boxes[0][1]):int(boxes[0][3]),int(boxes[0][0]):int(boxes[0][2])]
                    img = cv2.resize(img,(150,150))
                    cv2.imshow('Image', img)
                    filename = path + file[7:-4]+ '_' + str(n) +'.jpg'
                    cv2.imwrite(filename,img)
                    n = n + 1
                except:
                    pass
            else:
                break

            if n > 10:
                break
    return

if __name__ == "__main__":
    main()
