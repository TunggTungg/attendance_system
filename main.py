import pandas as pd
from datetime import datetime, date
from camera_face_reg import get_user_name
import torch
from torchvision import transforms
import pickle
import sys
from insightface.insight_face import iresnet100
from models.experimental import attempt_load
import time, playsound, os, threading
from gtts import gTTS
import socket

def export_data():
    # Data Export
    filename = str(time.strftime("%d-%m-%y")) + '.xlsx'
    # Append DataFrame to existing excel file
    with pd.ExcelWriter(filename, mode='a+') as writer:
        df.to_excel(writer, index=False)

    # Socket send FILE
    SEPARATOR = "<SEPARATOR>"
    BUFFER_SIZE = 4096  # send 4096 bytes each time step

    # the ip address or hostname of the server, the receiver
    host = "192.168.43.218"
    # the port, let's use 5001
    port = 65432
    # get the file size
    filesize = os.path.getsize(filename)

    # create the client socket
    s = socket.socket()

    print(f"[+] Connecting to {host}:{port}")
    s.connect((host, port))
    print("[+] Connected.")

    # send the filename and filesize
    s.send(f"{filename}{SEPARATOR}{filesize}".encode())

    # start sending the file
    with open(filename, "rb") as f:
        while True:
            # read the bytes from the file
            bytes_read = f.read(BUFFER_SIZE)
            if not bytes_read:
                # file transmitting is done
                break
            # we use sendall to assure transimission in
            # busy networks
            s.sendall(bytes_read)
            # update the progress bar

def bella_speak(text):   # Bella speak to customers
    tts = gTTS(text=text, lang='vi')
    tts.save("sound.mp3")
    playsound.playsound("sound.mp3")
    os.remove("sound.mp3")

def bella_speak_thread(text):
    speak = threading.Thread(target=bella_speak, args=(text,))
    speak.start()

# Set Up
sys.path.insert(0, "yolov5_face")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weight = torch.load("insightface/16_backbone.pth", map_location=device)
model_emb = iresnet100()
model_emb.load_state_dict(weight)
model_emb.to(device)
model_emb.eval()

face_preprocess = transforms.Compose([
    transforms.ToTensor(),  # input PIL => (3,56,56), /255.0
    transforms.Resize((112, 112)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = attempt_load("yolov5_face/yolov5s-face.pt", map_location=device)

# Load data
with open('faces_emb_data.pkl', 'rb') as f:
    (name_images, emb_images) = pickle.load(f)


# Define Variables
df = pd.DataFrame(columns = ['ID','Name','Time In','Time Out','Status'])
user = []
id = ["Toan", "An", "Chien", "Chuong", "Doan", "Duc", "Duy", "Huy", "Hào", "Nhật", "Phát"
    ,"Phong", "Thanh", "Thái", "Thịnh", "Toán", "Thang", "Trực", "Tung", "Vu", "Vy"]

# Mark Hours
now = datetime.now()
mark = now.replace(hour=18, minute=54, second=0, microsecond=0)


# Input to continue

while(1):
    if(input("Press a key to continue: ") == "q"):
        # Get Name
        name = get_user_name(model, model_emb, face_preprocess, device, name_images, emb_images).capitalize()

        # Track
        if(name not in user):

            bella_speak_thread("Chào " + name + " một ngày tốt lành")

            # Append Name
            user.append(name)

            # Get Time In
            now = datetime.now()

            # Append Data
            temp_row = {'ID': id.index(name), 'Name': name, 'Time In': now.strftime("%H:%M:%S"), 'Time Out': "", 'Status': u'\u2717'}
            df = df.append(temp_row, ignore_index=True)
        else:
            # Find Index
            index = df.index[df['Name'] == name].tolist()

            # Compare and Replace Data
            # Get Time In
            now = datetime.now()
            if(now >= mark):
                bella_speak_thread("Một ngày làm việc vất vã rồi " + name + " nhỉ, hẹn gặp lại bạn vào ngày mai")
                df.loc[index[0], ['Time Out', 'Status']] = [now.strftime("%H:%M:%S"), u'\u2714']
            else:
                df.loc[index[0], ['Time Out', 'Status']] = [now.strftime("%H:%M:%S"), u'\u2717']
                bella_speak_thread("Ây da hôm nay" + name + " vẫn chưa đủ KPI, ngày mai tiếp tục cố gắng nhé")

            user.remove(name)
        # export_data()
        print(df)


