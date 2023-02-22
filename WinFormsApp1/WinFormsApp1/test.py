import socket

HOST = "192.168.1.204"
SERVER_PORT = 65432

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


s.bind((HOST, SERVER_PORT))
s.listen()

conn, addr = s.accept()

