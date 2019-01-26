#coding=utf-8
import socket

HOST, PORT = '127.0.0.1', 8888
print HOST, PORT
listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listen_socket.bind((HOST, PORT))
listen_socket.listen(1)

print 'Serving HTTP on port %s ...' %PORT

def Processing(request):
    uid = request.split("key=")[1].split(" ")[0]
    return uid

class Rec():
    def process(self, request):
        uid = request.split("key=")[1].split(" ")[0]
        return self.rec(uid)

    def rec(self, uid):
        rec = "123,123222333"
        return rec

    def gp(self, uid):
        click_action = {} #key uid, value video_ids
        file = open("./logfile.txt", 'r')
        for line in file.readlines():
            ls = line.strip().split("&")
            if ls[7] != "1":
                continue

            if ls[1] not in click_action.keys():
                click_action[ls[1]] = []
            click_action[ls[1]].append(ls[4])

        if uid in click_action.keys():
            return "&&".join(click_action[uid])


r = Rec()


while True:

    client_connection , client_address = listen_socket.accept()
    request = client_connection.recv(1024)
    http_response = r.gp(request)
    print http_response
    client_connection.sendall(http_response)
    client_connection.close()