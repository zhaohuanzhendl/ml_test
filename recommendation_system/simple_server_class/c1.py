#coding=utf-8
#author: zhzcsp@gmail.com

import socket
import os

ss = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ss.connect(('127.0.0.1', 8888))

ss.sendall("one")
os.system('sleep 1')
ss.send('EOF')
data = ss.recv(1024)
print "server dafu: %s"%data
