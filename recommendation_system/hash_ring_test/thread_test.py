#coding=utf-8
#

import threading
import random
from time import ctime, sleep

test = {"1":10}
lock = 0
def read():
    print 'read'
    while True:
        if lock == 0 :
            for k, v in test.items():
                print k + "\t" + str(v)
            print "read succ"
            break
        else:
            print "writing...."
            sleep(1)

def write():
    print 'write'
    while True:
        if lock == 0:
            lock = 1
            test[str(int(random.random()*10))] = int(random.random()*10) * 10
            lock = 0
            print "write succ"
        else:
            print "wait! writine..."
            sleep(1)


threads = []
t1 = threading.Thread(target=read, args=())
threads.append(t1)
t2 = threading.Thread(target=write, args=())
threads.append(t2)

for t in threads:
    #t.setDaemon(True)
    t.start()
