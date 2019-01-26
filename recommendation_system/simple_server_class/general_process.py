#coding=utf-8
#author zhzcsp@gmail.com
#date 2019.01.20

import random

"""
#log_type notes
#1 点击
#2 播放
#3 点赞
#4 收藏
#5 付费观看
#6 站外分享
#7 评论
"""


"""
click_action = {} #key-uid, value-video_ids
def readfile():
    file = open("./logfile.txt", 'r')
    for line in file.readlines():
        ls = line.strip().split("&")
        if ls[7] != "1":
            continue
        #print "log_type:%s"%ls[7]
        print ls[1] + "\t" + ls[4]
        if click_action.has_key(ls[1]):
            click_action[ls[1]].append(ls[4])
        else:
            click_action[ls[1]] = []
            click_action[ls[1]].append(ls[4])


readfile()

"""

albet_Num = ["a","b", "c", "e", "f", "g", "h", "1", "2", "3", "4", "5", "6", "7", "A", "B", "O", "P", "Q"]
user_list = ["one", "two", "three", "four", "five"]
num = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
log_type_array = ["1", "2", "3", "4", "5", "6", "7"]

file_object = open('./logfile.txt', 'w')

def produce():
    for idx in range(0,2000):
        cookie = "".join(random.sample(albet_Num, 6))
        uid = "".join(random.sample(user_list, 1))
        user_agent = "Macintosh Chrome Safari"
        ip = "192.168.89.177"
        video_id = "".join(random.sample(num, 7))
        topic = "坚果发布会"
        order_id = "0"
        log_type = "".join(random.sample(log_type_array, 1))
        
        final = cookie + "&" + uid + "&" + user_agent + "&" + ip + "&" +video_id + "&" + topic + "&" + order_id + "&" + log_type + "\r\n"
        file_object.write(final)
        print final

produce()
file_object.close()

