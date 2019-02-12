#coding=utf-8
#author: zhzcsp@gmail.com
#create date: 2019/01/31

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import jieba

class Prefile():
    file_one = open("files_fiction/1")
    file_two = open("files_fiction/2")
    file_three = open("files_fiction/3")
    file_four = open("files_fiction/4")

    def __init__(self):
        rsu_1 = self.tag(self.file_one)
        self.save(rsu_1, "1")
        rsu_2 = self.tag(self.file_two)
        self.save(rsu_2, "2")
        rsu_3 = self.tag(self.file_three)
        self.save(rsu_3, "3")
        rsu_4 = self.tag(self.file_four)
        self.save(rsu_4, "4")

    def tag(self, file_x):
        print "\n", "_"*30
        print "object is ready!"
        result = {}
        result_filter = {}
        line_count = 0
        for line in file_x.readlines():
            line_count += 1
            #print line
            seg_list = jieba.cut(line.strip())
            #print ",".join(seg_list)
            for seg in seg_list:
                if not result.has_key(seg):
                    result[seg] = 1 
                else:
                    result[seg] += 1

        
        for k, v in result.items():
            if v <= 5:
                continue
            if k == u"的":
                continue
            result_filter[k] = v
            print k + "\t" + str(v)

        return result_filter


    def read(self,file):
        pass

    def buid(self, file):
        pass

    def save(self,re, fn):
        fo = open("./index_cut.txt", "aw")
        for k, v in  re.items():
            line = "filename :" + fn + "\t" + k.encode("utf-8") + "\t" + str(v) + "\r\n"
            fo.write(line)
        
        fo.close()


if __name__ == "__main__":
    p = Prefile()



