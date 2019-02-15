#coding=utf-8

import math
class ItemBasedCF:
    
    def __init__(self, train_file):
        """
        初始化对象
        """
        self.train_file = train_file
        self.readData()
    
    def readData(self):
        """
        读取文件，并生成用户-物品，测试集
        用户-物品的评分表
        训练集
        """
        self.train = {}
        # 打开文件，读取训练集
        for line in open(self.train_file):
            user, item, score, _ = line.strip().split("\t")
            self.train.setdefault(user, {})
            self.train[user][item] = int(score)
    
    def ItemSimilarity(self):
        """
        计算物品之间的相似度
        """
        C = {}
        N = {}
        p_count = 0
        for user, items in self.train.items():
            p_count += 1
            for i in items.keys():
                #累加所有用户对每个物品各自的评分过的次数
                if i not in N.keys():
                    N.setdefault(i, 0)
                N[i] += 1 

                #一下循环物品与物品的共现矩阵+1处理
                if i not in C.keys():
                    C.setdefault(i, {})
                for j in items.keys():
                    if i == j: 
                        continue
                    if j not in C[i].keys():
                        C[i].setdefault(j, 0)
                    C[i][j] += 1
        #self.W = {}
        self.W = dict()
        for i, related_items in C.items():
            if i not in self.W.keys():
                self.W.setdefault(i, {})
            for j, cij in related_items.items():
                self.W[i][j] = cij / (math.sqrt(N[i] * N[j]))

        print "p_count:%s"%p_count
        return self.W
    
    def Recommend(self, user, K=3, N=10):
        """
        给用户推荐物品，取相似度最大的K个物品，推荐排名靠前的10个物品
        """
        # 用户对物品的偏好值
        rank = {}
        # 用户产生过行为的物品项和评分
        action_item = self.train[user]
        # print(action_item)
        for item, score in action_item.items():
            #print(sorted(self.W[item].items(), key=lambda x:x[1], reverse=True))
            #print("-----------------------------------------------------------------")
            #print(sorted(self.W[item].items(), key=lambda x:x[1], reverse=True)[0:K])
            for j, wj in sorted(self.W[item].items(), key=lambda x:x[1], reverse=True)[0:K]:
                if j in action_item.keys():
                    continue
                if j not in rank.keys():
                    rank.setdefault(j, 0)
                rank[j] += score * wj
        import pdb; pdb.set_trace()
        return sorted(rank.items(), key=lambda x:x[1], reverse=True)[0:N]

# 为用户3推荐未看过的10个电影
if __name__ == "__main__":
    cf = ItemBasedCF("/Users/zhz/Downloads/ml-100k/u.data")
    cf.ItemSimilarity()
    print(cf.Recommend("3"))
