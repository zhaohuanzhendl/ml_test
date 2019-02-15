#coding=utf-8

import math
from scipy.spatial import distance

class UserBasedCF:
    def __init__(self, train_file):
        """
        初始化对象
        """
        self.train_file = train_file
        self.readData()
        self.UserSimilarity()
    
    def readData(self):
        """
        读取文件，并生成物品-用户，测试集
        物品-用户的评分表
        训练集
        """
        self.train = {}
        # 打开文件，读取训练集
        for line in open(self.train_file):
            user, item, score, _ = line.strip().split("\t")
            self.train.setdefault(user, {})
            self.train[user][item] = int(score)
    
    def UserSimilarity(self):
        self.item_users = {}
        for user, items in self.train.items():
            for i in items.keys():
                if i not in self.item_users.keys():
                    self.item_users.setdefault(i, set())
                self.item_users[i].add(user)
        C = {}
        N = {}
        for i, users in self.item_users.items():
            # i-> item_id, users->set(['u1', 'u2', ...])
            for u in users:
                if u not in N.keys():
                    N.setdefault(u, 0)
                N[u] += 1
                if u not in C.keys():
                    C.setdefault(u, {})
                for v in users:
                    if u == v:
                        continue
                    if v not in C[u].keys():
                        C[u].setdefault(v, 0)
                    C[u][v] += 1
        self.W = {}
        self.Euc = dict()
        for u, related_users in C.items():
            if u not in self.W.keys():
                self.W.setdefault(u, {})
            for v, cuv in related_users.items():
                self.W[u][v] = cuv / math.sqrt(N[u] * N[v])
        return self.W
    
    def Recommend(self, user, K=3, N=10):
        rank = {}
        action_item = self.train[user].keys()
        for v, wuv in sorted(self.W[user].items(), key=lambda x:x[1], reverse=True)[0:K]:
            for i, rvi in self.train[v].items():
                if i in action_item:
                    continue
                if i not in rank.keys():
                    rank.setdefault(i, 0)
                rank[i] += rvi * wuv
        return sorted(rank.items(), key=lambda x:x[1], reverse=True)[0:N]
    
if __name__ == "__main__":
    cf = UserBasedCF("/Users/zhz/Downloads/ml-100k/u.data")
    print(cf.Recommend("3"))
