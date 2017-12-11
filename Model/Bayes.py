
import math
class NaiveBayes(object):
    '''
    朴素贝叶斯 情感二分类器
    '''
    def __init__(self,posPath,negPath):
        self.Dict,self.pos_num,self.neg_num=self._data_deal(posPath,negPath)
        self.PosY=(float(self.pos_num)+1.0)/(float(self.pos_num+self.neg_num)+2.0)
        self.NegY=(float(self.pos_num)+1.0)/(float(self.pos_num+self.neg_num)+2.0)
        # self.Train(posPath,negPath)

    def _data_deal(self,pos_path,neg_path):
        '''
        数据预处理，主要功能有：构建词典（因为bayes主要是对不同类别下的词语进行统计概率分析）
        :param pos_path: 
        :param neg_path: 
        :return: 
        '''
        # 构建一个词典
        Dict={"None":0}
        # 每个词的id 从1开始 0为‘None’代表未识别词
        id=1
        pos_num=0
        with open(pos_path,'r') as pos:
            for line in pos:
                pos_num+=1
                # 没行文本是以空各 分割的，并末尾有换行符
                line=line.replace("\n","")#去除换行符
                for word in line.split(" "):
                    # 先判断词word是否存在于字典中
                    if word and word not in Dict: # 如果word不在字典中
                        Dict[word]={"id":id,"pos_count":1,"neg_count":0,"sum_count":1}
                        id+=1
                    elif word in Dict:# 若word已经在dict中，则
                        value=Dict[word]
                        value["pos_count"]+=1
                        value["sum_count"]+=1
        neg_num=0
        with open(neg_path,'r') as neg:
            for line in neg:
                neg_num+=1
                # 没行文本是以空各 分割的，并末尾有换行符
                line=line.replace("\n","")#去除换行符
                for word in line.split(" "):
                    # 先判断词word是否存在于字典中
                    if word and word not in Dict : # 如果word不在字典中
                        Dict[word]={"id":id,"pos_count":0,"neg_count":1,"sum_count":1}
                        id+=1
                    elif word in Dict:# 若word已经在dict中，则
                        value=Dict[word]
                        value["neg_count"]+=1
                        value["sum_count"]+=1
        return Dict,pos_num,neg_num

    def Classify(self,sentence):
        '''
        分类器
        :param sentence: 
        :return: 
        '''
        sentence=str(sentence).strip().replace("\n","")
        pos_pro=1.0
        neg_pro=1.0
        for word in sentence.split(" "):
            if word in self.Dict:
                pos_neg=(float(self.Dict[word]["pos_count"]) + self.Dict[word]["neg_count"] + 2)
                pos_=float(self.Dict[word]["pos_count"]+1)/pos_neg
                neg_=float(self.Dict[word]["neg_count"]+1)/pos_neg
                pos_pro+=math.log(pos_)
                neg_pro+=math.log(neg_)
        fin_pos=math.log(self.PosY)+pos_pro
        fin_neg=math.log(self.NegY)+neg_pro
        if fin_pos>=fin_neg:
            print(fin_pos, fin_neg,1)
        else:
            print(fin_pos, fin_neg,0)

    def Train(self,pos_path,neg_path):
        '''
        bayes分类器训练模型
        :param pos_path: 
        :param neg_path: 
        :return: 
        '''
        with open(neg_path,'r') as f:
            for sent in f:
                self.Classify(sent)
    def Predict(self):
        '''
        从
        :return: 
        '''
        while True:
            sentence=input("输入句子")
            self.Classify(sentence)


if __name__ == '__main__':

    na=NaiveBayes('../Data/pos.txt','../Data/neg.txt')
    na.Predict()