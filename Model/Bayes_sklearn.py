import sklearn
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
import numpy as np


class BayesSklearn(object):

    def __init__(self):
        self.wordDict={"None":0}
        self.bayes=MultinomialNB()
        self.svc=svm.SVC(C=5.0, kernel='rbf', degree=3, gamma='auto',max_iter=10000)
        self.kn=KNeighborsClassifier(5)

    def _data_deal(self,*args):
        '''
        数据预处理
        :param posPath: 
        :param negPath: 
        :return: 
        '''
        id = 1
        returnX=[]
        returnY=[]
        returnXWord=[]
        for index,path in enumerate(args):
            with open(path,'r') as p:
                for sentence in p:
                    sentence=sentence.strip()
                    returnXWord.append(sentence)
                    sentList=[]
                    for word in sentence.split(" "):
                        if word and word not in self.wordDict:
                            self.wordDict[word]=id
                            id+=1
                        if word:
                            sentList.append(self.wordDict[word])
                    returnX.append(sentList)
                    returnY.append(index)
        return returnX,returnY,returnXWord

    def _convert_array(self,X,max_len):
        '''
        将样本的X转换为矩阵
        :param X: 
        :return: 
        '''
        new_X=[]
        for ele in X:
            if len(ele)>=max_len:
                s=ele[:max_len]
            else:
                s=list(ele[:])
                ss=[0]*(max_len-len(ele))
                s.extend(ss)
            new_X.append(s)
        new_X_array=np.array(new_X)
        return new_X_array

    def Train(self,*args):
        '''
        训练
        :param args: 
        :return: 
        '''
        classifer = self.bayes
        returnX,returnY,returnXWord=self._data_deal(*args)
        # X_array=self._convert_array(returnX,8)
        Y_array=np.array(returnY)
        countvec=CountVectorizer()
        tfvec=TfidfVectorizer()
        X_array=tfvec.fit_transform(returnXWord)
        # X_array=tfvec.fit_transform(returnXWord)
        Xtrain,Xdev,Ytrain,Ydev=train_test_split(X_array,Y_array,test_size=0.2)


        classifer.fit(Xtrain,Ytrain)
        joblib.dump(classifer,"ClassiferModel.m")
        print("class acc:",classifer.score(Xtrain, Ytrain))
        print("class acc:",classifer.score(Xdev, Ydev))

    def predict(self):
        '''
        预测
        :param sentence: 
        :return: 
        '''
        classifer=joblib.load("./ClassiferModel.m")

        while True:
            sentence=input("输入：")
            tfvec = TfidfVectorizer()
            countvec = CountVectorizer()
            # X_array = tfvec.fit_transform(sentence)
            X_array = countvec.fit_transform(sentence)

            result=classifer.predict(X_array)
            print(result)


if __name__ == '__main__':

    bs=BayesSklearn()
    posPath="../Data/pos.txt"
    negPath="../Data/neg.txt"
    # bs.Train(posPath,negPath)
    bs.predict()



