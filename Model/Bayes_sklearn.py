import sklearn
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
import numpy as np
import pickle


class BayesSklearn(object):

    def __init__(self):
        self.wordDict={"None":0}
        self.bayes=MultinomialNB()
        self.svc=svm.SVC(C=5.0, kernel='rbf', degree=3, gamma='auto',max_iter=10000)
        self.knn=KNeighborsClassifier()
        pickle.dump(self.wordDict,open("wordDict.p",'wb'))
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
        Y_array=np.array(returnY)
        classiferMode="bayes"

        if classiferMode=="bayes":
            countvec=CountVectorizer()
            tfvec=TfidfVectorizer()
            X_array=tfvec.fit_transform(returnXWord)
            # X_array=tfvec.fit_transform(returnXWord)
            Xtrain,Xdev,Ytrain,Ydev=train_test_split(X_array,Y_array,test_size=0.2)
            classifer.fit(Xtrain,Ytrain)
            joblib.dump(classifer,"bayesClassiferModel.m")
            joblib.dump(tfvec,"bayesTfidvector.m")
            print("bayes train acc:",classifer.score(Xtrain, Ytrain))
            print("bayes test acc:",classifer.score(Xdev, Ydev))

        elif classiferMode=="svm":
            X_array=self._convert_array(returnX,8)
            Xtrain,Xdev,Ytrain,Ydev=train_test_split(X_array,Y_array,test_size=0.2)
            self.svc.fit(Xtrain,Ytrain)
            joblib.dump(self.svc,"svmClassiferModel.m")
            print("svm train acc:", self.svc.score(Xtrain, Ytrain))
            print("svm test acc:", self.svc.score(Xdev, Ydev))

        elif classiferMode=="knn":
            X_array = self._convert_array(returnX, 8)
            Xtrain, Xdev, Ytrain, Ydev = train_test_split(X_array, Y_array, test_size=0.2)
            self.knn.fit(Xtrain, Ytrain)
            joblib.dump(self.svc,"knnClassiferModel.m")
            print("knn train acc:", self.knn.score(Xtrain, Ytrain))
            print("knn test acc:", self.knn.score(Xdev, Ydev))


    def predict_pre_deal(self,sentence,max_len):
        '''
        预测模块预处理
        :param sentence: 
        :return: 
        '''
        wordDict=pickle.load(open("./wordDict.p",'rb'))
        sentences=sentence.strip().split(" ")
        sentId=[]
        for word in sentences:
            if word in wordDict:
                sentId.append(wordDict[word])
            else:
                sentId.append(0)
        if len(sentId)>=max_len:
            new_sent=sentId[:max_len]
        else:
            new_sent=sentId[:]
            new_sent.extend([0]*(max_len-len(sentId)))
        new_sent=np.array(new_sent)
        return new_sent

    def predict(self):
        '''
        预测
        :param sentence: 
        :return: 
        '''
        mode="bayes"
        classifer=joblib.load("./bayesClassiferModel.m")
        tfvec=joblib.load("./bayesTfidvector.m")
        print(tfvec)

        while True:
            sentence=input("输入：")
            if mode=="bayes":
                # tfvec = TfidfVectorizer()
                # countvec = CountVectorizer()
                # X_array = tfvec.fit_transform(sentence)
                X_array = tfvec.fit_transform(sentence)

                result=classifer.predict(X_array)
                print(result)

            # X_vec=self.predict_pre_deal(sentence,8)
            # X_vec=np.reshape(X_vec,[1,X_vec.shape[0]])
            # print(classifer.predict(X_vec))



if __name__ == '__main__':

    bs=BayesSklearn()
    posPath="../Data/pos.txt"
    negPath="../Data/neg.txt"
    # bs.Train(posPath,negPath)
    bs.predict()



