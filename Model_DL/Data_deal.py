import numpy as np
import pickle
import os
global PATH
from sklearn.utils import shuffle
PATH=os.path.split(os.path.realpath(__file__))[0]

class DataDeal(object):
    def __init__(self,negTrainPath,posTrainPath,negdevPath,posdevPath,max_len,flag,batch_size):
        self.negTrainPath=negTrainPath
        self.posTrainPath=posTrainPath
        self.negdevPath=negdevPath
        self.posdevPath=posdevPath
        self.batch_size=batch_size
        self.max_len = max_len
        if flag=="train_new":
            self.vocab=self._get_vocab(self.negTrainPath,self.posTrainPath,
                                       self.negdevPath,self.posdevPath)
            pickle.dump(self.vocab,open(PATH+"/vocab.p",'wb'))
        elif flag=="test" or flag=="train":
            self.vocab=pickle.load(open(PATH+"/vocab.p",'rb'))
        self.index=0
        self.getAllarray()
    def _get_vocab(self,*args):
        '''
        构造字典
        :return: 
        '''
        vocab={"NONE":0}
        for path in args:
            with open(path,'r') as file:
                index=1
                for ele in file:
                    for word in ele.strip().split(" "):
                        word=word.lower()
                        if word and word not in vocab:
                            vocab[word]=index
                            index+=1
        return vocab

    def _sent2vec(self,sent,max_len):
        '''
        根据vocab将句子转换为向量
        :param sent: 
        :return: 
        '''

        sent=str(sent).strip()
        sent_list=[]
        real_len=len(sent.split(" "))
        for word in sent.split(" "):
            word=word.lower()
            if word and word in self.vocab:
                sent_list.append(self.vocab[word])
            else:
                sent_list.append(0)
        if len(sent_list)>=max_len:
            new_sent_list=sent_list[0:max_len]
            real_len=max_len
        else:
            new_sent_list=sent_list
            ss=[0]*(max_len-len(sent_list))
            new_sent_list.extend(ss)
        sent_vec=np.array(new_sent_list)
        return sent_vec,real_len

    def _shuffle(self,Q,A,label):
        '''
        将矩阵X打乱
        :param x: 
        :return: 
        '''
        ss=list(range(Q.shape[0]))
        np.random.shuffle(ss)
        new_Q=np.zeros_like(Q)
        new_A=np.zeros_like(A)
        new_label=np.zeros_like(label)
        for i in range(Q.shape[0]):
            new_Q[i]=Q[ss[i]]
            new_A[i]=A[ss[i]]
            new_label[i]=label[ss[i]]

        return new_Q,new_A,new_label

    def getAllarray(self):
        X_list = []
        Y_list = []
        X_seq_list = []
        file_size = 0
        with open(self.posTrainPath, 'r') as file:
            for sentence in file:
                file_size+=1
                Y_list.append(1)
                # 获取句子的id表示，并规范长度，不够的由0代替
                X_vec, X_len = self._sent2vec(sentence, self.max_len)
                X_list.append(X_vec)
                X_seq_list.append(X_len)
        with open(self.negTrainPath, 'r') as file:
            for sentence in file:
                file_size+=1
                Y_list.append(0)
                # 获取句子的id表示，并规范长度，不够的由0代替
                X_vec, X_len = self._sent2vec(sentence, self.max_len)
                X_list.append(X_vec)
                X_seq_list.append(X_len)
        self.file_size=file_size
        X_array = np.array(X_list)
        X_seq_array = np.array(X_seq_list)
        Y_array = np.array(Y_list)
        self.X_array, self.X_seq_array, self.Y_array = shuffle(X_array, X_seq_array, Y_array)

    def next_batch(self):
        '''
        获取训练机的下一个batch
        :return: 
        '''

        num_iter=int(self.file_size/self.batch_size)
        if self.index<=num_iter:
            X_arrat_batch=self.X_array[self.index*self.batch_size:(self.index+1)*self.batch_size]
            Y_array_batch=self.Y_array[self.index*self.batch_size:(self.index+1)*self.batch_size]
            X_seq_array_batch=self.X_seq_array[self.index*self.batch_size:(self.index+1)*self.batch_size]
            self.index+=1
        else:
            self.index=0
            X_arrat_batch=self.X_array[0:self.batch_size]
            Y_array_batch=self.Y_array[0:self.batch_size]
            X_seq_array_batch = self.X_seq_array[0:self.batch_size]
        return X_arrat_batch,Y_array_batch,X_seq_array_batch

    def get_dev(self):
        '''
        读取验证数据集
        :return: 
        '''
        dev_file = open(self.dev_path, 'r')
        Q_list = []
        A_list = []
        label_list = []
        train_sentcens = dev_file.readlines()
        for sentence in train_sentcens:
            sentences=sentence.split("	")
            Q_sentence=sentences[0]
            A_sentence=sentences[1]
            label=sentences[2]
            Q_array=self.sent2array(Q_sentence,self.Q_len)
            A_array=self.sent2array(A_sentence,self.A_len)

            Q_list.append(list(Q_array))
            A_list.append(list(A_array))
            label_list.append(int(label))
        dev_file.close()
        result_Q=np.array(Q_list)
        result_A=np.array(A_list)
        result_label=np.array(label_list)
        return result_Q,result_A,result_label

    def get_test(self):
        '''
        读取测试数据集
        :return: 
        '''
        test_file = open(self.test_path, 'r')
        Q_list = []
        A_list = []
        label_list = []
        train_sentcens = test_file.readlines()
        for sentence in train_sentcens:
            sentences=sentence.split("	")
            Q_sentence=sentences[0]
            A_sentence=sentences[1]
            label=sentences[2]
            Q_array,_=self.sent2array(Q_sentence,self.Q_len)
            A_array,_=self.sent2array(A_sentence,self.A_len)

            Q_list.append(list(Q_array))
            A_list.append(list(A_array))
            label_list.append(int(label))
        test_file.close()
        result_Q=np.array(Q_list)
        result_A=np.array(A_list)
        result_label=np.array(label_list)
        return result_Q,result_A,result_label


if __name__ == '__main__':
    dd = DataDeal(negTrainPath='../Data/neg.txt',posTrainPath="../Data/pos.txt",
                  negdevPath="../Data/neg_dev.txt",posdevPath="../Data/pos_dev.txt",
                  flag="train_new",max_len=30,batch_size=4)

    X,Y,X_seq=dd.nex_batch()
