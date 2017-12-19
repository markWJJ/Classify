import tensorflow as tf
from Model_DL import Data_deal
import numpy as np
import time
class Classifer(object):

    def __init__(self,vocab_size,init_dim,hidden_dim,max_len):
        self.init_dim=init_dim
        self.hidden_dim=hidden_dim
        self.max_len=max_len
        self.vacab_size=vocab_size
        self.numclass=2
        self.initModel()
        logit = self.Encoder()
        self.softlogit,self.loss=self.Softmax(logit)
        self.optim=tf.train.AdamOptimizer(0.7).minimize(self.loss)

    def initModel(self):
        '''
        构建图
        :return: 
        '''
        self.input=tf.placeholder(shape=(None,self.max_len),dtype=tf.int32)
        self.input_seq=tf.placeholder(shape=(None,),dtype=tf.int32)
        self.Y=tf.placeholder(shape=(None,),dtype=tf.int32)
        self.embeding=tf.Variable(tf.random_uniform(shape=(self.vacab_size,self.init_dim),minval=0.0,maxval=1.0,dtype=tf.float32))
        self.input_array=tf.nn.embedding_lookup(self.embeding,self.input)

    def Encoder(self):
        '''
        编码层，将词向量通过lstm 或者 cnn 进行编码
        :return: 
        '''
        encoderMode="cnn"
        input_list=tf.unstack(self.input_array,self.max_len,1)
        if encoderMode=="lstm":
            with tf.variable_scope("lstm"):
                cell=tf.contrib.rnn.LSTMCell(self.hidden_dim,
                                             initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
                                             state_is_tuple=False
                                             )
                out,state=tf.contrib.rnn.static_rnn(cell,input_list,dtype=tf.float32,sequence_length=self.input_seq)
                return out[-1]
        elif encoderMode=="bilstm":
            bilstm_hidden_dim=int(self.hidden_dim/2) # 为了保持encoder的维度一致，将hiddendim减半
            with tf.variable_scope("bilstm"):
                fw_cell=tf.contrib.rnn.LSTMCell(bilstm_hidden_dim,
                                               initializer=tf.random_uniform_initializer(-0.1,0.1),
                                               state_is_tuple=False)
                bw_cell = tf.contrib.rnn.LSTMCell(bilstm_hidden_dim,
                                                 initializer=tf.random_uniform_initializer(-0.1, 0.1),
                                                 state_is_tuple=False)
                (out,fw_state,bw_state)=\
                    tf.contrib.rnn.static_bidirectional_rnn(fw_cell,bw_cell,input_list,dtype=tf.float32,sequence_length=self.input_seq)
                return out[-1]
        elif encoderMode=="cnn":
            cnn_input=tf.expand_dims(self.input_array,-1) #[batch_size,hin_height, in_width, in_channels]
            filters1=tf.Variable(tf.random_uniform([3,self.max_len,1,10],dtype=tf.float32))
            strides1=[1,1,1,1]
            bias1=tf.Variable([10,],dtype=tf.float32)
            con=tf.nn.relu(tf.nn.conv2d(cnn_input,filters1,strides=strides1,padding="SAME")+bias1)
            con_pool=tf.nn.max_pool(con,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
            # 全链接层
            pull_con=tf.reshape(con_pool,[-1,15*25*10])

            # 隐含层
            f1=tf.Variable(tf.random_uniform(shape=[15*25*10,self.hidden_dim],dtype=tf.float32))
            b1=tf.Variable([self.hidden_dim,],dtype=tf.float32)
            out=tf.matmul(pull_con,f1)+b1
            return out

    def Softmax(self,logit):
        '''
        分类
        :param logit: 
        :return: 
        '''
        weight=tf.Variable(tf.random_uniform((self.hidden_dim,self.numclass),dtype=tf.float32))
        bias=tf.Variable(tf.constant(0.0,shape=(self.numclass,),dtype=tf.float32))

        logit=tf.add(tf.matmul(logit,weight),bias)
        soft_logit=tf.nn.softmax(logit)
        label_hot=tf.one_hot(self.Y,self.numclass,1.0,0.0,dtype=tf.float32)
        loss=tf.losses.softmax_cross_entropy(label_hot,soft_logit)
        return soft_logit,loss


    def Train(self,dd):

        config = tf.ConfigProto(device_count={"CPU": 8},  # limit to num_cpu_core CPU usage
                                inter_op_parallelism_threads=8,
                                intra_op_parallelism_threads=8,
                                log_device_placement=False)
        X_all,X_seq_all,Y_all=dd.X_array,dd.X_seq_array,dd.Y_array
        saver = tf.train.Saver()
        with tf.Session(config=config) as sess:
            init_loss = 999
            init_acc = 0.0
            saver.restore(sess,"./model.ckpt")
            # sess.run(tf.global_variables_initializer())
            for i in range(5000):
                X_batch,Y_batch,X_seq_batch=dd.next_batch()
                soft,loss,_=sess.run([self.softlogit,self.loss,self.optim],feed_dict={self.input:X_batch,
                                                                  self.input_seq:X_seq_batch,
                                                                  self.Y:Y_batch})
                soft=np.argmax(soft,1)
                acc=float(sum([1 for e,e1 in zip(soft,Y_batch) if e==e1]))/float(len(Y_batch))
                if i%100==0:
                    if loss<init_loss and acc>init_acc:

                        init_loss=loss
                        init_acc=acc
                        print(loss, acc)
                        saver.save(sess,"model.ckpt")
                        print("save")
            softAll=sess.run(self.softlogit,feed_dict={self.input:X_all,
                                               self.input_seq:X_seq_all,
                                               self.Y:Y_all})
            softAll=np.argmax(softAll,1)
            accAll=float(sum([1 for e,e1 in zip(softAll,Y_all) if e==e1]))/float(len(Y_all))
            print(accAll)




if __name__ == '__main__':
    startTime=time.time()
    max_len=30
    batch_size=20
    dd = Data_deal.DataDeal(negTrainPath='../Data/neg.txt', posTrainPath="../Data/pos.txt",
                  negdevPath="../Data/neg_dev.txt", posdevPath="../Data/pos_dev.txt",
                  flag="train_new", max_len=max_len, batch_size=batch_size)
    vocab_size=len(dd.vocab)
    init_dim=50
    hidden_dim=100
    classifer=Classifer(vocab_size=vocab_size,init_dim=init_dim,hidden_dim=hidden_dim,max_len=max_len)
    classifer.Train(dd)
    endTime=time.time()
    print("all Time",endTime-startTime)
