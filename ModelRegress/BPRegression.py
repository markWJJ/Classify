import tensorflow as tf
from sklearn.cross_validation import train_test_split
import xlrd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
class BPRegression(object):

    def __init__(self,init_dim):
        self.dim=init_dim
        self.layer=2
        self.X=tf.placeholder(shape=(None,self.dim),dtype=tf.float32)
        self.Y=tf.placeholder(shape=(None,),dtype=tf.float32)

        w = tf.Variable(tf.random_uniform(shape=(self.dim, 1),
                                          dtype=tf.float32))
        # b=tf.Variable(tf.random_uniform(shape=(1,),dtype=tf.float32))
        # out = tf.add(tf.matmul(self.X, w), b)
        out=tf.matmul(self.X, w)

        # WList=[[self.dim,1000],[1000,1]]
        # out=self.X
        # for i in range(self.layer):
        #     with tf.variable_scope("nn%s"%i):
        #         w=tf.Variable(tf.random_uniform(shape=(WList[i][0],WList[i][1]),dtype=tf.float32))
        #         b=tf.Variable(tf.random_uniform(shape=(WList[i][1],),dtype=tf.float32))
        #
        label=tf.reshape(self.Y,(-1,1))
        self.loss=tf.reduce_mean(tf.abs(label-out))
        self.opt=tf.train.AdamOptimizer(0.3).minimize(self.loss)

    def TrianModel(self,X,Y,testX,testY):
        batch=20
        saver=tf.train.Saver()
        with tf.Session() as sess:
            # saver.restore(sess,'./model.ckpt')
            sess.run(tf.global_variables_initializer())
            init_loss=99999999999.99
            iter=int((X.shape[0] / batch))
            for _ in range(10000):
                for i in range(iter):
                    X_=X[i*batch:(i+1)*batch]
                    Y_=Y[i*batch:(i+1)*batch]

                    _,loss_=sess.run([self.opt,self.loss],feed_dict={self.X:X_,self.Y:Y_})
                    print(loss_)
                    if loss_<init_loss:
                        print('save')
                        init_loss=loss_
                        saver.save(sess,"./model.ckpt")
            testloss=sess.run(self.loss,feed_dict={self.X:testX,self.Y:testY})
            print("testLoss",testloss)




def LoadData(DataFilePath):

    excel=xlrd.open_workbook(DataFilePath)
    table=excel.sheets()[0]
    nrows = table.nrows # 行数
    ncols = table.ncols # 列数
    X=[]
    Y=[]
    for i in range(nrows):
        X.append(table.row_values(i)[:-1])
        Y.append(table.row_values(i)[-1])
    return X,Y

if __name__ == '__main__':

    X,Y=LoadData("./Data/ssad.xls")
    quadratic_featurizer = PolynomialFeatures(degree=2)  # 多项式回归模型的特征

    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2)
    TrainX=np.array(quadratic_featurizer.fit_transform(train_x))
    TestX=np.array(quadratic_featurizer.transform(test_x))
    test_y=np.array(test_y)
    train_y=np.array(train_y)
    bpreg = BPRegression(init_dim=TrainX.shape[1])

    bpreg.TrianModel(TrainX,train_y,TestX,test_y)


