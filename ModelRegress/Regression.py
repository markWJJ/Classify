import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import Ridge,SGDRegressor,Lasso,LassoCV
import xlrd
import logging
import matplotlib.pyplot as plt
import random
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')
_logger=logging.getLogger("Regression")
from abc import ABCMeta,abstractmethod

def runplt():
    plt.figure()
    plt.title(u'diameter-cost curver')
    plt.xlabel(u'diameter')
    plt.ylabel(u'cost')
    plt.axis([0, 25, 0, 25])
    plt.grid(True)
    return plt
class basicRegression(metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def LoadData(self,DataFilePath):
        '''
        从xls，csv等加载数据，
        输出为list[list[]]
        :return: 
        '''
        pass

    @abstractmethod
    def TainModel(self):
        '''
        训练模型，并将模型持久化保存
        :return: 
        '''
        pass

    @abstractmethod
    def Predict(cls,inputData):
        '''
        对输入数据 预测其值 
        :param inputData: 
        :return: 
        '''
        pass

class Correlation(object):
    '''
    相关性分析
    '''
    def __init__(self,X,Y):
        self.X=X
        self.Y=Y
        X_average, Y_average=self.ComputeAverageValue()


    def ComputeAverageValue(self):
        '''
        计算平均值
        :return: 
        '''
        arrayX=np.array(self.X)
        arrayY=np.array(self.Y)
        arrayX_=arrayX.T
        X_average=[]
        for e in arrayX_:
            X_average.append(np.mean(e))
        Y_average=np.mean(arrayY)
        return X_average,Y_average

    def ComputeCovVaule(self):
        '''
        计算协方差
        :return: 
        '''
        pass






class Regression(basicRegression):

    def __init__(self):
        self.regressor = LinearRegression()
        self.quadratic_featurizer = PolynomialFeatures(degree=2) # 多项式回归模型的特征
        self.RideRegress=Ridge(alpha=5)
        self.SGDregressor=SGDRegressor(loss='squared_loss',penalty='l1',max_iter=10000)
        self.lasso=LassoCV()
        self.X=None
        self.Y=None

    def LoadData(self,DataFilePath):

        excel=xlrd.open_workbook(DataFilePath)
        table=excel.sheets()[0]
        nrows = table.nrows # 行数
        ncols = table.ncols # 列数
        _logger.info("数据共有 %s 行， %s 列"%(nrows,ncols))
        self.X=[]
        self.Y=[]
        # 0 1 9 2 16 15 19 8 4 24
        # random_ncol=list(range(30))
        # random.shuffle(random_ncol)
        # ncolsList=random_ncol[0:10]
        # print(ncolsList)
        ncolsList=[0, 1, 9, 2, 16, 15, 19, 8, 4, 24, 14, 7, 11, 5, 28, 3, 13, 6, 10, 20, 17, 21, 18, 26, 29, 23, 12, 25, 22, 27]
        # ncolsList=[0, ]
        #
        all=[]
        for i in range(nrows):
            ss=[]
            for e in ncolsList:
                ss.append(table.row_values(i)[e])
            self.X.append(ss)
            # self.X.append(table.row_values(i)[:-2])
            # self.X.append(table.row_values(i)[:-2])
            self.Y.append(table.row_values(i)[-1])
            all.append(table.row_values(i))
        arrayAll=np.array(all)
        arrayAll=arrayAll.T
        correlation=np.corrcoef(arrayAll)
        corlist=[]
        for e in correlation:
            corlist.append(e[-1])

        ss=[]
        for index,ele in enumerate(corlist):
            ss.append([index,abs(ele)])
        ss.sort(key=lambda x:x[1],reverse=True)
        # print([s[0] for s in ss])
        print(ss)

        Xarray=np.array(self.X)
        Yarray=np.array(self.Y)
        X_1=np.linalg.pinv(Xarray)
        print(np.dot(X_1,Yarray))



    def TainModel(self,mode="Linear"):
        '''
        
        :return: 
        '''
        if mode=="Linear":
            train_x,test_x,train_y,test_y=train_test_split(self.X,self.Y,test_size=0.2)
            _logger.info("训练数据的size：%s"%len(train_x))
            _logger.info("测试数据的size：%s"%len(test_x))
            self.regressor.fit(train_x, train_y)
            yy = self.regressor.predict(test_x)
            for e,e1 in zip(test_y,yy):
                print(e,":",e1)
        elif mode=="UnLinear":
            train_x, test_x, train_y, test_y = train_test_split(self.X, self.Y, test_size=0.2)
            X_train=self.quadratic_featurizer.fit_transform(train_x)
            print(X_train.shape)
            X_test=self.quadratic_featurizer.transform(test_x)
            print(X_test.shape)
            self.regressor=self.lasso
            self.regressor.fit(X_train,train_y)
            predict_train=self.regressor.predict(X_train)
            predict_test=self.regressor.predict(X_test)

            train_sum=0.0
            for e,e1 in zip(train_y,predict_train):
                train_sum+=abs(e-e1)
            print(train_sum/float(len(train_y)))
            plt.plot(train_y,predict_train,"k.")
            plt.show()

            test_sum=0.0
            for e,e1 in zip(test_y,predict_test):
                test_sum+=abs(e-e1)
            print(test_sum/float(len(test_y)),len(test_y))
            #
            plt.plot(test_y,predict_test,"k.")
            plt.show()
        elif mode=="NuLinearRide":
            train_x, test_x, train_y, test_y = train_test_split(self.X, self.Y, test_size=0.2)
            X_train = self.quadratic_featurizer.fit_transform(train_x)
            X_test = self.quadratic_featurizer.fit_transform(test_x)

            self.RideRegress.fit(X_train, train_y)
            predict_train = self.RideRegress.predict(X_train)
            predict_test = self.RideRegress.predict(X_test)

            train_sum = 0.0
            for e, e1 in zip(train_y, predict_train):
                train_sum += abs(e - e1)
            print(train_sum / float(len(train_y)))
            plt.plot(train_y, predict_train, "k.")
            plt.show()

            test_sum = 0.0
            for e, e1 in zip(test_y, predict_test):
                test_sum += abs(e - e1)
            print(test_sum / float(len(test_y)), len(test_y))

            plt.plot(test_y, predict_test, "k.")
            plt.show()







    def Predict(cls,inputData):
        pass

if __name__ == '__main__':
    regree=Regression()
    regree.LoadData("./Data/ssad.xls")
    regree.TainModel(mode="UnLinear")


# X_train = [[6], [8], [10], [14], [18]]
# y_train = [[7], [9], [13], [17.5], [18]]
# X_test = [[6], [8], [11], [16]]
# y_test = [[8], [12], [15], [18]]
# # 建立线性回归，并用训练的模型绘图
# regressor = LinearRegression()
# regressor.fit(X_train, y_train)
# xx = np.linspace(0, 26, 100)
# yy = regressor.predict(xx.reshape(xx.shape[0], 1))
#
# # plt = runplt()
# # plt.plot(X_train, y_train, 'k.')
# # plt.plot(xx, yy)
#
# quadratic_featurizer = PolynomialFeatures(degree=2)
# X_train_quadratic = quadratic_featurizer.fit_transform(X_train)
# X_test_quadratic = quadratic_featurizer.transform(X_test)
# regressor_quadratic = LinearRegression()
# regressor_quadratic.fit(X_train_quadratic, y_train)
# xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))
# print(xx_quadratic)
# # plt.plot(xx, regressor_quadratic.predict(xx_quadratic), 'r-')
# # plt.show()
# # print(X_train)
# # print(X_train_quadratic)
# # print(X_test)
# # print(X_test_quadratic)
# # print('1 r-squared', regressor.score(X_test, y_test))
# # print('2 r-squared', regressor_quadratic.score(X_test_quadratic, y_test))