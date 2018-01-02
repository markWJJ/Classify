import numpy as np


class correlation(object):

    def __init__(self):
        pass


    def selfCorrelation(self,data,h):
        '''
        计算滞后h的自相关系数,
        :param data: list
        :param h: 
        :return: 
        '''
        data=np.array(data)
        data_avg=np.mean(data)
        data1=data[:-h]
        data2=data[h::]
        den=np.sum((data-data_avg)**2) #分母

        s=((data1-data_avg)*(data2-data_avg))/den
        corr=np.sum(s)
        print(corr)
        return corr

if __name__ == '__main__':
    data=[1,2,3,4,5,6,7,8,9,10]
    co=correlation()
    co.selfCorrelation(data,1)