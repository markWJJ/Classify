import xmlrpclib

s = xmlrpclib.ServerProxy('http://localhost:18112/RPC2')
sentence="我 最 爱喝 蒙牛 酸牛奶 天然 无害  助消化  乖宝宝   经常 喝 酸奶 "
res=s.getBayesClassifer(sentence)
print(res)