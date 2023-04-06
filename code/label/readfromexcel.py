import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import h5py

df = pd.read_excel('label.xlsx')
df3 = df.iloc[0:665,1:16]

data1 = df3['one'].str.split(",")
data2 = df3['two'].str.split(",")
data3 = df3['three'].str.split(",")
data4 = df3['four'].str.split(",")
data5 = df3['five'].str.split(",")
data6 = df3['six'].str.split(",")
data7 = df3['seven'].str.split(",")
data8 = df3['eight'].str.split(",")
data9 = df3['nine'].str.split(",")
data10 = df3['ten'].str.split(",")
data11 = df3['eleven'].str.split(",")
data12 = df3['twelve'].str.split(",")
data13 = df3['thirteen'].str.split(",")
data14 = df3['fourteen'].str.split(",")
data15 = df3['fifteen'].str.split(",")

data = pd.concat([data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11,data12,data13,data14,data15],axis=0)
a = np.array(data)


#####onehot

one_hot = MultiLabelBinarizer()
#print(one_hot.fit_transform(a))
onehot = one_hot.fit_transform(a)
#print(onehot)
#print(onehot) #(915, 7)

onehot1=onehot[0:61,:]
onehot2=onehot[61:122,:]
onehot3=onehot[122:183,:]
onehot4=onehot[183:244,:]
onehot5=onehot[244:305,:]
onehot6=onehot[305:366,:]
onehot7=onehot[366:427,:]
onehot8=onehot[427:488,:]
onehot9=onehot[488:549,:]
onehot10=onehot[549:610,:]
onehot11=onehot[610:671,:]
onehot12=onehot[671:732,:]
onehot13=onehot[732:793,:]
onehot14=onehot[793:854,:]
onehot15=onehot[854:915,:]

#print(onehot1.shape)
#print(onehot2.shape)
com = np.array([onehot1,onehot2,onehot3,onehot4,onehot5,onehot6,onehot7,onehot8,onehot9,onehot10,onehot11,onehot12,onehot13,onehot14,onehot15])

print(com.shape)  #(15, 61, 7)

y = np.transpose(com, (1,0,2))
print(y.shape)
#onehot1 = np.reshape(onehot,(61,2,6))
#print(onehot1)

with h5py.File('./label.h5', 'w') as hf:
    hf.create_dataset("dataset",  data=y)


