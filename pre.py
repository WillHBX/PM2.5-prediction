import json
import numpy as np

# fill empty value
def fill_x_data(data):
    start = -1
    for i in range(len(data)):
        if data[i] == 'x':
            if start == -1:
                start = i
        else:
            data[i] =  float(data[i])
            if start!= -1:
                for ind in range(start,i):
                    data[ind] = (data[start-1] + data[i]) / 2
                start = -1
    return data
    
    
#------------------------------training set----------------------------------------
data = []
for i in range(7,7+12):
    file = str(i%11+1)+'.json'
    with open(file,'r',encoding="utf-8") as inFile:
        raw_data = json.load(inFile) 

    data += [line['Concentration'] for line in raw_data['result']['records']]

data = fill_x_data(data)

data_set = np.array([data[i:i+4] for i in range(len(data)-4)])
trainx = np.array([data_set[i][:3] for i in range(data_set.shape[0])])
trainy = np.array([data_set[i][-1] for i in range(data_set.shape[0])])

# print(np.average(np.amax(trainx, axis = 1)))
# print(np.min(data))            
# print(data_set[:4])
# print(data_set.shape)
# print(train_x[:4])
# print(train_y[:4])
#----------------------------------------------------------------------------------

#------------------------------testing set-----------------------------------------
data = []
file = '_8.json'
with open(file,'r',encoding="utf-8") as inFile:
    raw_data = json.load(inFile) 

    data += [line['Concentration'] for line in raw_data['result']['records']]

data = fill_x_data(data)

data_set = np.array([data[i:i+4] for i in range(len(data)-4)])
testx = np.array([data_set[i][:3] for i in range(data_set.shape[0])])
testy = np.array([data_set[i][-1] for i in range(data_set.shape[0])])

# print(data_set[:4])
# print(data_set.shape)
# print(test_x[:4])
# print(test_y[:4])
#----------------------------------------------------------------------------------

np.savez('WanHua_PM2d5_data.npz', train_x = trainx, train_y = trainy, test_x = testx, test_y = testy)
      

#print(data)
