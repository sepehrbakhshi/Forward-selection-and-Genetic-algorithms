import numpy as np
import random
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix,f1_score
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sys
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
"""
loadedData_train = np.loadtxt("train_data.txt")


counter_1 = 0
counter_2 = 0
counter_3 = 0

first_data = []
second_data = []
third_data = []
loadedData_train = np.array(loadedData_train)
for i in range(0 , loadedData_train.shape[0]):
    if(loadedData_train[i,21] == 1):
        for i in range(0,22):
            first_data.append(loadedData_train[i])
            counter_1+=1
    elif(loadedData_train[i,21] == 2):
        for i in range(0,20):
            second_data.append(loadedData_train[i])
            counter_2+=1
    elif(loadedData_train[i,21] == 3):
        third_data.append(loadedData_train[i])
        counter_3+=1
first_data = np.array(first_data)
second_data =  np.array(second_data)
third_data = np.array(third_data)
print(first_data.shape)
print(second_data.shape)
print(third_data.shape)
train_data = np.concatenate((first_data,second_data))
train_data = np.concatenate((train_data,third_data))
print(train_data.shape)
random.shuffle(train_data)

x_train = train_data[slice(None),0:21]
x_train = np.array(x_train)

labels_train = train_data[(slice(None),21)]
labels_train = np.array(labels_train)
print(x_train.shape)
print(labels_train.shape)


loadedData_test = np.loadtxt("test_data.txt")

counter_1 = 0
counter_2 = 0
counter_3 = 0

first_data = []
second_data = []
third_data = []

for i in range(0 , loadedData_test.shape[0]):
    if(loadedData_train[i,21] == 1):
        for i in range(0,22):
            first_data.append(loadedData_test[i])
            counter_1+=1
    elif(loadedData_train[i,21] == 2):
        for i in range(0,20):
            second_data.append(loadedData_test[i])
            counter_2+=1
    elif(loadedData_train[i,21] == 3):
        third_data.append(loadedData_test[i])
        counter_3+=1
first_data = np.array(first_data)
second_data =  np.array(second_data)
third_data = np.array(third_data)

print(first_data.shape)
print(second_data.shape)
print(third_data.shape)

test_data = np.concatenate((first_data,second_data))
test_data = np.concatenate((test_data,third_data))

random.shuffle(test_data)

x_test = test_data[slice(None),0:21]
x_test = np.array(x_test)


labels_test = test_data[(slice(None),21)]
labels_test = np.array(labels_test)
"""
trainData_ = np.loadtxt(fname="train_data.txt")
testData = np.loadtxt(fname="test_data.txt")
train = trainData_.copy()
test = testData.copy()
def oversampler(data,factor):
    ones_twos = np.zeros((0,22))
    for z in data:
        if (z[21] == 2 or z[21] == 1):
            ones_twos = np.vstack([ones_twos,z])

    newData = data
    for _ in range(factor):
        newData = np.vstack([newData,ones_twos])

    random.shuffle(newData)
    return newData

oversamplingFactor = 30
trainData = oversampler(trainData_, oversamplingFactor)
testData = oversampler(testData, oversamplingFactor)

X_train = trainData[:, :-1]
y_train = trainData[:, -1]
X_test = testData[:, :-1]
y_test = testData[:, -1]
counter_1 = 0
counter_2 = 0
counter_3 = 0
for i in range(0,y_test.shape[0] ):
    if(y_train[i] == 1):
        counter_1 += 1
    if(y_train[i] == 2):
        counter_2 += 1
    if(y_train[i] == 3):
        counter_3 += 1

loadedData_cost = pd.read_csv('cost.txt', delimiter= '\s+', header=None)
#loadedData_cost = pd.read_csv('cost.txt', delimiter = )
loadedData_cost = np.array(loadedData_cost)
loadedData_cost = loadedData_cost[(slice(None),1)]
model = DecisionTreeClassifier()

model = DecisionTreeClassifier()
print(X_train.shape)
print(loadedData_cost.shape)
def forward_feature_selection(x_train, x_cv, y_train, y_cv, n):
    feature_set = []
    for num_features in range(n):
        cost = 0
        metric_list = [] # Choose appropriate metric based on business problem
        model = SGDClassifier() # You can choose any model you like, this technique is model agnostic
        k = 0
        for i in range(0 ,  x_train.shape[1]):
            if i not in feature_set:
                f_set = feature_set.copy()
                f_set.append(i)
                if(i == 20 and (18 in feature_set) and (19 in feature_set)):
                    print("first")
                    cost = 0
                elif(i == 20 and (18 in feature_set) and (19 not in feature_set)):
                    print("second")
                    cost =  loadedData_cost[19]
                elif(i == 20 and (18 not in feature_set) and (19 in feature_set)):
                    print("third")
                    cost = loadedData_cost[18]
                elif(i == 20 and (18 not in feature_set) and (19 not in feature_set)):
                    print("fourth")
                    cost =  loadedData_cost[18] + loadedData_cost[19]

                    print(cost)
                else:
                    cost =  loadedData_cost[i]
                model.fit(x_train[:,f_set], y_train)
                y_pred = model.predict( x_cv[:,f_set])
                acc = accuracy_score(y_cv, y_pred)
                acc = acc
                cost = cost/76.11
                fitness = (acc - (cost/ (acc + 1))) + 76.11
                metric_list.append((fitness , i))
                    #metric_list.append((evaluate_metric(model, x_cv[:,f_set], y_cv,cost), i))

        metric_list.sort(key=lambda x : x[0], reverse = True) # In case metric follows "the more, the merrier"
        print(metric_list)
        feature_set.append(metric_list[0][1])
    return feature_set

#mlp = MLPClassifier(alpha=1e-2,hidden_layer_sizes=(100), random_state=1)
f = forward_feature_selection(X_train, X_test, y_train, y_test, 5)
model = DecisionTreeClassifier()
model.fit(X_train[:,f], y_train)
y_pred = model.predict( X_test[:,f])
acc = accuracy_score(y_test, y_pred)
print(f)
print("total acc")
print(acc)
y_pred = model.predict( X_train[:,f])
acc = accuracy_score(y_train, y_pred)
print("train acc")
print(acc)
total_cost = 0
for i in range(0,len(f)):
    if((f[i] == 20) and (18 in f) and (19 in f)):
        total_cost = total_cost + 0
    elif((f[i] == 20) and (18 in f) and (19 not in f)):
        cost =  total_cost + loadedData_cost[19]
    elif((f[i] == 20) and (18 not in f) and (19 in f)):
        cost = total_cost + loadedData_cost[18]
    elif((f[i] == 20) and (18 not in f) and (19 not in f)):
        cost =  total_cost + loadedData_cost[18] + loadedData_cost[19]
    else:

        total_cost = total_cost + loadedData_cost[f[i]]
print(f)
print("total acc")
print(acc)
print(total_cost)
first_class = []
for i in range(0, testData.shape[0]):
    if (testData[i,21] == 1):
        first_class.append(testData[i])
first_class = np.array(first_class)
first_class_x = first_class[: , 0:21]
first_class_x = np.array(first_class_x)
print(first_class_x.shape)
first_class_y = first_class[: , 21]
first_class_y = np.array(first_class_y)
y_pred = model.predict( first_class_x[:,f])
acc = accuracy_score(first_class_y, y_pred)
print("first")
print(acc)
second_class = []
for i in range(0, testData.shape[0]):
    if (testData[i,21] == 2):
        second_class.append(testData[i])
second_class = np.array(second_class)
first_class_x = second_class[: , 0:21]
first_class_x = np.array(first_class_x)
print(first_class_x.shape)
first_class_y = second_class[: , 21]
first_class_y = np.array(first_class_y)
y_pred = model.predict( first_class_x[:,f])
acc = accuracy_score(first_class_y, y_pred)
print("second")
print(acc)
third_class = []
for i in range(0, testData.shape[0]):
    if (testData[i,21] == 3):
        third_class.append(testData[i])
third_class = np.array(third_class)
first_class_x = third_class[: , 0:21]
first_class_x = np.array(first_class_x)
print(first_class_x.shape)
first_class_y = third_class[: , 21]
first_class_y = np.array(first_class_y)
y_pred = model.predict( first_class_x[:,f])
acc = accuracy_score(first_class_y, y_pred)
print("second")
print(acc)


############
print("$$$$$$$$$$$$")
first_class = []
for i in range(0, trainData.shape[0]):
    if (trainData[i,21] == 1):
        first_class.append(trainData[i])
first_class = np.array(first_class)
first_class_x = first_class[: , 0:21]
first_class_x = np.array(first_class_x)
print(first_class_x.shape)
first_class_y = first_class[: , 21]
first_class_y = np.array(first_class_y)
y_pred = model.predict( first_class_x[:,f])
acc = accuracy_score(first_class_y, y_pred)
print("first")
print(acc)
second_class = []
for i in range(0, trainData.shape[0]):
    if (trainData[i,21] == 2):
        second_class.append(trainData[i])
second_class = np.array(second_class)
first_class_x = second_class[: , 0:21]
first_class_x = np.array(first_class_x)
print(first_class_x.shape)
first_class_y = second_class[: , 21]
first_class_y = np.array(first_class_y)
y_pred = model.predict( first_class_x[:,f])
acc = accuracy_score(first_class_y, y_pred)
print("second")
print(acc)
third_class = []
for i in range(0, trainData.shape[0]):
    if (trainData[i,21] == 3):
        third_class.append(trainData[i])
third_class = np.array(third_class)
first_class_x = third_class[: , 0:21]
first_class_x = np.array(first_class_x)
print(first_class_x.shape)
first_class_y = third_class[: , 21]
first_class_y = np.array(first_class_y)
y_pred = model.predict( first_class_x[:,f])
acc = accuracy_score(first_class_y, y_pred)
print("second")
print(acc)

"""
mlp.fit(X_train,y_train)
y_pred = mlp.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(acc)
"""
print(counter_1)
print(counter_2)
print(counter_3)
