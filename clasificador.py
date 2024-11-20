import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv("train.csv")

def getTrainTest(digit):
	digit_train = digit[0:int(len(digit)*.8)]
	digit_train = digit_train[digit_train.columns[1:]].T
	digit_test = digit[int(len(digit)*.8):]
	digit_test = digit_test[digit_test.columns[1:]]
	return (digit_train, digit_test)


zero = data[data['label']==0]
zero_train,zero_test = getTrainTest(zero)

one = data[data['label']==1]
one_train,one_test = getTrainTest(one)

two = data[data['label']==2]
two_train,two_test = getTrainTest(two)

three = data[data['label']==3]
three_train,three_test = getTrainTest(three)

four = data[data['label']==4]
four_train,four_test = getTrainTest(four)

zero_u,e,v = svd(zero_train,full_matrices=False)
one_u,e,v = svd(one_train,full_matrices=False)
two_u,e,v = svd(two_train,full_matrices=False)
three_u,e,v = svd(three_train,full_matrices=False)
four_u,e,v ==svd(four_train,full_matrices=False)

def classifyUnknownDigit(newDigit):
    classes = [zero_u,one_u,two_u,three_u,four_u]
    values = []
    for U in classes:
        values.append(np.linalg.norm((np.identity(len(U))-np.matrix(U)*np.matrix(U.T)).dot(newDigit),ord=2)/np.linalg.norm(newDigit,ord=2))
    return values.index(min(values))

zero_pred = []
one_pred = []
two_pred = []
three_pred = []
four_pred = []
for i in range(len(four_test)):
    four_pred.append(classifyUnknownDigit(four_test.iloc[i]))
for i in range(len(zero_test)):
    zero_pred.append(classifyUnknownDigit(zero_test.iloc[i]))
for i in range(len(two_test)):
    two_pred.append(classifyUnknownDigit(two_test.iloc[i]))
for i in range(len(one_test)):
    one_pred.append(classifyUnknownDigit(one_test.iloc[i]))
for i in range(len(three_test)):
    three_pred.append(classifyUnknownDigit(three_test.iloc[i])

print "Accuracy"
print "------------"
print "0: ", zero_pred.count(0)/1.0/len(zero_pred) #count the number of 0's, divide by length of list to get accuracy. 
print "1: ", one_pred.count(1)/1.0/len(one_pred)
print "2: ", two_pred.count(2)/1.0/len(two_pred)
print "3: ", three_pred.count(3)/1.0/len(three_pred)
