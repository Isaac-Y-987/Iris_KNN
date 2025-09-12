from data import data_train,data_val, data_test
import KNN
from matplotlib import pyplot as plt
x = range(1,30)
y = []
for k in x:
    classifier = KNN.KNN_classifier(k,data_train)
    percent_accuracy = classifier.accuracy(data_val)
    y.append(percent_accuracy)
plt.plot(x,y)
# plt.show()
k = 12
classifier = KNN.KNN_classifier(k,data_train)
percent_accuracy = classifier.accuracy(data_test)
print(percent_accuracy)