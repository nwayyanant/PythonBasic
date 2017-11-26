import matplotlib as plt

from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()
clf = svm.SVC(gamma =0.0001, C=100)

print(len(digits.data()))

x,y = digits.data[:-1], digits.target[:-1]

print ('Predicition:', clf.predict(digit.data[-6]))
plt.imgshow(digits.image[-6], cmpa=plt.cm_gray_r,interpolation="nearest")
plt.show
      
