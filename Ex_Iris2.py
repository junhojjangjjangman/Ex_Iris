import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
#print(dir(iris))

#print(iris.feature_names)
#print(iris.target_names)

#print(iris.target, iris.target.shape)

#print((iris.target[iris.target==0].shape, iris.target[iris.target==1].shape, iris.target[iris.target==2].shape))

#print(iris.data.shape, iris.data[:5])
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)
print(X_train[:5], y_train[:5])

iris_df = pd.DataFrame(X_train, columns=iris.feature_names)
print("\n")
print(iris_df[:5])


pd.plotting.scatter_matrix(iris_df, c=y_train, s=60, alpha=0.8, figsize=[12,12])

print('')

model = KNeighborsClassifier(n_neighbors=4) # 기본값은 5
model.fit(X_train, y_train)

print(model.predict([[6,3,10,1.5]]))

score = model.score(X_test, y_test) # score를 이용해 평가
print(score)

pred_y=model.predict(X_test)
print(pred_y==y_test)

print((model.predict(X_test)==y_test).mean())