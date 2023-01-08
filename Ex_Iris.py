import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

Image(url='https://user-images.githubusercontent.com/15958325/56006707-f69f3680-5d10-11e9-8609-25ba5034607e.png')

iris = load_iris()

features = iris['data']
#print(features[:5])

features_names = iris['feature_names']
#print(features_names)

labels = iris['target']
#print(labels)

#df = pd.DataFrame(features, columns=features_names)
df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
df.head()

df['target'] = iris['target']
df.head()

plt.figure(figsize=(10, 7))
sns.scatterplot(x=df.iloc[:, 0], y=df.iloc[:, 1], hue=df['target'], palette='muted')
plt.title('Sepal', fontsize = 17)
plt.show()

plt.figure(figsize=(10, 7))
sns.scatterplot(x=df.iloc[:, 2], y=df.iloc[:, 3], hue=df['target'], palette='muted')
plt.title('Petal', fontsize = 17)
plt.show()

x = df.iloc[:, :4]
x.head()

y = df['target']
y.head()

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=30)

print(x.shape)
print(x_train.shape, x_test.shape)
print(y.shape)
print(y_train.shape, y_test.shape)