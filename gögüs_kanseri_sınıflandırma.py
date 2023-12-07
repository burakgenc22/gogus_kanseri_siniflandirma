#Göğüs Kanseri Sınıflandırma

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis, LocalOutlierFactor
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option("display.width", 500)

df = pd.read_csv("cancer.csv")
df.head()
df.drop(["Unnamed: 32", "id"], inplace=True, axis=1)
df = df.rename(columns= {"diagnosis":"target"})

sns.countplot(x=df["target"])
plt.show(block=True)

df["target"].value_counts()
df["target"] = [1 if col.strip() == "M" else 0 for col in df["target"]]
print(len(df))
print("data shape:" , df.shape)
df.info()
df.describe().T
df.isnull().sum()


#Keşifçi Veri Analizi

#korelasyon matrisi
corr_matrix = df.corr()
sns.clustermap(corr_matrix, annot=True, fmt=".2f")
plt.title("Değişkenler arasındaki korelasyon")
plt.show(block=True)

threshold = 0.50
filtre = np.abs(corr_matrix["target"]) > threshold
corr_features = corr_matrix.columns[filtre].tolist()
sns.clustermap(df[corr_features].corr(), annot=True, fmt=".2f")
plt.title("Değişkenler arasındaki korelasyon 0.75")
plt.show(block=True)

#pairplot

sns.pairplot(df[corr_features], diag_kind="kde", markers="+", hue="target")
plt.show(block=True)

###veri setinden bir çarpıklıkl söz konusudur

#Outlier Detection: Local Outlier Factor
y = df["target"]
x = df.drop(["target"], axis=1)
columns = x.columns.tolist()

clf = LocalOutlierFactor()
y_pred = clf.fit_predict(x)
x_score = clf.negative_outlier_factor_
outlier_score = pd.DataFrame()
outlier_score["score"] = x_score
outlier_score.sort_values(by="score", ascending=False)

threshold = -2.5
filtre = outlier_score["score"] < threshold
outlier_index = outlier_score[filtre].index.tolist()

plt.figure()
plt.scatter(x.iloc[outlier_index,0], x.iloc[outlier_index,1], color="blue", s=50, label="outliers")
plt.show(block=True)

plt.figure()
plt.scatter(x.iloc[:,0], x.iloc[:,1], color="k", s=3, label="data point")
plt.show(block=True)

radius = (x_score.max() - x_score) / (x_score.max() - x_score.min())
outlier_score["radius"] = radius
outlier_score.head(50)

plt.scatter(x.iloc[:,0], x.iloc[:,1], edgecolors="r", s=1000*radius,  facecolors="none", label= "outlier scores")
plt.legend()
plt.show(block=True)

#Outlierların çıkartılması

x = x.drop(outlier_index)
y = y.drop(outlier_index).values

#verinin eğitilmesi(train-test split)

test_size=0.3
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

#Verinin standartlaşıtılması

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train_df = pd.DataFrame(x_train, columns=columns)
x_train_df.head(50)

#KNN algoritmasının kurulması
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
score = knn.score(x_test, y_test)
print("Score:", score)
print("Basic KNN ACC:", acc)
print("CM:", cm)

#Score: 0.9532163742690059
#Basic KNN ACC: 0.9532163742690059
#CM: [[108   1]
 #    [  7  55]]

#KNN en iyi parametre değerlerini bulma

def KNN_best_params(x_train, x_test, y_train, y_test):
    k_range = list(range(1,31))
    weight_options = ["uniform", "distance"]
    print()
    param_grid = dict(n_neighbors = k_range, weights = weight_options)

    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn, param_grid, cv=10, scoring="accuracy")
    grid.fit(x_train, y_train)

    print("best training score: {} with parameters: {}".format(grid.best_score_, grid.best_params_))
    print()

    knn = KNeighborsClassifier(**grid.best_params_)
    knn.fit(x_train, y_train)
    y_pred_test = knn.predict(x_test)
    y_pred_train = knn.predict(x_train)

    cm_test = confusion_matrix(y_test, y_pred_test)
    cm_train = confusion_matrix(y_train, y_pred_train)

    acc_test = accuracy_score(y_test, y_pred_test)
    acc_train = accuracy_score(y_train, y_pred_train)
    print("test score: {}, train score: {}".format(acc_test, acc_train))
    print()
    print("CM test: ", cm_test)
    print("CM train: ", cm_train)
    return grid

grid = KNN_best_params(x_train, x_test, y_train, y_test)

#best training score: 0.9670512820512821 with parameters: {'n_neighbors': 4, 'weights': 'uniform'}
#test score: 0.9590643274853801, train score: 0.9773299748110831

#CM test:  [[107   2]
#           [  5  57]]
#CM train:  [[248   0]
#            [  9 140]]








































































