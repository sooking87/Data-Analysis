# [기말 범위] [데이터분석및활용] 의사결정트리
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

def transform_status(x):
    if "Mrs" in x or "Ms" in x:
        return "Mrs"
    elif "Mr" in x:
        return "Mr"
    elif "Miss" in x:
        return "Miss"
    elif "Master" in x:
        return "Master"
    elif "Dr" in x:
        return "Dr"
    elif "Rev" in x:
        return "Rev"
    elif "Col" in x:
        return "Col"
    else:
        return "0"
    
# 데이터 준비
train_df = pd.read_csv("./Week13/train.csv")
test_df = pd.read_csv("./Week13/test.csv")

train_id = train_df["PassengerId"].values
test_id = test_df["PassengerId"].values
all_df = train_df._append(test_df).set_index('PassengerId')

## 데이터 전처리
all_df["Sex"] = all_df["Sex"].replace({"male":0,"female":1})

all_df["Age"].fillna(
    all_df.groupby("Pclass")["Age"].transform("mean"), inplace=True)

all_df["cabin_count"] = all_df["Cabin"].map(lambda x : len(x.split()) if type(x) == str else 0)

all_df["social_status"] = all_df["Name"].map(lambda x : transform_status(x))
all_df = all_df.drop([62,830])
train_id = np.delete(train_id, [62-1,830-1])

## null 값 채우기
all_df.groupby(["Pclass","Sex"])["Fare"].mean() 
all_df.loc[all_df["Fare"].isnull(), "Fare"] = 12.415462
all_df["cabin_type"] = all_df["Cabin"].map(lambda x : x[0] if type(x) == str else "None")

## 불필요 열 지우기
del all_df["Cabin"]
del all_df["Name"]
del all_df["Ticket"]
Y = all_df["Survived"]
del all_df["Survived"]

## 원핫인코딩과 스케일링
X_df = pd.get_dummies(all_df)
X = X_df.values

# 데이터셋 -> 학습데이터/시험데이터 분리
X_train = X[:len(train_id)]
X_test = X[len(train_id):]
Y_train = Y[:len(train_id)]
Y_test = Y[len(train_id):]

## 학습 데이터 scalling
minmax_scaler = MinMaxScaler()

minmax_scaler.fit(X_train)
X_train = minmax_scaler.transform(X_train)
X_test = minmax_scaler.transform(X_test)

# 훈련 및 평가
test_accuracy = []
train_accuracy = []
for idx in range(3, 20):
    dt = DecisionTreeClassifier(min_samples_leaf=idx)
    acc = cross_val_score(dt, X_train, Y_train, scoring="accuracy", cv=5).mean()
    train_accuracy.append(accuracy_score(dt.fit(X_train, Y_train).predict(X_train), Y_train))
    test_accuracy.append(acc)

result = pd.DataFrame(train_accuracy, index=range(3,20), columns=["train"])
result["test"] = test_accuracy

plt.plot(result['train'], label="train")
plt.plot(result['test'], label='test')
plt.legend()
plt.show()