# [기말 범위] [데이터분석및활용]로지스틱회귀
import matplotlib.pyplot as plt
from random import randint
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# 데이터 로드
digit_dataset = datasets.load_digits()

# 데이터셋 -> 테스트 데이터셋/훈련 데이터셋
X = digit_dataset["data"]
y = digit_dataset["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y)

# 학습
logreg = LogisticRegression(max_iter=5000)
logreg.fit(X_train, y_train)

# 평가
y_pred = logreg.predict(X_test).copy()
y_true = y_test.copy()
score = accuracy_score(y_true, y_pred)
print(score)