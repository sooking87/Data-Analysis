# [기말 범위] [데이터분석및활용]선형회귀심화 + 로지스틱회귀 (분류) 문제 평가 방법 실습
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import  linear_model
from sklearn.metrics import mean_squared_error

boston_dataset = fetch_openml(name='boston', parser='auto', version=1)

x_data = boston_dataset["data"]
y_data = boston_dataset["target"]

# 데이터 전처리
minmax_scale = preprocessing.MinMaxScaler()
minmax_scale.fit(x_data)
x_scaled_data = minmax_scale.transform(x_data)

# 데이터셋 -> 테스트 데이터셋/훈련 데이터셋
X_train, X_test, y_train, y_test = train_test_split(x_scaled_data, y_data, test_size=0.33)

# 훈련
regr = linear_model.LinearRegression(fit_intercept=True, copy_X=True, n_jobs=8)
ridge_regr = linear_model.Ridge(alpha=0.01, fit_intercept=True, copy_X=True) # 릿지
regr.fit(X_train, y_train)
ridge_regr.fit(X_train, y_train)

print('예측 결과: ', regr.predict(x_data[:5].to_numpy()))

# 평가
y_true = y_test.copy()
y_hat = regr.predict(X_test)
print("MSE: ", mean_squared_error(y_true, y_hat, squared=False))