# [기말 범위] [데이터분석및활용] 분류와군집화
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# 닥스훈트의 길이와 높이 데이터
dach_length = [55, 57, 64, 63, 58, 49, 54, 61]
dach_height = [30, 31, 36, 30, 33, 25, 37, 34]

# 진돗개의 길이와 높이 데이터
jin_length = [56, 47, 56, 46, 49, 53, 52, 48]
jin_height = [52, 52, 50, 53, 50, 53, 49, 54]

newdata_length = [59]    # 새로운 데이터의 길이
newdata_height = [35]    # 새로운 데이터의 높이

# 새 데이터의 표식은 오각형(pentagon)으로 설정하고, 레이블은 new Data로
plt.subplot(2, 1, 1)
plt.title("draw new data on existing data")
plt.scatter(dach_length, dach_height, c='r', label='Dachshund')
plt.scatter(jin_length, jin_height,c='b',marker='^', label='Jindo dog')
plt.scatter(newdata_length, newdata_height, s=100, marker='p',c='g', label='new Data')
plt.legend(loc='upper right')

# 닥스훈트는 0, 진돗개는 1로 레이브링
d_data = np.column_stack((dach_length, dach_height))
d_label = np.zeros(len(d_data))   
j_data = np.column_stack((jin_length, jin_height))
j_label = np.ones(len(j_data))   

newdata = np.column_stack((newdata_length, newdata_height))

dogs = np.concatenate((d_data, j_data))
labels = np.concatenate((d_label, j_label))

dog_classes = {0:'닥스훈트', 1:'진돗개'}

k = 3     # k를 3으로 두고 kNN 분류기를 만들어 보자
knn = KNeighborsClassifier(n_neighbors = k)
knn.fit(dogs, labels)

y_pred = knn.predict(newdata)
print('데이터', newdata, ', 판정 결과:', y_pred[0], dog_classes[y_pred[0]])
# 판정 결과값을 기존 데이터의 색깔과 맞추기
dot_color = ''
if dog_classes[y_pred[0]] == 0:
    dot_color = 'b'
else:
    dot_color = 'r'
# 판정 결과에 영향을 미치는 점 색칠하기
distances, indexes = knn.kneighbors(newdata)
nearest_neighbors = dogs[indexes]
plt.subplot(2, 1, 2)
plt.scatter(dach_length, dach_height, c='r', label='Dachshund')
plt.scatter(jin_length, jin_height,c='b',marker='^', label='Jindo dog')
plt.scatter(newdata_length, newdata_height, s=100, marker='p', c=dot_color, label='new Data')
plt.scatter(nearest_neighbors[0, :, 0], nearest_neighbors[0, :, 1], marker='D', color='orange', label='nearst Data')
plt.legend(loc='upper right')
plt.title("get result")
plt.show()