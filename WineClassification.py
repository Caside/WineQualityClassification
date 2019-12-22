# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 01:45:08 2019

@author: AlexandeR
"""
print("Какое вино будем оценивать? Введите 0 - красное, 1 - белое")
color = int(input())
print("Введите параметры в 1 строку")
import pandas as pd
from sklearn.preprocessing import scale
X_test = pd.DataFrame(list(map(float, input().split()))).T
if color == 0:
    from sklearn.ensemble import RandomForestClassifier
    red_wine_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')
    new_classes = {3: 0, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1, 9: 1}
    red_wine_data['quality_new'] = red_wine_data['quality'].map(new_classes)
    X = red_wine_data.drop(columns = ['quality', 'quality_new'])
    y = red_wine_data.quality_new
    X_test = pd.DataFrame(scale(X_test))
    Xs = pd.DataFrame(scale(X))
    rfc = RandomForestClassifier(random_state=0, n_estimators=40, max_depth=11, min_samples_leaf=1, min_samples_split=2, n_jobs =-1)
    rfc.fit(Xs, y)
    print("Предсказанный класс (0 или 1):",rfc.predict(X_test)[0])
elif color == 1:
    from sklearn.neighbors import KNeighborsClassifier
    white_wine_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv', sep=';')
    new_classes = {3: 0, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1, 9: 1}
    white_wine_data['quality_new'] = white_wine_data['quality'].map(new_classes)
    X = white_wine_data.drop(columns = ['quality', 'quality_new'])
    y = white_wine_data.quality_new
    X_test = pd.DataFrame(scale(X_test))
    Xs = pd.DataFrame(scale(X))
    knc = KNeighborsClassifier(metric='manhattan', n_neighbors=17, weights='distance', n_jobs=-1)
    knc.fit(Xs, y)
    print("Предсказанный класс (0 или 1):",knc.predict(X_test)[0])
    