import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from yellowbrick.classifier import ConfusionMatrix

# Importando base de dados
with open('credit.pkl', 'rb') as f:
    [x_credit_treinamento, y_credit_treinamento,
    x_credit_teste, y_credit_teste] = pickle.load(f)

print(x_credit_treinamento)

knn_credit = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn_credit.fit(x_credit_treinamento, y_credit_treinamento)

previsoes = knn_credit.predict(x_credit_teste)
print(accuracy_score(y_credit_teste, previsoes))
print(classification_report(y_credit_teste, previsoes))
print(confusion_matrix(y_credit_teste, previsoes))

cm = ConfusionMatrix(knn_credit)
cm.fit(y_credit_teste, previsoes)
print(cm.score(x_credit_teste, y_credit_teste))

plt.show()