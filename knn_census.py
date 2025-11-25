import pickle

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from yellowbrick.classifier import ConfusionMatrix

with open('census.pkl', 'rb') as f:
    [x_census_treinamento, y_census_treinamento,
    x_census_teste, y_census_teste] = pickle.load(f)

knn_census = KNeighborsClassifier(n_neighbors=10)
knn_census.fit(x_census_treinamento, y_census_treinamento)

previsoes = knn_census.predict(x_census_teste)
print(accuracy_score(y_census_teste, previsoes))
print(classification_report(y_census_teste, previsoes))
print(confusion_matrix(y_census_teste, previsoes))

cm = ConfusionMatrix(knn_census)
cm.fit(y_census_teste, previsoes)
print(cm.score(x_census_teste, y_census_teste))
plt.show()