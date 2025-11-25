import pickle

from meu_knn import MeuKNN
from sklearn.metrics import accuracy_score

with open('credit.pkl', 'rb') as f:
    [x_credit_treinamento, y_credit_treinamento,
    x_credit_teste, y_credit_teste] = pickle.load(f)

meu_knn_credit = MeuKNN(k=5)
meu_knn_credit.fit(x_credit_treinamento, y_credit_treinamento)
previsoes = meu_knn_credit.prever(x_credit_teste)
print(accuracy_score(y_credit_teste, previsoes))