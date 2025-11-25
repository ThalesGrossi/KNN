import numpy as np
from collections import Counter

class MeuKNN():
    def __init__(self, k=5):
        self.k = k
    
    def fit(self, x, y):
        self.x_treino = x
        self.y_treino = y
    
    def distancia_euclidiana(self, a, b):
        return np.sqrt(sum((a-b) ** 2))
    
    def prever(self, x_teste):
        x_teste = np.array(x_teste)
        previsoes = []

        for x in x_teste:
            distancias = [self.distancia_euclidiana(x, x_treino) for x_treino in self.x_treino]
            k_indices = np.argsort(distancias)[:self.k]
            k_labels = self.y_treino[k_indices]
            mais_comum = Counter(k_labels).most_common(1)[0][0]
            previsoes.append(mais_comum)
        return np.array(previsoes)