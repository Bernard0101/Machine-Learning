import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

class SVM:
    def __init__(self, features, targets, epochs, learning_rate):
        self.features=features
        self.targets=targets
        self.weights=np.random.randn(features)
        self.bias=np.random.randint()
        self.epochs=epochs
        self.lr=learning_rate
    
    #calcola la misura della margine dall'iperpiano ai due vettori di supporto
    def compute_margin(self):
        numerator=np.abs(np.dot(self.features, self.weights) + self.bias)
        denominator=np.linalg.norm(self.weights)
        margin=numerator/denominator
        return margin
    
    #calcola l'iperpiano ottimale per aggiornare le features
    def fit(self):
        pass

    #fa una previsione binaria, utilizzando la formula w*x+b
    def predict(self, value):
        prediction=np.sign(np.dot(value, self.weights) + self.bias)
        return prediction
        