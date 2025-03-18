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
    
    #calcola la misura del margine
    def compute_margin(self):
        numerator=np.abs(np.dot(self.features, self.weights) + self.bias)
        denominator=np.linalg.norm(self.weights)
        margin=numerator/denominator
        return margin
    
    #calcola l'iperpiano ottimale per aggiornare le features
    def fit(self):
        for epoch in range(self.epochs):
            for i, x in enumerate(self.features):
                condition=self.target[i] * (np.dot(x, self.weights) + self.bias) < 1
                if condition:
                    #Aggiornamento dei pesi e del bias per un punto mal classificato
                    self.weights += self.lr * (self.targets[i] * x - 2 * (1/self.epochs) * self.weights)
                    self.bias += self.lr * self.targets[i]
                else:
                    #Aggiornamento dei pesi solo con regolarizzazione
                    self.weights -= self.lr * (2 * (1/self.epochs) * self.weights)


    #fa una previsione binaria, utilizzando la formula w*x+b
    def predict(self, value):
        prediction=np.sign(np.dot(value, self.weights) + self.bias)
        return prediction

    #plotta alcuni relazioni interessanti nei dati
    def plot_graphics(self):
        
        