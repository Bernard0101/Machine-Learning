import numpy as np
import pandas as pd 
import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


data=pd.read_csv("Datasets/Legge_di_Ohm.csv")
f=data[["Corrente (Ampere)","Tensione (Volt)"]].values
l=data["Resistenza (Ohm)"].values


class Regressione_Lineare:
    def __init__(self, features, labels, tassa_apprendimento, colonne=f.shape[1], epochs=25):
        self.features=features
        self.labels=labels
        self.pesi=np.random.randn(colonne, 1)
        self.bias=np.random.randn(1)
        self.tassa_appredimento=tassa_apprendimento
        self.epochs=epochs
        self.predizioni=None
        self.errori=[]
        self.epoche=[]
    

    def prevedere(self, X, alpha=0.01):
        predizione=np.dot(X, self.pesi) + self.bias
        return predizione

    #funzione che implementa l'agoritmo di SGD per addestramento dei pesi
    def SGD_ottimizzatore(self, predizione, target):
        errore = (predizione - target.reshape(-1, 1))

        gradiente_pesi=np.dot(self.features.T, errore) / len(self.labels)

        self.pesi -= self.tassa_appredimento * gradiente_pesi
        self.bias -= self.tassa_appredimento * np.mean(errore)

        errore=np.mean(errore)
        return errore
        
    #loop di allenamento per diminuire la perdita a ogni epoca
    def allenare(self):
        self.epoche=[]
        self.errori=[]
        for epoch in range(self.epochs):
            preds=self.prevedere(X=self.features, alpha=0.01)            
            errore=self.SGD_ottimizzatore(predizione=preds, target=self.labels)
            self.epoche.append(epoch)
            self.errori.append(errore)
            if epoch % 5 == 0 :
                print(f"epoch: {epoch}| errore: {errore}")
        self.predizioni=preds
        return preds
       
    



