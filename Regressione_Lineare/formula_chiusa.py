import numpy as np
import pandas as pd
import matplotlib
import sys
matplotlib.use('TkAgg')  # ou 'Qt5Agg'
import matplotlib.pyplot as plt


data=pd.read_csv("Datasets/Legge_di_Ohm.csv")
f=data[["Corrente (Ampere)","Tensione (Volt)"]].values
l=data["Resistenza (Ohm)"].values


class Regressione_Lineare:
    def __init__(self, features, labels):
        self.features=features
        self.labels=labels
        self.coef_B=None
        self.matrice_X=features
        self.vettore_Y=labels
    
    #fare una predizione con la nostra funzione
    def prevedere(self):
        predizione=np.dot(self.matrice_X, self.coef_B)
        return predizione
    
    #calcolare l'errore e valutare il modelo
    def valutare(self):
        mae=np.mean(np.abs(self.labels-self.vettore_Y))
        return mae
    
    #aggiornare il coefficiente per farlo piu giusto ai nostri datti
    def allenare(self):
        X_transposed=np.dot(self.matrice_X.T, self.matrice_X)
        X_transposed_inv=np.linalg.inv(X_transposed)
        X_transposed_Y=np.dot(self.matrice_X.T, self.vettore_Y)

        #calcolare il vettore dei coefficienti
        coefficiente_B=np.dot(X_transposed_inv, X_transposed_Y)
        
        self.coef_B=coefficiente_B

        MAE=self.valutare()
        print(f"MAE dello modello: {MAE}")
        return coefficiente_B
    

