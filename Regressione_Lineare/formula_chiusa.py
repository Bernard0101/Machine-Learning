import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # ou 'Qt5Agg'
import matplotlib.pyplot as plt


data=pd.read_csv("Regressione_Lineare/dataset_velocita.csv")
features=data[["Tempo (s)", "Velocita (m/s)"]].values
labels=data["Distanza (m)"].values

#Aggiunta della colonna di bias (1) per l'intercetta
bias = np.ones((features.shape[0], 1))
X = np.hstack((bias, features))

class Regressione_Lineare:
    def __init__(self, X, Y):
        self.coef_B=None
        self.matrice_X=X
        self.vettore_Y=Y
    
    #aggiornare il coefficiente per farlo piu giusto ai nostri datti
    def fit(self):
        X_transposed=np.dot(self.matrice_X.T, self.matrice_X)
        X_transposed_inv=np.linalg.inv(X_transposed)
        X_transposed_Y=np.dot(self.matrice_X.T, self.vettore_Y)

        #calcolare il vettore dei coefficienti
        coefficiente_B=np.dot(X_transposed_inv, X_transposed_Y)
        
        self.coef_B=coefficiente_B
        return coefficiente_B
    
    #fare una predizione con la nostra funzione
    def prevedere(self):
        predizione=np.dot(self.matrice_X, self.coef_B)
        return predizione
    
    #calcolare l'errore e valutare il modelo
    def valutare(self, predizione):
        mae=np.mean(np.abs(labels-predizione))
        return mae
    
reg=Regressione_Lineare(X=X, Y=labels)
reg.fit()
predizioni=reg.prevedere()
print("predizioni: ", predizioni)
errore=reg.valutare(predizione=predizioni)
print("errore assoluto: ", errore)
    

