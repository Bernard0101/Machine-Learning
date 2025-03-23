import numpy as np
import pandas as pd 
import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from Funzioni_ed_utensili import functions


class Regressione_Lineare:
    def __init__(self, features, labels, tassa_apprendimento, inputs, outputs, epochs=25):
        self.features=features
        self.labels=labels
        self.inputs=inputs
        self.outputs=outputs
        self.pesi=np.random.randn(inputs, 1)
        self.bias=np.random.randn(1) 
        self.tassa_appredimento=tassa_apprendimento
        self.epochs=epochs
        self.predizioni=None
        self.errori=[]

    def inizializzazione_pesi(self, init):
        init_Xavier=np.sqrt(2 / (self.inputs + self.outputs))
        init_He=np.sqrt(2 / self.inputs)
        if init == "Xavier":
            self.pesi *= init_Xavier
            self.bias *= init_Xavier
        if init == "He":
            self.pesi *= init_He
            self.bias *= init_He


    def prevedere(self, X, alpha=0.01):
        predizione=np.dot(X, self.pesi) + self.bias
        return predizione

    #funzione che implementa l'agoritmo di SGD per addestramento dei pesi
    def SGD_ottimizzatore(self, predizione, target, funzione="MSE"):
        errore=(predizione - target.reshape(-1, 1))
        
        gradiente_pesi=np.dot(self.features.T, errore) / len(self.labels)

        self.pesi -= self.tassa_appredimento * gradiente_pesi
        self.bias -= self.tassa_appredimento * np.mean(errore)
        
        MSE_loss=functions.Loss_MSE(y_pred=predizione, y_label=target)
        return MSE_loss
    
    #loop di allenamento per diminuire la perdita a ogni epoca
    def allenare(self, inizializzazione="He"):
        errori=[]
        self.inizializzazione_pesi(init=inizializzazione)
        for epoch in range(self.epochs):
            preds=self.prevedere(X=self.features, alpha=0.01)            
            errore=self.SGD_ottimizzatore(predizione=preds, target=self.labels)
            errori.append(errore)
            if epoch % 5 == 0 :
                print(f"epoch: {epoch}| errore: {errore}")
        self.predizioni=preds
        self.errori=errori
        return errori
       
    



