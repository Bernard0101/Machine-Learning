import numpy as np
import pandas as pd 
import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from Funzioni_ed_utensili import functions


class Regressione_Lineare:
    def __init__(self, features, labels, tassa_apprendimento, inputs, outputs, epochs=25, funzione="MSE"):
        self.features=features
        self.labels=labels
        self.inputs=inputs
        self.outputs=outputs
        self.pesi=np.random.randn(inputs, 1)
        self.bias=np.random.randn(1) 
        self.funzione=funzione
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

    def perdita(self, predizione, target, funzione="MSE"):
        if funzione == "MAE":
            MAE=functions.Loss_MAE(y_pred=predizione, y_label=target)
            return MAE
        if funzione == "MSE":
            MSE=functions.Loss_MSE(y_pred=predizione, y_label=target)
            return MSE
        
    #funzione che implementa l'agoritmo di SGD per addestramento dei pesi
    def SGD_ottimizzatore(self, predizione, target):

        #mescolare il dataset
        ordine=np.arange(len(self.labels))
        np.random.shuffle(ordine)

        #ottenere l'errore del modello per ogni batch del dataset
        for batch in range(len(ordine)): 
            batch_pred=predizione[ordine[batch]]
            batch_target=target[ordine[batch]]

            #calcolare la derivata della funzione di costo rispetto al batch di predizione e target
            errore_batch=functions.Loss_MSE_derivative(y_pred=batch_pred, y_label=batch_target)
            gradiente_peso_batch=errore_batch * self.features[ordine[batch]]
            gradiente_bias_batch=errore_batch

            self.pesi -= self.tassa_appredimento * gradiente_peso_batch
            self.bias -= self.tassa_appredimento * gradiente_bias_batch
            
        
    
    #loop di allenamento per diminuire la perdita a ogni epoca
    def allenare(self, inizializzazione="He"):
        errori=[]
        self.inizializzazione_pesi(init=inizializzazione)
        for epoch in range(self.epochs):
            preds=self.prevedere(X=self.features, alpha=0.01)   
            errore=self.perdita(predizione=preds, target=self.labels, funzione=self.funzione)
            self.SGD_ottimizzatore(predizione=preds, target=self.labels)
            errori.append(errore)
            if epoch % 5 == 0 :
                print(f"epoch: {epoch}| errore: {errore}")
        self.predizioni=preds
        self.errori=errori
        return errori
       
    



