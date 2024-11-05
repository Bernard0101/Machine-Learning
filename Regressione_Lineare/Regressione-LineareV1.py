import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # ou 'Qt5Agg'
import matplotlib.pyplot as plt


data=pd.read_csv("dataset_velocita.csv")
features=data[["Tempo (s)", "Velocita (m/s)"]].values
labels=data["Distanza (m)"].values

class Visualizare_Data:
    def __init__(self, dataset):
        self.dataset=dataset        

    def plotData(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.dataset["Distanza (m)"], self.dataset["Tempo (s)"], marker='o', linestyle='-', c="b")
        plt.xlabel("Distanza (m)")
        plt.ylabel("Tempo (s)")
        plt.title("grafico distanza a rispeto di tempo")
        plt.grid(True)
        plt.show()
        plt.pause(0.001)

        plt.figure(figsize=(10, 6))
        plt.plot(self.dataset["Distanza (m)"], self.dataset["Velocita (m/s)"], marker='o', linestyle='-', c="y")
        plt.xlabel("distanza (m)")
        plt.ylabel("Velocita (m/s)")
        plt.title("grafico distanza a rispeto della velocita")
        plt.grid(True)
        plt.show()

    def plotModelo(self, epochi, errori, predizioni, target):
        plt.figure(figsize=(10, 6))
        plt.plot(epochi, errori, marker='o', linestyle='-', c='red')
        plt.xlabel("epochi")
        plt.ylabel("errori")
        plt.title("Progresso modelo")
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(target, label='labels', marker='o')  # Linea dei label
        plt.plot(predizioni, label='Predizioni', marker='x')  # Linea delle predizioni
        plt.xlabel('Index')
        plt.ylabel('Values')
        plt.title('Confronto tra Predizioni e Valori Veri')
        plt.legend()
        plt.show()




class Regressione_Lineare:
    def __init__(self, tassa_apprendimento, n_variabili, epochs, features, labels):
        lim = np.sqrt(6 / (n_variabili + 1)) 
        self.pesi=np.random.uniform(-lim, lim, n_variabili)
        self.bias=np.random.uniform(-lim, lim)
        self.tassa_appredimento=tassa_apprendimento
        self.features=features
        self.labels=labels
        self.epochs=epochs
    
    #esegue la formula per calcolo della predizione y=W*X ritornando un vettore di predizione
    def prevedere(self, X):
        predizione=np.dot(X, self.pesi) + self.bias
        return predizione

    #questo utiliza l'algoritmo di MSE per calcolare l'errore
    def calcolareErrore(self, predizione, target):
        errore=np.mean( (target-predizione) ** 2)
        return errore
    
    #qui calcoliamo la derivata di MSE a rispetto della nostra predizione e dopo aggiornamo il pesi di nostro modelo
    def aggiornarePesi(self, predizione, target):

        # Calcolo del gradiente rispetto ai pesi
        derivata_perdita = 2 * np.mean((target - predizione)[:, np.newaxis] * self.features, axis=0)

        # Calcolo del gradiente rispetto al bias
        derivata_bias = 2 * np.mean(target - predizione)

        # Aggiornamento dei pesi e del bias senza clipping per debug
        self.pesi += derivata_perdita * self.tassa_appredimento
        self.bias += derivata_bias * self.tassa_appredimento
        
    #qui il loop di allenamento...
    def allenare(self):
        errori=[]
        epochi=[]
        for epoch in range(self.epochs):
            preds=self.prevedere(X=self.features)
            error=self.calcolareErrore(predizione=preds, target=labels)
            self.aggiornarePesi(predizione=preds, target=labels)
            epochi.append(epoch)
            errori.append(error)
            if epoch % 1 == 0 :
                print(f"epoch: {epoch} errore: {error}")
        return epochi, errori, preds

    



visualizare=Visualizare_Data(dataset=data)
visualizare.plotData()
reg=Regressione_Lineare(n_variabili=2, epochs=50, features=features, labels=labels, tassa_apprendimento=0.0001)
epochi, errori, preds=reg.allenare()
visualizare.plotModelo(epochi=epochi, errori=errori, predizioni=preds, target=labels)

