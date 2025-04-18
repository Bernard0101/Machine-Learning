import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from Regressione_Lineare import SGD_Regressore
from Funzioni_ed_utensili import processore
from Funzioni_ed_utensili import functions


data=pd.read_csv("Datasets/Legge_di_Ohm.csv")
standard_data=functions.standartizzareData(df=data)
corrente=standard_data[["Corrente (Ampere)"]].values
tensione=standard_data["Tensione (Volt)"].values
k_folds=10



#alleno del modello sul dataset utilizzando la metrica di cross validation
SGD_reg=SGD_Regressore.Regressione_Lineare(features=corrente, labels=tensione, tassa_apprendimento=0.01, inputs=1, outputs=1, epochs=150, funzione="MSE")
processore=processore.Processore_dati(modello=SGD_reg, dataset="Datasets/Legge_di_Ohm.csv")
errore_fold, errore_alleno, ordine=processore.cross_validation(K=k_folds, features=corrente, labels=tensione)


#processamento dei risultati e preparo per l'analizze grafica
features_denormalizzata=functions.denormalizzaData(standard_data=corrente, colonna=data["Corrente (Ampere)"].values)
target_denormalizzato=functions.denormalizzaData(standard_data=tensione, colonna=data["Tensione (Volt)"].values)
pred_denormalizzata=functions.denormalizzarePredizione(pred=SGD_reg.predizioni, target=data["Tensione (Volt)"].values)
pred_ordinata=np.empty_like(SGD_reg.predizioni)
ordine=np.array(ordine)
pred_ordinata[ordine]=pred_denormalizzata

#visualizzazione grafica delle features e labels
plt.figure(figsize=(10, 6))
plt.scatter(x=features_denormalizzata, y=target_denormalizzato, c="darkorange", alpha=0.7)
plt.title("Corrente contro resistenza")
plt.xlabel("Corrente (A)")
plt.ylabel("Tensione (V)")
plt.grid(True)
plt.show()

#visualizzazione del progresso del modello
plt.figure(figsize=(10, 6))
plt.plot(np.arange(0, len(SGD_reg.errori), 1), SGD_reg.errori, c="r")
plt.title("Errore del modello")
plt.xlabel("Epoche")
plt.ylabel("Errore")
plt.grid(True)
plt.show()

#visualizzazione della metrica di cross-validation
K_folder=np.arange(0, k_folds, 1)
plt.figure(figsize=(10, 6))
plt.bar(x=K_folder, height=errore_alleno, color="royalblue", edgecolor="black")
plt.title("Analizze Cross-validation")
plt.xlabel("numero di alleni")
plt.ylabel("sbaglio del modello")
plt.grid(True)
plt.show()

#visualizzazione delle predizioni contro le targets
plt.figure(figsize=(10, 6))
plt.scatter(x=features_denormalizzata, y=target_denormalizzato, c="darkorange", alpha=0.7)
plt.plot(features_denormalizzata, pred_ordinata, c="mediumturquoise")
plt.title("Regression lineare tra corrente e tensione")
plt.xlabel("corrente (Ampere)")
plt.ylabel("Voltaggio (Volt)")
plt.grid(True)
plt.legend()
plt.show()