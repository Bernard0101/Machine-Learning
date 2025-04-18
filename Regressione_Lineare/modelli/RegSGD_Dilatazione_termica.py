import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from Regressione_Lineare import SGD_Regressore
from Funzioni_ed_utensili import processore
from Funzioni_ed_utensili import functions

#dataset e separazione in features e targets
data_path="Datasets/Dilatazione_termica_acciaio.csv"
data=pd.read_csv(data_path)
std_data=functions.standartizzareData(df=data)
delta_T=np.array(std_data["Temperatura_Finale (C)"] - std_data["Temperatura_Iniziale (C)"].values)
std_features=np.expand_dims(delta_T, axis=1)
std_targets=std_data["Allungamento (m)"].values
k_folds=5


#alleno del modello sul dataset utilizzando la metrica di cross validation
SGD_reg=SGD_Regressore.Regressione_Lineare(features=std_features, labels=std_targets, tassa_apprendimento=0.003, inputs=1, outputs=1, epochs=50, funzione="MSE")
processore=processore.Processore_dati(modello=SGD_reg, dataset=data_path)
errore_fold, errore_alleno, ordine=processore.cross_validation(K=k_folds, features=std_features, labels=std_targets)


#processamento dei risultati e preparo per l'analizze grafica
features_denormalizzata=functions.denormalizzaData(standard_data=std_features, colonna=data[["Temperatura_Iniziale (C)","Temperatura_Finale (C)","Coefficiente_Dilatazione (C)","Lunghezza_Iniziale (m)"]].values)
target_denormalizzato=functions.denormalizzaData(standard_data=std_targets, colonna=data["Allungamento (m)"].values)
pred_denormalizzata=functions.denormalizzarePredizione(pred=SGD_reg.predizioni, target=data["Allungamento (m)"].values)
pred_ordinata=np.empty_like(SGD_reg.predizioni)
ordine=np.array(ordine)
pred_ordinata[ordine]=pred_denormalizzata

#visualizzazione della variazione della temperatura con l'allungamento 
plt.figure(figsize=(10, 6))
plt.scatter(x=features_denormalizzata, y=target_denormalizzato, c="darkred", alpha=0.7)
plt.title("Variazione della temperatura contro Allungamento dell'accaiao")
plt.xlabel("Variazione della temperatura (ΔT)")
plt.ylabel("Allungamento (m)")
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
plt.scatter(x=features_denormalizzata, y=target_denormalizzato, c="darkred", alpha=0.7, label="Variazione Temperatura (ΔT)")
plt.plot(features_denormalizzata, pred_ordinata, c="crimson", label="Regressione Lineare")
plt.title("Regression lineare tra variazione Temperatura contro Allungamento")
plt.xlabel("Variazione della temperatura (ΔT)")
plt.ylabel("Allungamento (m)")
plt.grid(True)
plt.legend()
plt.show()


