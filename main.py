import pandas as pd 
import numpy as np

from Datasets import visualizzazione_dati
from Regressione_Lineare import formula_chiusa
from Regressione_Lineare import SGD_Regressore
from Funzioni_ed_utensili import processore
from Funzioni_ed_utensili import functions


data=pd.read_csv("Datasets/Legge_di_Ohm.csv")
standard_data=functions.standartizzareData(df=data)
corrente=standard_data[["Corrente (Ampere)"]].values
tensione=standard_data["Tensione (Volt)"].values
k_folds=8



#alleno del modello sul dataset utilizzando la metrica di cross validation
SGD_reg=SGD_Regressore.Regressione_Lineare(features=corrente, labels=tensione, tassa_apprendimento=0.01, inputs=1, outputs=1, epochs=200, funzione="MSE")
processore=processore.Processore_dati(modello=SGD_reg, dataset="Datasets/Legge_di_Ohm.csv")
errore_fold, errore_alleno=processore.cross_validation(K=k_folds, features=corrente, labels=tensione)



#processamento dei risultati e preparo per l'analizze grafica
feature_mean=data["Corrente (Ampere)"].mean()
feature_std=data["Corrente (Ampere)"].std()
target_mean=data["Tensione (Volt)"].mean()
target_std=data["Tensione (Volt)"].std()
features_denormalizzata=functions.denormalizzaData(standard_data=corrente, colonna=data["Corrente (Ampere)"].values)
target_denormalizzato=functions.denormalizzaData(standard_data=tensione, colonna=data["Tensione (Volt)"].values)
pred_denormalizzata=functions.denormalizzarePredizione(pred=SGD_reg.predizioni, mean=target_mean, std=target_std)




#plotaggio e analisi dei grafici

print(f"corrente: {len(corrente)}\n tensione: {len(tensione)}\n preds: {len(pred_denormalizzata)}")
visualizzazione_dati.relazione_voltaggio_corrente(x=features_denormalizzata, y=target_denormalizzato)
visualizzazione_dati.Progresso_modello(x=np.arange(0, len(SGD_reg.errori), 1), y=SGD_reg.errori)
visualizzazione_dati.cross_validation_analise(K=k_folds, perdita=errore_alleno)
visualizzazione_dati.Comparazione_predizioni(x=features_denormalizzata, y=target_denormalizzato, preds=pred_denormalizzata)

