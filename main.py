import pandas as pd 
import numpy as np

from Datasets import visualizzazione_dati
from Regressione_Lineare import formula_chiusa
from Regressione_Lineare import SGD_Regressore
from Funzioni_ed_utensili import processore
from Funzioni_ed_utensili import functions



data=pd.read_csv("Datasets/Legge_di_Ohm.csv")
standard_data=functions.standartizzareData(df=data)
f=standard_data[["Corrente (Ampere)"]].values
l=standard_data["Tensione (Volt)"].values
k_folds=5

#organizzazione e visualizzazione dei dati e avaluazione dei risultati
SGD_reg=SGD_Regressore.Regressione_Lineare(features=f, labels=l, tassa_apprendimento=0.001, inputs=1, outputs=1, epochs=25)
processore=processore.Processore_dati(modello=SGD_reg, dataset="Datasets/Legge_di_Ohm.csv")
errore_fold, errore_alleno=processore.cross_validation(K=k_folds, features=f, labels=l)

#processamento di alcuni dati
corrente=standard_data["Corrente (Ampere)"].values
tensione=standard_data["Tensione (Volt)"].values
corrente.sort()
tensione.sort()
pred_std=np.std(SGD_reg.predizioni)
pred_mean=np.mean(SGD_reg.predizioni)
pred_denormalizzata=functions.denormalizzareData(pred=SGD_reg.predizioni, mean=pred_mean, std=pred_std)

#plotaggio e analisi dei grafici
visualizzazione_dati.relazione_voltaggio_corrente(x=corrente, y=tensione)
visualizzazione_dati.Progresso_modello(x=np.arange(0, len(SGD_reg.errori), 1), y=SGD_reg.errori)
visualizzazione_dati.cross_validation_analise(K=k_folds, perdita=errore_alleno)
visualizzazione_dati.Comparazione_predizioni(x=corrente, y=tensione, preds=pred_denormalizzata)
