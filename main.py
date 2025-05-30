import pandas as pd 
import numpy as np

from Datasets import visualizzazione_dati
from Regressione_Lineare import formula_chiusa
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
k_folds=10


#alleno del modello sul dataset utilizzando la metrica di cross validation
SGD_reg=SGD_Regressore.Regressione_Lineare(features=std_features, labels=std_targets, tassa_apprendimento=0.0003, inputs=1, outputs=1, epochs=150, funzione="MSE")
processore=processore.Processore_dati(modello=SGD_reg, dataset=data_path)
errore_fold, errore_alleno, ordine=processore.cross_validation(K=k_folds, features=std_features, labels=std_targets)


#processamento dei risultati e preparo per l'analizze grafica
features_denormalizzata=functions.denormalizzaData(standard_data=std_features, colonna=data[["Temperatura_Iniziale (C)","Temperatura_Finale (C)","Coefficiente_Dilatazione (C)","Lunghezza_Iniziale (m)"]].values)
target_denormalizzato=functions.denormalizzaData(standard_data=std_targets, colonna=data["Allungamento (m)"].values)
pred_denormalizzata=functions.denormalizzarePredizione(pred=SGD_reg.predizioni, target=data["Allungamento (m)"].values)
pred_ordinata=np.empty_like(SGD_reg.predizioni)
ordine=np.array(ordine)
pred_ordinata[ordine]=pred_denormalizzata


#plotaggio grafico e analisi dei risultati del modello 
visualizzazione_dati.data_features_targets(x=delta_T, y=target_denormalizzato, x_label="variazione Temperatura (ΔT)", y_label="Allungamento Acciaio (m)")
visualizzazione_dati.Progresso_modello(x=np.arange(0, len(SGD_reg.errori), 1), y=SGD_reg.errori)
visualizzazione_dati.cross_validation_analise(K=k_folds, perdita=errore_alleno)
visualizzazione_dati.Comparazione_predizioni(x=features_denormalizzata, y=target_denormalizzato, preds=pred_ordinata)
