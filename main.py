import numpy as np
import pandas as pd 
import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from Regressione_Lineare import formula_chiusa
from Regressione_Lineare import SGD_Regressore
from Funzioni_utili import processore

data=pd.read_csv("Datasets/Legge_di_Ohm.csv")
f=data[["Corrente (Ampere)","Tensione (Volt)"]].values
l=data["Resistenza (Ohm)"].values

#organizzazione e visualizzazione dei dati e avaluazione dei risultati
Reg_formula_chiusa=formula_chiusa.Regressione_Lineare(features=f, labels=l,)
processore=processore.processore_dati(modello=Reg_formula_chiusa, dataset="Datasets/Legge_di_Ohm.csv")
processore.standartizzareData()
processore.cross_validation(K=5, features=Reg_formula_chiusa.features, labels=Reg_formula_chiusa.labels)

