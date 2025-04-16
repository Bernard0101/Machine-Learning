import numpy as np
import matplotlib.pyplot as plt


        
def relazione_voltaggio_corrente(x, y):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, c="darkorange", alpha=0.7)
    plt.title("Corrente contro Resistenza")
    plt.xlabel("Corrente (Ampere)")
    plt.ylabel("Voltaggio (Volt)")
    plt.grid(True)
    plt.show()


def Progresso_modello(x, y):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, c="r")
    plt.title("Progresso del modello")
    plt.xlabel("epoche")
    plt.ylabel("errore")
    plt.grid(True)
    plt.show()



def cross_validation_analise(K, perdita):
    K_folder=np.arange(0, K, 1)
    plt.figure(figsize=(10, 6))
    plt.bar(x=K_folder, height=perdita, color="royalblue", edgecolor="black")
    plt.title("Analizze Cross-validation")
    plt.xlabel("numero di alleni")
    plt.ylabel("sbaglio del modello")
    plt.grid(True)
    plt.show()


def Comparazione_predizioni(x, y, preds):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label="dati sperimentali", c="darkorange")
    plt.plot(x, preds, label="Regressione Lineare", c="mediumturquoise", linewidth=2)
    plt.title("Regressione Lineare tra Tensione e Corrente")
    plt.xlabel("corrente (Ampere)")
    plt.ylabel("Voltaggio (Volt)")
    plt.grid(True)
    plt.legend()
    plt.show()


