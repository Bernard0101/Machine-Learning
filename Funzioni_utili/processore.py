import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



class processore_dati:

    def __init__(self, modello, dataset):
        self.modello=modello
        self.dataset=pd.read_csv(dataset)


    def split_data(self, fattore, features, labels):
        split=fattore*len(features)
        X_train=features[:split]
        X_test=features[:split]
        y_train=labels[split:]
        y_test=labels[split:]
        return X_train, y_train, X_test, y_test


    def cross_validation(self, K, features, labels):

        #crea un numero specifico che divide ugualmente tutti elementi dello dataset
        fold_size=len(features) // K

        #mescola i dati ogni volta che e necessario esseguire una nuova validazione
        indices=np.arange(len(features))
        np.random.shuffle(indices)
        features, labels=np.array(features[indices]), np.array(labels[indices])

        #crea una lista dove ogni elemento di essa e un altra lista contenente fold size elementi 
        feature_folds=[features[i*fold_size:fold_size*(i+1)]for i in range(K)]
        label_folds=[labels[i*fold_size:fold_size*(i+1)]for i in range(K)]

        #prendere la parte di test e training
        for i in range(K):
            print(f"\n\n\nalleno {i}:")
            x_test, y_test=feature_folds[i], label_folds[i]
            x_train=np.concatenate([feature_folds[j] for j in range(K) if j != i], axis=0)
            y_train=np.concatenate([label_folds[j] for j in range(K) if j != i], axis=0)


            self.modello.features=x_train
            self.modello.labels=y_train
            self.modello.allenare()
            

    def standartizzareData(self):
        mean=np.mean(self.dataset, axis=0)
        deviation=np.std(self.dataset, axis=0)
        standardize=(self.dataset-mean) / deviation
        standardized_df=pd.DataFrame(standardize, columns=self.dataset.columns)
        self.modello.dataset=standardized_df
        return standardized_df