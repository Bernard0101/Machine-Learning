import numpy as np


#ReLU function ativazione
def activation_ReLU(Z):
    return np.maximum(0, Z)

def activation_ReLU_derivative(Z):
    return np.where(Z > 0, 1, 0)

#Leaky ReLU variant ativazione
def activation_leaky_ReLU(Z, alpha=0.03):
    return np.where(Z >= 0, Z, alpha * Z)

def activation_leaky_ReLU_derivative(Z, alpha=0.03):
    return np.where(Z > 0, 1, alpha)

#Sigmoid function ativazione
def activation_Sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def activation_Sigomid_derivative(Z):
    s = activation_Sigmoid(Z)
    return s * (1-s)

#Tanh function ativazione
def activation_tanh(Z):
    return np.tanh(Z)

def activation_tanh_derivative(Z):
    return 1-(activation_tanh(Z) ** 2)


#mse Loss
def Loss_MSE(y_pred, y_label):
    return np.mean((y_pred-y_label)**2)
    
def Loss_MSE_derivative(y_pred, y_label):
    return 2 * (y_pred-y_label) / len(y_label)

#MAE Loss
def Loss_MAE(y_pred, y_label):
    return np.mean(y_pred-y_label)

def Loss_MAE_derivative(y_pred, y_label):
    n = len(y_label)
    gradients = np.where(y_pred < y_label, -1 / n, 1 / n)
    gradients[y_pred == y_label] = 0 
    return gradients    

#Binary Cross Entropy Loss
def Loss_Binary_Cross_Entropy(y_pred, y_label):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    loss = -np.mean(y_label * np.log(y_pred) + (1 - y_label) * np.log(1 - y_pred))
    return loss

def Loss_Binary_Cross_Entropy_derivative(y_pred, y_label):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    derivative = -(y_label / y_pred) + (1 - y_label) / (1 - y_pred)
    return derivative

def Loss_Softmax(Z):
    exp_z = np.exp(Z - np.max(Z)) 
    return exp_z / np.sum(exp_z)

def Loss_Softmax_derivative(Z):
    s = Loss_Softmax(Z).reshape(-1, 1) 
    return np.diagflat(s) - np.dot(s, s.T) 





#funzione che scalabilizza i dati 
def standartizzareData(df):
    mean=df.mean()
    std=df.std()
    standard_data=(df - mean) / std
    return standard_data


def denormalizzareData(pred, mean, std):
    data_normalizzata=pred * std + mean
    return data_normalizzata