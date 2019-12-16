import numpy as np

def rmsle(y_pred,y_actual):
    rmsle = np.sqrt(np.mean(np.power((np.log1p(y_pred)-np.log1p(y_actual)),2)))
    return rmsle