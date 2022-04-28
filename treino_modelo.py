import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

import itertools
import pickle

target = 'HEMOGLOBINA GLICADA'
parameters = [
            'Idade',
            'CREATININA EM SANGUE',
            'GLICOSE EM SANGUE',
            'HEMOGRAMA_BASO%',
            'HEMOGRAMA_CHCM',
            'HEMOGRAMA_HCM',
            'HEMOGRAMA_HT',
            'HEMOGRAMA_LEUC',
            'HEMOGRAMA_LINFO%',
            'HEMOGRAMA_MONO%',
            'HEMOGRAMA_MPV',
            'HEMOGRAMA_PLAQ',
            'HEMOGRAMA_RDW',
            'HEMOGRAMA_SEG%',
            'HEMOGRAMA_VCM',
]

ipd = 5.7
idb = 6.5

pred_offset = 0
true_offset = 0

def classifica(y, offset): 

    ipd_ajust = ipd+offset
    idb_ajust = idb+offset

    yb = np.zeros(y.shape)
    yb = np.where(y >= (ipd_ajust), 1, yb)
    yb = np.where(y >= (idb_ajust), 2, yb)

    return yb


scaler = StandardScaler()

if os.path.exists('exames2019Total.csv'):
    data = pd.read_csv('exames2019Total.csv')
    df_final = data.loc[(data.Idade > 19) & (data.Idade < 100)].copy(True)

    df_train, df_test = train_test_split(df_final, test_size=0.3)

    X_train = df_train[parameters].values
    y_train = df_train[target].values

    X_test = df_test[parameters].values
    y_test = df_test[target].values

    X_train_scl = scaler.fit_transform(X_train)
    X_test_scl = scaler.fit_transform(X_test)

    rg_ann = MLPRegressor(hidden_layer_sizes=[20,50], solver='adam', learning_rate='adaptive', activation='relu'
                        , batch_size='auto', max_iter=300, verbose=True, tol=0.00001
                        , alpha=0.1, random_state=10)
    model_rg_ann = rg_ann.fit(X_train_scl, y_train)

    pickle.dump(model_rg_ann, open("modelo_treinado.sav", 'wb'))

    y_test_pred_rg_ann = model_rg_ann.predict(X_test_scl)

    print("RMSE: ", mean_squared_error(y_test, y_test_pred_rg_ann, squared=False))
    print("MSE: ", mean_squared_error(y_test, y_test_pred_rg_ann, squared=True))
    print("MAE: ", mean_absolute_error(y_test, y_test_pred_rg_ann))

    y_train_pred_rg_ann = model_rg_ann.predict(X_train_scl)
    yb_train_pred = classifica(y_train_pred_rg_ann, pred_offset)
    yb_train_true = classifica(y_train, true_offset)
    print('Acuracia treino:', '{:0.2f}%'.format(accuracy_score(yb_train_true, yb_train_pred)*100))

    yb_test_pred = classifica(y_test_pred_rg_ann, pred_offset)
    yb_test_true = classifica(y_test, true_offset)
    print('Acuracia teste:', '{:0.2f}%'.format(accuracy_score(yb_test_true, yb_test_pred)*100))

else:
    print('Planilha de dados não econtrada, insira-a na mesma pasta que o código')