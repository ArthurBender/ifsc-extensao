import pandas as pd
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix

from IPython.display import HTML
from IPython.core.interactiveshell import InteractiveShell
import seaborn as sns
from scipy import stats
InteractiveShell.ast_node_interactivity = "all"
from sklearn.preprocessing import label_binarize 
from sklearn.metrics import roc_curve, auc, precision_score, r2_score

import pickle

import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    #else:
    #    print('Confusion matrix, without normalization')

   # print(cm)

    _ = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    _ = plt.title(title)
    _ = plt.colorbar()
    tick_marks = np.arange(len(classes))
    _ = plt.xticks(tick_marks, classes, rotation=45)
    _ = plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        _ = plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center", fontsize=12,
                 color="white" if cm[i, j] > thresh else "black")

    _ = plt.ylabel('True value')
    _ = plt.xlabel('Predicted value')
    _ = plt.tight_layout()
    
    
def plot_cfm(y_true, y_pred, labels, categories, img_name=''):
    cnf_matrix = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)

    _ = plt.figure(figsize=(10,5))
    _ = plt.subplot(121)
    _ = plot_confusion_matrix(cnf_matrix, classes=categories, title=r'$HbA_{1c}$', normalize=False)

    _ = plt.subplot(122)
    _ = plot_confusion_matrix(cnf_matrix, classes=categories, title=r'$HbA_{1c}$', normalize=True)
    if (img_name != ''):
        plt.savefig(img_name, bbox_inches="tight")
    plt.show()


def classifica(y, offset): 

    ipd_ajust = ipd+offset
    idb_ajust = idb+offset

    yb = np.zeros(y.shape)
    yb = np.where(y >= (ipd_ajust), 1, yb)
    yb = np.where(y >= (idb_ajust), 2, yb)

    return yb

# Criação do array com as informações da planilha
data= pd.read_csv('exames2019Total.csv')

# Filtro de idade
df_final = data.loc[(data.Idade > 19) & (data.Idade < 100)].copy(True)

# Divide informações da planilha em treino e teste
df_train, df_test = train_test_split(df_final, test_size=0.3)

parameters = [
            'Idade',
            #'COLESTEROL HDL',
            #'COLESTEROL TOTAL',
            #'CREATININA EM SANGUE_AdultExRNeg',
            #'CREATININA EM SANGUE_AdultRNeg',
            'CREATININA EM SANGUE',
            #'ESTIMULADOR DA TIREOIDE (TSH), DOSAGEM DO HORMONIO',
            'GLICOSE EM SANGUE',
            #'HEMOGLOBINA GLICADA',
            'HEMOGRAMA_BASO%',
            #'HEMOGRAMA_BASOmm3',
            'HEMOGRAMA_CHCM',
            #'HEMOGRAMA_EOS%',
            #'HEMOGRAMA_EOSmm3',
            'HEMOGRAMA_HCM',
            #'HEMOGRAMA_HMC',
            'HEMOGRAMA_HT',
            #'HEMOGRAMA_Hb',
            'HEMOGRAMA_LEUC',
            'HEMOGRAMA_LINFO%',
            #'HEMOGRAMA_LINFOmm3',
            'HEMOGRAMA_MONO%',
            #'HEMOGRAMA_MONOmm3',
            'HEMOGRAMA_MPV',
            'HEMOGRAMA_PLAQ',
            'HEMOGRAMA_RDW',
            'HEMOGRAMA_SEG%',
            #'HEMOGRAMA_SEGmm3',
            'HEMOGRAMA_VCM',
            #'PARCIAL DE URINA_DENSIDADE',
            #'PARCIAL DE URINA_HEMAC',
            #'PARCIAL DE URINA_pH',
            #'POTASSIO EM SANGUE',
            #'PROTEINA C REATIVA, PESQUISA DE',
            #'SODIO EM SANGUE',
            #'TRIGLICERIDIOS_TJejum',
            #'TIROXINA (T4) LIVRE, DOSAGEM DO',
            #'TRANSAMINASE ALT (GPT), ATIVIDADE DA',
            #'TRANSAMINASE AST (GOT), ATIVIDADE DA',
            #'TRIGLICERIDIOS_TRIG',
            #'UREIA EM SANGUE',
            #'VITAMINA "B12"',
            #'VITAMINA "D" 25 HIDROXI'
]

target = 'HEMOGLOBINA GLICADA'

#inicio da prediabetes, inclusive
ipd = 5.7

#inicio da diabetes, inclusive
idb = 6.5


#individuos saudaveis, com prediabetes e diabetes
# X = Lista de valores dos parametros, Y = Lista de valores do alvo
X_train = df_train[parameters].values
y_train = df_train[target].values

X_test = df_test[parameters].values
y_test = df_test[target].values


#normalizacao das entradas
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import Normalizer
scaler = StandardScaler()
#scaler = QuantileTransformer()
#scaler = Normalizer()

# Converte valores para Standard ????????
X_train_scl = scaler.fit_transform(X_train)
X_test_scl = scaler.fit_transform(X_test)

#########################################################################################
#Modelo ANN
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# TREINO EM SI
# rg_ann = MLPRegressor(hidden_layer_sizes=[20,50], solver='adam', learning_rate='adaptive', activation='relu'
#                      , batch_size='auto', max_iter=150, verbose=True, tol=0.00001
#                      , alpha=0.1, random_state=10)
# model_rg_ann = rg_ann.fit(X_train_scl, y_train)

# pickle.dump(model_rg_ann, open("modelo_treinado.sav", 'wb'))

model_rg_ann = pickle.load(open("modelo_treinado.sav", 'rb'))

# PREDIÇÃO
y_test_pred_rg_ann = model_rg_ann.predict(X_test_scl)

print("RMSE: ", mean_squared_error(y_test, y_test_pred_rg_ann, squared=False))
print("MSE: ", mean_squared_error(y_test, y_test_pred_rg_ann, squared=True))
print("MAE: ", mean_absolute_error(y_test, y_test_pred_rg_ann))

# Apenas acuracia abaixo
from sklearn.metrics import accuracy_score

pred_offset = 0
true_offset = 0

y_train_pred_rg_ann = model_rg_ann.predict(X_train_scl)
yb_train_pred = classifica(y_train_pred_rg_ann, pred_offset)
yb_train_true = classifica(y_train, true_offset)
print('Acuracia treino:', '{:0.2f}%'.format(accuracy_score(yb_train_true, yb_train_pred)*100))

yb_test_pred = classifica(y_test_pred_rg_ann, pred_offset)
yb_test_true = classifica(y_test, true_offset)
print('Acuracia teste:', '{:0.2f}%'.format(accuracy_score(yb_test_true, yb_test_pred)*100))


categorias = ["Diabetic", "Prediabetic", "Healthy"]
labels = [2,1,0]

plot_cfm(yb_test_true, yb_test_pred, labels, categorias)