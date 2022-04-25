import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn import datasets

scaler = StandardScaler()

ipd = 5.7
idb = 6.5

def analise(idade, creat_sangue, glic_sangue, hemo_baso, hemo_chcm, hemo_hcm, hemo_ht, hemo_leuc, hemo_linf, hemo_mono, hemo_mpv, hemo_plaq, hemo_rdw, hemo_seg, hemo_vcm):
    df_predict_data = {
                'Idade': [idade],
                'CREATININA EM SANGUE': [creat_sangue],
                'GLICOSE EM SANGUE': [glic_sangue],
                'HEMOGRAMA_BASO%': [hemo_baso],
                'HEMOGRAMA_CHCM': [hemo_chcm],
                'HEMOGRAMA_HCM': [hemo_hcm],
                'HEMOGRAMA_HT': [hemo_ht],
                'HEMOGRAMA_LEUC': [hemo_leuc],
                'HEMOGRAMA_LINFO%': [hemo_linf],
                'HEMOGRAMA_MONO%': [hemo_mono],
                'HEMOGRAMA_MPV': [hemo_mpv],
                'HEMOGRAMA_PLAQ': [hemo_plaq],
                'HEMOGRAMA_RDW': [hemo_rdw],
                'HEMOGRAMA_SEG%': [hemo_seg],
                'HEMOGRAMA_VCM': [hemo_vcm]
            }

    df_predict = pd.DataFrame(data=df_predict_data)

    df_predict_scl = scaler.fit_transform(df_predict)

    trained_model = pickle.load(open("modelo_treinado.sav", 'rb'))

    predict_result = trained_model.predict(df_predict_scl)

    return predict_result[0]

def classifica(hemo_glic):
    if hemo_glic < 5.7:
        return "saudável"
    elif hemo_glic < 6.5:
        return "pré-diabético"
    else:
        return "diabético"

hemo_glic = analise(
    idade = 46.0,
    creat_sangue = 1.06,
    glic_sangue = 96.0,
    hemo_baso = 0.9,
    hemo_chcm = 35.4,
    hemo_hcm = 4.75,
    hemo_ht = 42.0,
    hemo_leuc = 7030.0,
    hemo_linf = 28.1,
    hemo_mono = 8.1,
    hemo_mpv = 8.8,
    hemo_plaq = 169.0,
    hemo_rdw = 13.7,
    hemo_seg = 59.8,
    hemo_vcm = 88.5
)

resultado = classifica(hemo_glic)

print("Você é {}, sua hemoglobina glicada é: {}".format(resultado, hemo_glic))