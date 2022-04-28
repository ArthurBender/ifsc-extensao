from flask import Flask, render_template
from app.executa_analise import analise, classifica

app = Flask(__name__)
 
@app.route("/")
def tela_inicial():
        codigo_pagina = render_template("tela_inicial.html")
        return codigo_pagina

@app.route("/consulta")
def consulta():
        codigo_pagina = render_template("consulta.html")
        return codigo_pagina

@app.route("/resultado")
def resultado():
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

        codigo_pagina = render_template("resultado.html", hemo_glic=hemo_glic, resultado=resultado)
        return codigo_pagina
