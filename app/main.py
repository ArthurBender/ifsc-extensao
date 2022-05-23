from flask import Flask, render_template, request
from app.executa_analise import analise

app = Flask(__name__)
 
@app.route("/")
def tela_inicial():
        codigo_pagina  = render_template("html_basic.html", custom_css="tela_inicial")
        codigo_pagina += render_template("tela_inicial.html")
        codigo_pagina += render_template("footer.html")
        
        return codigo_pagina

@app.route("/consulta")
def consulta():
        codigo_pagina  = render_template("html_basic.html", custom_css="consulta")
        codigo_pagina += render_template("consulta.html")
        codigo_pagina += render_template("footer.html")

        return codigo_pagina

@app.route("/resultado")
def resultado():
        # hemo_glic = analise(
        # idade = 46.0,
        # creat_sangue = 1.06,
        # glic_sangue = 96.0,
        # hemo_baso = 0.9,
        # hemo_chcm = 35.4,
        # hemo_hcm = 4.75,
        # hemo_ht = 42.0,
        # hemo_leuc = 7030.0,
        # hemo_linf = 28.1,
        # hemo_mono = 8.1,
        # hemo_mpv = 8.8,
        # hemo_plaq = 169.0,
        # hemo_rdw = 13.7,
        # hemo_seg = 59.8,
        # hemo_vcm = 88.5
        # )
        
        hemo_glic = analise(
        idade = request.args.get("idade"),
        creat_sangue = request.args.get("creat_sangue"),
        glic_sangue = request.args.get("glic_sangue"),
        hemo_baso = request.args.get("hemo_baso"),
        hemo_chcm = request.args.get("hemo_chcm"),
        hemo_hcm = request.args.get("hemo_hcm"),
        hemo_ht = request.args.get("hemo_ht"),
        hemo_leuc = request.args.get("hemo_leuc"),
        hemo_linf = request.args.get("hemo_linf"),
        hemo_mono = request.args.get("hemo_mono"),
        hemo_mpv = request.args.get("hemo_mpv"),
        hemo_plaq = request.args.get("hemo_plaq"),
        hemo_rdw = request.args.get("hemo_rdw"),
        hemo_seg = request.args.get("hemo_seg"),
        hemo_vcm = request.args.get("hemo_vcm")
        )

        codigo_pagina  = render_template("html_basic.html", custom_css="resultado")
        codigo_pagina += render_template("resultado.html", hemo_glic=round(hemo_glic, 2))
        codigo_pagina += render_template("footer.html")
        
        return codigo_pagina
