from flask import Flask, render_template
 
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
        codigo_pagina = render_template("resultado.html")
        return codigo_pagina
