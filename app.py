from turtle import title
from flask import Flask, render_template, redirect
from src.source import Cargar_Modelo, clasificarNoticia
from flask import request


app = Flask(__name__)
modelo = "Algo"

@app.route('/')
def index():
    return render_template('index.html', title="Clasificador - inicio")

@app.route('/documentation')
def documentation():
    return render_template('documentation.html', title="Documentación")

@app.route('/clasificarNoticia', methods=['POST', 'GET'])
def clasificar():
    if request.method == 'POST':
        modelo, bagOfWords = Cargar_Modelo()
        noticia = request.form['data']
        response = clasificarNoticia(modelo,noticia,bagOfWords)
        return render_template(
            "result.html", 
            response = response[0], 
            title="Clasificador - response",
            notice = noticia,
        )
    return render_template("index.html")



if __name__ == '__main__':
    app.run(debug=True)