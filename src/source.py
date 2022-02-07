import joblib

def Cargar_Modelo():
    """Cargar el modelo ya entrenado y la bolsa de palabras
    ya vectorizada.
    """
    svc = joblib.load('./src/modelo_entrenado_svc.pkl') # Carga del modelo.
    bg = joblib.load('./src/bolsa.pkl')
    return svc, bg

#Clasificar la noticia
def clasificarNoticia(svc, noticia,tfidf):
    """Clasificas una noticia

    Args:
        svc (plk): modelo entrenado
        noticia (string): noticia a clasificar
        tfidf (plk): [bolsa de palabras]

    Returns:
        array: clasificación de la noticia
    """
    tipoNoticia=svc.predict(tfidf.transform([noticia]))
    return tipoNoticia