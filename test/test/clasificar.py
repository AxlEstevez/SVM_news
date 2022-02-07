import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

#Funcion Leer datos de Dataframe previamente tratado
def leerDatos():
    df=pd.read_json('./df_Balanceado_Guardado.json')
    return df

# Sacar el conjunto de entrenamiento y prueba
def Test_Train(df):
    X = df.iloc[:,0]
    y = df.iloc[:,1].values
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
    return X_train, X_test, y_train, y_test

# Vectorizar con la funcion TfidfVectorizer
def Vectorizar(X_train, X_test):
    tfidf = TfidfVectorizer(stop_words='english')
    X_train_numeros=tfidf.fit_transform(X_train)
    X_test_numeros = tfidf.transform(X_test)
    return X_train_numeros, X_test_numeros, tfidf

#Cargar modelo previamente guardado
def Cargar_Modelo():
    svc = joblib.load('./src/modelo_entrenado_svc.pkl') # Carga del modelo.
    bg = joblib.load('./src/bolsa.pkl')
    return svc, bg

#Clasificar la noticia
def clasificarNoticia(svc, noticia,tfidf):
    tipoNoticia=svc.predict(tfidf.transform([noticia]))
    return tipoNoticia

#datos = leerDatos()
#X_train, X_test, y_train, y_test = Test_Train(datos)
#X_train_numeros, X_test_numeros, tfidf = Vectorizar(X_train, X_test)

#svc, tfidf = Cargar_Modelo()
#print(svc.score(X_test_numeros, y_test))

#noticia='In one text message, Jacqueline Ades allegedly told her date she d like to bathe in his blood.'
#clasificacion=clasificar(svc, noticia, tfidf)
#print(clasificacion)