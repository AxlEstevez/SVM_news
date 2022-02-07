"""
Proyecto clasificación de noticias
"""
import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, f1_score, precision_score, accuracy_score
import joblib
#Funcion Leer datos de Dataframe previamente tratado
def leerDatos():
    df=pd.read_json('datos.json', lines=True)
    return df

def limpieza(df):
    # Seleccion de columnas a usar (categoria, short_description)
    df1 = df.iloc[:,[0,4]]
    print('\n\nColumnas selecionadas para trabajar\n\n',df1)
    
    #verificar la existencia de valores nulos
    print('\n\nVer si hay valores nulos:\n\n',df1.isnull().sum())
    
    # Verificar existencia de cadenas vacias
    df_cadenavacia=df1['short_description']==''
    #print(df_cadenavacia)
    print('\n\nRegistros con cadenas vacias encontrados:\n\n',df1[df_cadenavacia])
    
    #Eliminar registros con cadenas vacias
    df_sinCadenasVacias=df1.drop(df1[df1["short_description"]==""].index)
    print('\n\nData set sin cadenas vacias\n\n',df_sinCadenasVacias)
    
    # volver a validar que ya no haya registros con cadenas vacias
    dfp=df_sinCadenasVacias['short_description']==''
    print('\n\nVerificacion de que ya no se encuentran cadenas vacias\n\n',df_sinCadenasVacias[dfp])
    return df_sinCadenasVacias

def balancear_dataset(df_sinCadenasVacias):
    print('\n\nNumero de registros que existen por Categoria\n\n',df_sinCadenasVacias.value_counts('category'))
    
    balancear = RandomUnderSampler(random_state=0)
    df_balanceado, df_balanceado['category']=balancear.fit_resample(df_sinCadenasVacias[['short_description']],df_sinCadenasVacias['category'])
    print('\n\nDataset balanceado\n\n',df_balanceado)

    print('\n\nVerificar que el dataset se encuentre balanceado\n\n',df_balanceado.value_counts('category'))
    return df_balanceado

def GuardarDataset(df_balanceado):
    df_balanceado.to_json('prueba_Balanceado_Guardado.json' )

def Test_Train(df):
    X = df.iloc[:,0]
    y = df.iloc[:,1].values
    print('\n\nValores de X\n\n',X.head(),"Valores de y\n\n",y)
    
    #Obtener el conjunto de entrenamiento y de prueba
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
    print('Numero de registros X_train:',X_train.shape)
    print('Numero de registros X_test:',X_test.shape)
    return X_train, X_test, y_train, y_test

def Vectorizar(X_train, X_test):
    tfidf = TfidfVectorizer(stop_words='english')
    X_train_numeros=tfidf.fit_transform(X_train)
    X_test_numeros = tfidf.transform(X_test)

    #print('\n\nVocabulario de la bolsa de palabras\n\n',tfidf.vocabulary_)
    print('\n\nVisualizar cuantas columnas son ahora despues de separar las palabras\n\n',X_train_numeros.shape)
    print('\n\nVer matriz dispersa\n\n',pd.DataFrame.sparse.from_spmatrix(X_train_numeros,index=X_train.index,columns=tfidf.get_feature_names()))
    return X_train_numeros, X_test_numeros, tfidf

def EntrenarModelo(X_train_numeros, X_test_numeros, y_train, tfidf):
    svc = SVC()
    svc.fit(X_train_numeros, y_train)
    print('\n\nResultado de clasificacion\n\n')
    print(svc.predict(tfidf.transform(['In one text message, Jacqueline Ades allegedly told her date she d like to bathe in his blood.'])))
    print(svc.predict(tfidf.transform(['A nasty primary race has made the congressional candidate tell the difficult story of escaping her violent ex-husband'])))
    print(svc.predict(tfidf.transform(['The five-time all-star center tore into his teammates Friday night after Orlando committed 23 turnovers en route to losing'])))
    print(svc.predict(tfidf.transform(['Maria married the best singer in the United States'])))
    print(svc.predict(tfidf.transform(['american football game in mexico'])))
    print(svc.predict(tfidf.transform(['E-cigarettes are also dangerous. The WHO revealed that the smoke produced by these devices is as unhealthy as the one that comes from common cigarettes'])))
    #Realizando de clasificaciones
    y_predsvc=svc.predict(X_test_numeros[0:1000])
    return svc, y_predsvc

def Metricas_Evaluacion(y_test,y_predsvc, X_test_numeros, svc):
    cmsvc=confusion_matrix(y_test[0:1000],y_predsvc)
    print('\n\nMatriz de confisión\n\n',cmsvc)
    
    f1scoresvc = f1_score(y_test[0:1000],y_predsvc, average = 'macro')
    print("\n\nF1-score: ",f1scoresvc)

    precisionsvc = precision_score(y_test[0:1000],y_predsvc, average ='macro')
    print("\n\nPrecision: ",precisionsvc)

    print("\n\nAccuracy_score:", accuracy_score(y_test[0:1000],y_predsvc))

    # ver cual es el score antes de guardar el modelo para despues ver si cuando se cargue de nuevo sigue siendo el mismo.
    print("\n\nScore:",svc.score(X_test_numeros, y_test))

def GuardarModelo(svc):
    joblib.dump(svc, 'prueba_modelo_entrenado_svc.pkl') # Guardo el modelo.


df = leerDatos()
print('\n\nData set completo\n\n',df)
print('\n\nDescripcion de dataset\n\n',df.describe())
df_sinCadenasVacias =limpieza(df)
df_balanceado=balancear_dataset(df_sinCadenasVacias)
GuardarDataset(df_balanceado)
X_train, X_test, y_train, y_test = Test_Train(df_balanceado)
X_train_numeros, X_test_numeros, tfidf = Vectorizar(X_train, X_test)
svc, y_predsvc =EntrenarModelo(X_train_numeros, X_test_numeros, y_train, tfidf)
Metricas_Evaluacion(y_test,y_predsvc, X_test_numeros, svc)
GuardarModelo(svc)