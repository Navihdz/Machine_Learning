import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

def categorizar():
    #read a csv
    df_categoria = pd.read_csv("databaseFromSQL.csv", index_col=0)

    '''Hacemos los labels considerando personas analfabetas, sin estudios y primaria sin terminar como 0, 
    personas con primaria, secundaria incompleta y secundaria como 0.5 y personas con mayor o igual a posbasica como 1.
    Nota: hay bastantes datos faltantes en cada una de las categorias, para eviar que desechar todo el row, consideraré
    el total de población de mayores a 15 años y veré si con los datos que hay es suficiente para poder escoger una 
    categoria y sino desecho el row'''

    # reemplazar los "*" por 0 solo para las columnas de educacion
    educ_cols = ['P15YM_AN', 'P15YM_SE', 'P15PRI_IN', 'P15PRI_CO', 'P15SEC_IN', 'P15SEC_CO', 'P18YM_PB']
    df_categoria[educ_cols] = df_categoria[educ_cols].replace("*", "0")
    #drop rows with N/D
    df_categoria=df_categoria.replace(["N/D"], "0")
    # llenar todos los valores faltantes con cero
    df_categoria = df_categoria.fillna(0)
    # elimnar todos los rows con "*" en la columna P_15YMAS
    df_categoria = df_categoria[~df_categoria['P_15YMAS'].str.contains('\*')]
    # elimina todos los rows con 0 en la columna P_15YMAS
    df_categoria = df_categoria[~df_categoria['P_15YMAS'].str.contains('0')]

    # crear una función para calcular la categoría
    def calcular_categoria(row):
        lista_educacion = row[educ_cols].astype(int).tolist()
        total_not_null = sum(lista_educacion)
        poblacion_total = int(row['P_15YMAS'])
        if total_not_null >= 0.8*poblacion_total:
            index = lista_educacion.index(max(lista_educacion))
            if index in [0, 1, 2]:
                return 0
            elif index in [3, 4, 5]:
                return 0
            elif index == 6:
                return 1
        else:
            return None

    # aplicar la función a cada fila del dataframe
    df_categoria['categoria'] = df_categoria.apply(calcular_categoria, axis=1)
    # eliminar las filas con valor nulo en la columna "categoria"
    df_categoria = df_categoria.dropna(subset=['categoria'])
    # guardar el dataframe en un archivo csv
    #df_categoria.to_csv("database_categorical.csv", index=False)

    return df_categoria

def clean_nans(df_categoria):
    '''buscamos columnas con mas nans y las eliminamos primero, y luego procedemos con los rows'''
    educ_cols = ['P15YM_AN', 'P15YM_SE', 'P15PRI_IN', 'P15PRI_CO', 'P15SEC_IN', 'P15SEC_CO', 'P18YM_PB']
    df_categoria.drop(columns=educ_cols, inplace=True)

    #elimina todos los rows con 0 en la columna P_15YMAS
    df_categoria = df_categoria[~df_categoria['P_15YMAS'].isin([0])]

    #contar * por columna
    df_categoria=df_categoria.replace(["*"], [np.nan])
    nans_por_columna = df_categoria.isnull().sum(axis=0)

    # crear lista de números enteros para el eje x
    x_indices = range(len(df_categoria.columns))

    # plotear en un gráfico de barras
    plt.bar(x_indices, nans_por_columna)
    plt.title('Número de NaNs por columna')
    plt.xlabel('Columnas')
    plt.ylabel('Número de NaNs')
    plt.savefig('generated_images\CleaningData_nans_por_columna.png')

    #si removieramos rows with NaN directamente, perderíamos muchos datos
    #df_sin_nans=df_categoria.dropna()

    # por lo tanto eliminamos columnas con numero de nans> a 15 mil y luego eliminamos rows con nans, 
    # así evitamos perder una gran cantidad de muestras

    #eliminar columnas con numeros de NaNs mayores a 1000
    df_cleaned_nans_outliers = df_categoria.drop(columns=nans_por_columna[nans_por_columna > 5000].index)
    #elimina las columnas donde el numero de 0 es mayor al 60% de la columna
    df_cleaned_nans_outliers = df_cleaned_nans_outliers.drop(columns=df_cleaned_nans_outliers.columns[(df_cleaned_nans_outliers == '0').sum() > 0.6*len(df_cleaned_nans_outliers)])
    #quita la "A" de los rows que tienen algunas columnas en 'AGEB'
    df_cleaned_nans_outliers['AGEB'] = df_cleaned_nans_outliers['AGEB'].str.replace('A', '')
    #elimina los rows con NaNs
    df_sin_nans=df_cleaned_nans_outliers.dropna()
    #df_sin_nans.to_csv("database_jalisco_cleaned.csv")

    #contar los "1" por columna en 'categoria'
    num_data_categoria=df_sin_nans['categoria'].value_counts()
    print('Número de datos en categoría 0: ', num_data_categoria[0])
    print('Número de datos en categoría 1: ', num_data_categoria[1])
    #covertimos de string a double
    df_sin_nans = df_sin_nans.astype(float)
    #df_sin_nans = df_sin_nans.astype()

    return df_sin_nans, num_data_categoria



def nearest_neighbors(X, n_neighbors=5):
    # crear un objeto NearestNeighbors para buscar los 5 vecinos más cercanos
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='kd_tree',metric='euclidean').fit(X)
    # calcular la distancia entre los vecinos
    distances, indices = nbrs.kneighbors(X)
    return distances, indices
def smote(X,num_samples=10):
    # calcular las distancias y los índices de los vecinos más cercanos, la primer distancia es 0 porque es el mismo punto
    #por lo tanto no se toma en cuenta al igual que el primer indice
    distances, indices = nearest_neighbors(X)
    # seleccionamos aleatoriamente alguno de los vecinos más cercanos, sin incluir el primer vecino
    X_syntehtic = np.zeros((len(X), len(X[0])))
    for i in range(len(X)):
        random=np.random.randint(low=1,high=5,size=1)   #genera un numero aleatorio entre 1 y 4
        index=indices[i][random]
        # calculamos el punto medio entre el punto i y el punto aleatorio
        random_number = np.random.random()    #genera un numero aleatorio entre 0 y 1
        X_syntehtic[i] = X[i] + (X[index] - X[i]) * random_number

    #seleccionamos solamente el numero de muestras que queremos aleatoriamente
    random=np.random.randint(low=0,high=len(X_syntehtic),size=num_samples)   #genera num_samples aleatorios entre 0 y len(X_syntehtic)
    X_syntehtic=X_syntehtic[random]
    
    return X_syntehtic 

def rand_subsampling(X, num_final_samples=1000):
    # Generamos una permutación aleatoria de los números del 0 al len(X)
    random_permutation = np.random.permutation(len(X))
    # Tomamos los primeros num_final_samples elementos de la permutación aleatoria
    random_numbers = random_permutation[:num_final_samples]
    X_subsampling=X[random_numbers]
    return X_subsampling



def main():
    df_categoria = categorizar()
    df_sin_nans, num_data_categoria=clean_nans(df_categoria)

    #separamos datos de las dos categorias y los convertimos a array
    #categoria_0_array=df_sin_nans[df_sin_nans['categoria']==0]
    #categoria_1_array=df_sin_nans[df_sin_nans['categoria']==1]
    categoria_0_array=(df_sin_nans[df_sin_nans['categoria']==0]).to_numpy()
    categoria_1_array=(df_sin_nans[df_sin_nans['categoria']==1]).to_numpy()

    #generamos datos sinteticos para categorias 0
    categoria_0_array_syntehtic=smote(categoria_0_array, num_samples=13970)

    #observamos que son muy parecidos los promedios y desviaciones estandar de categoria 0 y categoria 0 datos sinteticos
    print('--------promedios y desviaciones estandar de categoria 0 y categoria 0 datos sinteticos--------')
    print('promedio de categoria 0: ',categoria_0_array.mean(axis=0)[:5])
    print('promedio de categoria 0 datos sinteticos: ',categoria_0_array_syntehtic.mean(axis=0)[:5])
    print('desviacion de categoria 0: ',categoria_0_array.std(axis=0)[:5])
    print('desviacion de categoria 0 datos sinteticos: ',categoria_0_array_syntehtic.std(axis=0)[:5])


    #unimos los datos sinteticos con los datos originales para categoria 0
    categoria_0_array_syntehtic=np.concatenate((categoria_0_array,categoria_0_array_syntehtic),axis=0)

    #dado que aun tenemos una diferencia entre las dos clases, se utilizara un subsampling en clase 1 para no generar
    # mas del doble de datos synteticos (Data reduction by randomization subsampling)
    categoria_1_array_subsampling=rand_subsampling(categoria_1_array, num_final_samples=27940)

    #union de los datos sinteticos con los datos originales para categoria 0 y categoria 1
    training_data=np.concatenate((categoria_0_array_syntehtic,categoria_1_array_subsampling),axis=0)

    #normalizamos los datos
    scaler = StandardScaler()
    scaler.fit(training_data)
    training_data=scaler.transform(training_data)

    #regresamos a dataframe
    training_data=pd.DataFrame(training_data)
    #le agregamos los nombres de las columnas
    training_data.columns=df_sin_nans.columns
    #guardamos df
    training_data.to_csv("training_data.csv")


    #guardamos los datos de entrenamiento
    #np.savetxt("training_data.csv", training_data, delimiter=",")



if __name__ == "__main__":
    main()



