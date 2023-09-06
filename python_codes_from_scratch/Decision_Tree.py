import numpy as np
import pandas as pd
import mlflow

#esto es para poder crear la grafica del arbol
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import write_dot
import jax.numpy as jnp
from jax import jit
import jax

import pandas as pd
import jax.numpy as jnp
from jax import jit
from jax.scipy.special import expit


def get_split(data_array,labels):
    gini_por_feature = []
    best_splits_parameter = []
    for feature in range(data_array.shape[1]):
        feature_array = data_array[:,feature]

        #lista ordenada de mayor a menor de los valores unicos de la columna
        sorted_array = np.unique(feature_array)
        if len(sorted_array)> 0.5*len(feature_array):
            sorted_array = sorted_array[::5]    #cremos una nueva lista saltando de 5 en 5 para reducir el numero de valores a probar
        elif len(sorted_array)> 0.3*len(feature_array) and len(sorted_array)<= 0.5*len(feature_array):
            sorted_array = sorted_array[::3]
        elif len(sorted_array)> 0.1*len(feature_array) and len(sorted_array)<= 0.3*len(feature_array):
            sorted_array = sorted_array[::2]
        else:
            sorted_array = sorted_array
        



        #crear lista solo con los indices pares de feature array, esto con la finalidad de 
        # reducir el numero de valores a probar
        sorted_array = sorted_array[::3]
        
        gini_list = []
        for i in range(1,len(sorted_array)):
            sorted_value=sorted_array[i]
            first_split=labels[feature_array<=sorted_value]
            second_split=labels[feature_array>sorted_value]
            gini_first_split=1-(np.sum((np.unique(first_split,return_counts=True)[1]/len(first_split))**2))
            gini_second_split=1-(np.sum((np.unique(second_split,return_counts=True)[1]/len(second_split))**2))
            gini_impurity=(len(first_split)/len(labels))*gini_first_split+(len(second_split)/len(labels))*gini_second_split
            gini_list.append(gini_impurity)
        
        #seleccionar el valor de la columna que minimiza el gini_impurity
        min_gini = np.min(gini_list)
        min_index = np.argmin(gini_list)
        min_value = sorted_array[min_index+1]   # se suma 1 porque no se toma en cuenta el primer valor de la lista ordenada
                                                #ya que en ese caso el gini_impurity es 0 osea todos los datos estan contenidos 
        gini_por_feature.append(min_gini)
        best_splits_parameter.append(min_value)
    
    #seleccionar la columna que minimiza el gini_impurity
    min_gini = np.min(gini_por_feature)
    min_index = np.argmin(gini_por_feature)
    min_value = best_splits_parameter[min_index]

    #print("El mejor split es: ",min_value," en la columna ",min_index)
    best_feature = min_index
    best_value = min_value
    return best_feature,best_value




def split_data(data_array,labels):
    #buscamos el mejor split usando la funcion get_split
    best_feature,best_value = get_split(data_array,labels)
    #best_feature,best_value = get_split(data_array,labels)

    #separamos los datos en dos grupos
    first_split = data_array[data_array[:,best_feature]<=best_value]
    second_split = data_array[data_array[:,best_feature]>best_value]
    #separamos las etiquetas en dos grupos
    first_labels = labels[data_array[:,best_feature]<=best_value]
    second_labels = labels[data_array[:,best_feature]>best_value]

    return first_split,first_labels,second_split,second_labels,best_feature,best_value



def train_tree(data_array,labels, max_depth, min_gini, current_depth=0,nodo_padre=0, nodo_hijo=0):
    #if current depth is 0 borramos el archivo csv node_data.csv (de existir) y creamos uno nuevo con los encabezados
    #donde iremos guardando los datos de cada nodo
    if current_depth==0:
        df_node_data=pd.DataFrame(columns=['Nodo padre','Nodo hijo','Caracteristica','Valor','Gini impurity','div 1 No in class 1','div 1 No in class 2','div 2 No in class 1','div 2 No in class 2'])
        df_node_data.to_csv('node_data.csv', index=False)
    

    node_data=[]
    #separamos los datos en dos grupos
    first_split,first_labels,second_split,second_labels, best_feature, best_value = split_data(data_array,labels)
    #calculamos el gini_impurity de cada grupo
    gini_first = 1-(np.sum((np.unique(first_labels,return_counts=True)[1]/len(first_labels))**2))
    gini_second = 1-(np.sum((np.unique(second_labels,return_counts=True)[1]/len(second_labels))**2))
    gini_impurity=(len(first_labels)/len(labels))*gini_first+(len(second_labels)/len(labels))*gini_second
    #se guardan los datos de cada nodo
    #node_data.append([first_split,first_labels,second_split,second_labels,best_feature, best_value,gini_first,gini_second])
    #node_data={'Nodo padre':nodo_padre,'Nodo hijo':current_depth, 'Caracteristica': best_feature, 'Valor': best_value, 'Gini impurity': gini_impurity}


    
    node_data.append({
    'Nodo padre':nodo_padre,
    'Nodo hijo':nodo_hijo, 
    'Caracteristica': best_feature, 
    'Valor': best_value, 
    'Gini impurity': gini_impurity,
    'div 1 No in class 1': np.sum(first_labels==0),
    'div 1 No in class 2': np.sum(first_labels==1),
    'div 2 No in class 1': np.sum(second_labels==0),
    'div 2 No in class 2': np.sum(second_labels==1)
    })

    #añadimos los datos de cada nodo al dataframe
    df_node_data=pd.DataFrame(data=node_data, columns=['Nodo padre','Nodo hijo','Caracteristica','Valor','Gini impurity','div 1 No in class 1','div 1 No in class 2','div 2 No in class 1','div 2 No in class 2'], index=range(len(node_data)))    
    
    
    #añadimos los datos de cada nodo al dataframe
    #df_node_data=pd.DataFrame(data=node_data, columns=['Nodo padre','Nodo hijo','Caracteristica','Valor','Gini impurity'], index=range(max_depth))

    #guardar los datos de cada nodo en un archivo csv
    #df = pd.DataFrame(node_data)
    df_node_data.to_csv('node_data.csv', mode='a', header=False, index=False)
    #print('current depth',current_depth)
    #print('gini_first: ',gini_first)
    #print('gini_second: ',gini_second)
    #print('gini_impurity: ',gini_impurity)
    #print('No in class 1: ',np.sum(first_labels==0))
    #print('No in class 2: ',np.sum(first_labels==1))
    #print('No in class 1: ',np.sum(second_labels==0))
    #print('No in class 2: ',np.sum(second_labels==1))

    #si el gini_impurity de algun grupo es menor al minimo gini_impurity se detiene el entrenamiento solo de ese nodo
    #tamaño de first_split
    if gini_first<min_gini or gini_second<min_gini or current_depth==max_depth-1 or len(first_split)< 40 or len(second_split)<40: #or first_split menor a 40
        #se para el entrenamiento de ese nodo y se sigue con el siguiente nodo (for)
        #print('se detuvo el entrenamiento de este nodo, por limite de gini_impurity o por limite de profundidad')
        #que no haga nada
        pass
    
    #si i es igual a max_depth se detiene el entrenamiento de ese nodo y se sigue con el siguiente nodo (for)
    #elif current_depth==max_depth-1:
        #print('se detuvo el entrenamiento de este nodo por limite de profundidad')
    
    #si no se cumple la condicion se vuelve a llamar a la funcion train_tree
    else:
        #read csv para obtener el ultimo valor de nodo_hijo
        df_node_data=pd.read_csv('node_data.csv')
        new_nodo_hijo=df_node_data['Nodo hijo'].max()+1
        train_tree(first_split,first_labels,max_depth,min_gini,current_depth+1,nodo_padre=nodo_hijo, nodo_hijo=new_nodo_hijo)

        #read csv para obtener el ultimo valor de nodo_hijo
        df_node_data=pd.read_csv('node_data.csv')
        new_nodo_hijo=df_node_data['Nodo hijo'].max()+1
        train_tree(second_split,second_labels,max_depth,min_gini,current_depth+1,nodo_padre=nodo_hijo, nodo_hijo=new_nodo_hijo)
        

    return node_data
    


            
def plt_tree():
        # Leer los datos del archivo CSV
    df = pd.read_csv('node_data.csv')
    plt.figure(figsize=(10, 5))

    # Crea un nuevo gráfico dirigido
    G = nx.DiGraph()

    # Agrega todos los nodos al gráfico
    for node_id in pd.concat([df['Nodo padre'], df['Nodo hijo']]).unique():
        G.add_node(node_id)
        
    # Agrega todas las aristas al gráfico
    for _, row in df.iterrows():
        parent = row['Nodo padre']
        child = row['Nodo hijo']
        feature = row['Caracteristica']
        value = row['Valor']
        impurity = row['Gini impurity']

        G.add_edge(parent, child, label=f"{feature} <= {value}\nGini impurity: {impurity:.3f}")
        #G.add_edge(parent, child)

   

    #pos = nx.kamada_kawai_layout(G)
    pos = nx.planar_layout(G,center=(10,10))
    labels = nx.get_edge_attributes(G, 'label') 
    #tamaño de rows
    nx.draw_networkx(G, pos, with_labels=True, node_size=200, font_size=10)
    #nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    nx.draw_networkx_edges(G, pos, width=0.5, arrowsize=0.5)
    plt.show()

 

def plt_tree_2():
    # Leer los datos del archivo CSV
    df = pd.read_csv('node_data.csv')
    plt.figure(figsize=(20, 10))

    # Crea un nuevo gráfico dirigido
    G = nx.DiGraph()

    # Agrega todos los nodos al gráfico
    for node_id in pd.concat([df['Nodo padre'], df['Nodo hijo']]).unique():
        G.add_node(node_id)

    # Agrega los bordes al gráfico
    for index, row in df.iterrows():
        G.add_edge(row['Nodo padre'], row['Nodo hijo'])

    pos = nx.planar_layout(G,scale=1)
    #pos = nx.kamada_kawai_layout(G)
    
    # Dibujar los nodos con forma cuadrada y etiquetas
    nx.draw_networkx_nodes(G, pos=pos, node_shape='s', node_size=700)
    labels = {}
    for node_id in G.nodes():
        labels[node_id] = str('Nodo: '+str(node_id))
        impurity = df[df['Nodo hijo'] == node_id]['Gini impurity'].values[0]
        caracteristica = df[df['Nodo hijo'] == node_id]['Caracteristica'].values[0]
        valor=df[df['Nodo hijo'] == node_id]['Valor'].values[0]
        #sumo el numero de muestras en div 1 No in class 1	div 1 No in class 2	div 2 No in class 1	div 2 No in class 2
        numero_muestras_nodo=df[df['Nodo hijo'] == node_id]['div 1 No in class 1'].values[0]+df[df['Nodo hijo'] == node_id]['div 1 No in class 2'].values[0] + df[df['Nodo hijo'] == node_id]['div 2 No in class 1'].values[0] + df[df['Nodo hijo'] == node_id]['div 2 No in class 2'].values[0]

        #solo 3 decimales
        labels[node_id]=str('ginni: '+str(round(impurity,2)) +'\n feature: '+str(caracteristica)+ '<= '+str(round(valor,4)) +'\n muestras: '+str(numero_muestras_nodo))
    nx.draw_networkx_labels(G, pos={k: (x-0.05, y+0.06) for k, (x, y) in pos.items()}, labels=labels, font_size=10, font_color='black', font_weight='bold')

    # Dibuja el gráfico con la posición en forma de árbol
    nx.draw(G, pos=pos, with_labels=True)

    #save image
    plt.savefig('generated_images\plot_tree.png')



def find_next_node(df, X_to_predict , current_nodo_padre, current_nodo_hijo):
    #buscamos el nodo padre
    nodo_padre=df[df['Nodo hijo']==current_nodo_padre]['Nodo padre'].values[0]
    #buscamos la caracteristica
    caracteristica=df[df['Nodo hijo']==current_nodo_hijo]['Caracteristica'].values[0]
    #buscamos el valor
    valor=df[df['Nodo hijo']==current_nodo_hijo]['Valor'].values[0]

    #buscamos el hacia que nodo hijo se va a ir
    if X_to_predict[caracteristica]<=valor:
            #buscamos el nodo hijo con el valor mas bajo pero que tenga el nodo padre= lista_nodos[i]
            nodo_hijo=df[df['Nodo padre']==current_nodo_hijo]['Nodo hijo'].min()

            #previene de regresar en bucle a nodo raiz
            if nodo_hijo==0:
                nodo_hijo+=1

    else:
        #buscamos el nodo hijo con el valor mas alto
        nodo_hijo=df[df['Nodo padre']==current_nodo_hijo]['Nodo hijo'].max()
    
    current_nodo_padre=nodo_padre
    current_nodo_hijo=nodo_hijo

    return nodo_padre,nodo_hijo

def find_next_node_optimized(df, X_to_predict , current_nodo_padre, current_nodo_hijo):
    # Buscamos el nodo padre
    nodo_padre = df.loc[df['Nodo hijo'] == current_nodo_padre, 'Nodo padre'].iloc[0]

    # Buscamos la característica y el valor
    row1 = df.loc[df['Nodo hijo'] == current_nodo_hijo]
    caracteristica = row1.at[row1.index[0], 'Caracteristica']
    valor = row1.at[row1.index[0], 'Valor']

    # Buscamos el nodo hijo al que se va a ir
    filtro = (df['Nodo padre'] == current_nodo_hijo)
    nodo_hijo = df.loc[filtro, 'Nodo hijo'].min() if X_to_predict[caracteristica] <= valor else df.loc[filtro, 'Nodo hijo'].max()

    # Prevenimos de regresar en bucle al nodo raíz
    if nodo_hijo == 0:
        nodo_hijo += 1

    return nodo_padre, nodo_hijo



#------Metricas----------------
def precision_jax(y, y_hat):
    """
    precision
    args:
        y: Real Labels
        y_hat: estimated labels
    return TP/(TP+FP)
    """
    TP = jnp.sum((y > 0) & (y_hat > 0))
    FP = jnp.sum((y <= 0) & (y_hat > 0))

    #evitar division por cero
    precision_cpu = jax.lax.cond(
        TP + FP == 0,
        lambda _: 0.0,
        lambda _: TP / (TP + FP),
        operand=None,
    )
    

    return float(precision_cpu)


def recall_jax(y, y_hat):
    """
        recall
        args:
            y: Real Labels
            y_hat: estimated labels
        return TP/(TP+FN)
    """
    TP = jnp.sum((y > 0) & (y_hat > 0))
    FN = jnp.sum((y > 0) & (y_hat <= 0))

    #evitar division por cero
    recall_cpu = jax.lax.cond(
        TP + FN == 0,
        lambda _: 0.0,
        lambda _: TP / (TP + FN),
        operand=None,
    )

    return float(recall_cpu)

    
def accuracy_jax(y, y_hat):
    """
        accuracy
        args:
            y: Real Labels
            y_hat: estimated labels
        return  TP +TN/ TP +FP +FN+TN
    """
    TP = jnp.sum((y > 0) & (y_hat > 0))
    FP = jnp.sum((y <= 0) & (y_hat > 0))
    FN = jnp.sum((y > 0) & (y_hat <= 0))
    TN = jnp.sum((y <= 0) & (y_hat <= 0))
    
    #evitar division por cero         
    if (TP+FP+TN+FN)==0:
        return 0
    else:
        accuracy_cpu = jit(lambda x: x, device=jax.devices("cpu")[0])((TP+TN)/(TP+FP+TN+FN))
        return float(accuracy_cpu)                                              



    



def prediccion(X_to_predict, Y_labels=None):

    # Leer los datos del archivo CSV
    df = pd.read_csv('node_data.csv')
    #ordenamos los datos por el nodo padre pero tambien se ordenan los nodos hijos
    df=df.sort_values(by=['Nodo padre'])
    #para cala row de X_to_predict
    prediccion=[]
    for row in range(len(X_to_predict)):
        row_to_predict=X_to_predict[row]
        #inicializamos para el primer nodo raiz
        current_nodo_padre=0
        current_nodo_hijo=0
        while True:
            #si solo tenemos un nodo osea un stump entonces solo se hace la prediccion y no se busca el siguiente nodo
            if len(df['Nodo padre'].unique())!=1:
                current_nodo_padre,current_nodo_hijo=find_next_node(df, row_to_predict , current_nodo_padre, current_nodo_hijo)
            
            #si el current hijo no esta en la lista de nodos hijos entonces es una hoja
            if current_nodo_hijo not in df['Nodo padre'].values or len(df['Nodo padre'].unique())==1:
                #vemos caracteristica y valor
                caracteristica=df[df['Nodo hijo']==current_nodo_hijo]['Caracteristica'].values[0]
                valor=df[df['Nodo hijo']==current_nodo_hijo]['Valor'].values[0]
                #calido si es menor o mayor
                if row_to_predict[caracteristica]<=valor:
                    #vemos que clase es la que tiene mas cantidad de datos en df['div 1No in class 1'] y df['div 1 No in class 2'] en ese nodo
                    if df[df['Nodo hijo']==current_nodo_hijo]['div 1 No in class 1'].values[0]>df[df['Nodo hijo']==current_nodo_hijo]['div 1 No in class 2'].values[0]:
                        prediccion.append(0)
                    else:
                        prediccion.append(1)
                    break
                else:
                    #vemos que clase es la que tiene mas cantidad de datos en df['div 2 No in class 1'] y df['div 2 No in class 2'] en ese nodo
                    if df[df['Nodo hijo']==current_nodo_hijo]['div 2 No in class 1'].values[0]>df[df['Nodo hijo']==current_nodo_hijo]['div 2 No in class 2'].values[0]:
                        prediccion.append(0)
                    else:
                        prediccion.append(1)
                    break
    
    #if exist  y_labels
    if Y_labels is not None:
        #metrics
        #convert to jnp
        Y_labels=jnp.array(Y_labels)
        prediccion=jnp.array(prediccion)

        precision=precision_jax(Y_labels, prediccion)
        recall=recall_jax(Y_labels, prediccion)
        accuracy=accuracy_jax(Y_labels, prediccion)
        return prediccion, precision, recall, accuracy
    else:
        return prediccion





def predict_single_row(row, df):
    current_nodo_padre = 0
    current_nodo_hijo = 0
    while True:
        if (current_nodo_hijo not in df['Nodo padre'].to_numpy()) or (len(df['Nodo padre'].unique()) == 1):
            row1 = df.loc[df['Nodo hijo'] == current_nodo_hijo]
            caracteristica = row1.at[row1.index[0], 'Caracteristica']
            valor = row1.at[row1.index[0], 'Valor']
            if row[caracteristica] <= valor:
                if row1.at[row1.index[0], 'div 1 No in class 1'] > row1.at[row1.index[0], 'div 1 No in class 2']:
                    return 0
                else:
                    return 1
            else:
                if row1.at[row1.index[0], 'div 2 No in class 1'] > row1.at[row1.index[0], 'div 2 No in class 2']:
                    return 0
                else:
                    return 1
        else:
            current_nodo_padre, current_nodo_hijo = find_next_node_optimized(df, row, current_nodo_padre, current_nodo_hijo)


def prediccion_optimized(X_to_predict, Y_labels=None):
    df = pd.read_csv('node_data.csv')
    df = df.sort_values(by=['Nodo padre'])
    prediccion = np.apply_along_axis(predict_single_row, 1, X_to_predict, df)
    # Usa funciones vectorizadas
    #if exist  y_labels
    if Y_labels is not None:
        #metrics
        #convert to jnp
        Y_labels=jnp.array(Y_labels)
        prediccion=jnp.array(prediccion)

        precision=precision_jax(Y_labels, prediccion)
        recall=recall_jax(Y_labels, prediccion)
        accuracy=accuracy_jax(Y_labels, prediccion)
        return prediccion, precision, recall, accuracy
    else:
        return prediccion

    


def main_tree_train(data_array,labels,data_val,labels_val, max_depth, min_gini):
    experiment_name = "DecisionTree"
    # Verificar si el experimento ya existe
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        # Si no existe, crear el experimento
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        # Si ya existe, obtener el experimento
        experiment_id = experiment.experiment_id

    with mlflow.start_run(experiment_id=experiment_id,run_name="DecisionTree") as run:
        #log params
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_gini", min_gini)
        numero_features=len(data_array[0])
        mlflow.log_param("NumberOfFeatures", numero_features)

        #train
        nodes_data=train_tree(data_array,labels, max_depth, min_gini)
        
        #prediction, precision, recall, accuracy=prediccion(data_array,labels)
        prediction, precision, recall, accuracy=prediccion_optimized(data_array,labels)
        
        plt_tree_2()
        print('----------------------Metricas con datos de entrenamiento----------------------')
        print("precision: ", precision)
        print("recall: ", recall)
        print("accuracy: ", accuracy)

        #log metrics
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("accuracy", accuracy)

        #predicciones validacion
        prediction_val, precision_val, recall_val, accuracy_val=prediccion_optimized(data_val,labels_val)
        print('----------------------Metricas con datos de validacion----------------------')
        print("precision: ", precision_val)
        print("recall: ", recall_val)
        print("accuracy: ", accuracy_val)

        #log metrics
        mlflow.log_metric("precision_validation", precision_val)
        mlflow.log_metric("recall_validation", recall_val)
        mlflow.log_metric("accuracy_validation", accuracy_val)

    
        
        #guardar imagen del arbol
        mlflow.log_artifact("generated_images\plot_tree.png")

        return prediction, precision, recall, accuracy


if __name__ == "__main__":
    nodes_data=train_tree(data_array,labels, max_depth, min_gini)
    clf=create_tree(node_data)
    plt_tree_2()
    prediccion(X_to_predict)
    prediction, precision, recall, accuracy=main_tree_train(data_array,labels,data_val,labels_val, max_depth, min_gini)
