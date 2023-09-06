import numpy as np
import Decision_Tree as dt
import pandas as pd
import jax.numpy as jnp
from jax import jit
import jax
import mlflow




def first_weights(X):
    return np.ones(len(X)) / len(X)

def update_weights(w_correct,w_incorrect,prediction,y):
    prediccion_correcta=(np.equal(prediction,y)).astype(int)
    prediccion_incorrecta=(np.not_equal(prediction,y)).astype(int)
    w_correct=w_correct * prediccion_correcta
    w_incorrect=w_incorrect * prediccion_incorrecta
    pesos_finales=w_correct + w_incorrect
    #normalizamos los pesos
    pesos_finales=pesos_finales / np.sum(pesos_finales)

    return pesos_finales

    

def amount_of_say(error):
    return 1/2 * np.log((1 - error) / error)


def total_error(y,prediction,w):
    return np.sum(w * (y != prediction))

def update_weights_correct(w,amount):
    return w * np.exp(-amount)

def update_weights_incorrect(w,amount):
    return w * np.exp(amount)

def vector_rango_pesos(w):
    vector_rangos_pesos=[]
    for i in range(len(w)):
        if i==0:
            vector_rangos_pesos.append(w[i])
        else:
            vector_rangos_pesos.append(w[i]+ vector_rangos_pesos[-1])
    return vector_rangos_pesos


def write_csv_stump(stump_data, amount_of_say):
    #if current depth is 0 borramos el archivo csv node_data.csv (de existir) y creamos uno nuevo con los encabezados
    #donde iremos guardando los datos de cada nodo
    df_node_data=pd.DataFrame(columns=['Stump','Amount of say','Caracteristica','Valor','Prediction div 1 menor que valor','Prediction div 2 mayor que valor'])
    df_node_data.to_csv('AdaBoost_stumps_data.csv', index=False)

    #guardamos los datos del nodo en el csv
    for i in range(len(stump_data)):
        #añadimos los datos de cada nodo al dataframe
        df_node_data=pd.DataFrame(data=stump_data[i], columns=['Nodo padre','Nodo hijo','Caracteristica','Valor','Gini impurity','div 1 No in class 1','div 1 No in class 2','div 2 No in class 1','div 2 No in class 2'], index=range(len(stump_data[i])))    

        #quitamos las colyumnas de nodo padre y nodo hijo y agregamos una columna para identificar el stump y su amount of say
        df_node_data=df_node_data.drop(columns=['Nodo padre','Nodo hijo'])
        #agregandolos al inicio del dataframe
        df_node_data.insert(0, 'Stump', i)
        df_node_data.insert(1, 'Amount of say', amount_of_say[i])

        #para div 1 y div 2 cual tiene mas elementos en cada clase y esa será la prediccion para ese stump
        div_1_no_in_class_1=df_node_data['div 1 No in class 1'].sum()
        div_1_no_in_class_2=df_node_data['div 1 No in class 2'].sum()
        div_2_no_in_class_1=df_node_data['div 2 No in class 1'].sum()
        div_2_no_in_class_2=df_node_data['div 2 No in class 2'].sum()
        if div_1_no_in_class_1 > div_1_no_in_class_2:
            df_node_data['Prediction div 1 menor que valor']=0
        else:
            df_node_data['Prediction div 1 menor que valor']=1

        if div_2_no_in_class_1 > div_2_no_in_class_2:
            df_node_data['Prediction div 2 mayor que valor']=0
        else:
            df_node_data['Prediction div 2 mayor que valor']=1

        #borramos columnas que ya no importan tanto en adaboost como gini impurity y el numero de elementos por clase
        df_node_data=df_node_data.drop(columns=['Gini impurity','div 1 No in class 1','div 1 No in class 2','div 2 No in class 1','div 2 No in class 2'])


        



        #guardar los datos de cada nodo en un archivo csv
        df_node_data.to_csv('AdaBoost_stumps_data.csv', mode='a', header=False, index=False)


def weighted_list(X,y,vector_rangos_pesos):
    nuevo_X=[]
    nuevo_y=[]
    for i in range(len(X)):
        # Generar número aleatorio entre 0 y 1
        random_number = np.random.rand()

        # Encontrar el índice del primer valor en la lista que es mayor que el número aleatorio
        index_of_next_value = np.searchsorted(vector_rangos_pesos, random_number, side='right')+1
        #maneja el caso especial donde el número aleatorio es mayor que el último valor en la lista
        if index_of_next_value > len(vector_rangos_pesos):
            index_of_next_value = len(vector_rangos_pesos)
        # Restar uno para obtener el índice del valor anterior
        index_of_previous_value = index_of_next_value - 1

        # Manejar el caso especial donde el número aleatorio es menor que el primer valor en la lista
        if index_of_previous_value < 0:
            index_of_previous_value = 0

        #anexar a la nueva lista el valor de X en el índice del valor anterior
        nuevo_X.append(X[index_of_previous_value])
        nuevo_y.append(y[index_of_previous_value])

    nuevo_X=np.array(nuevo_X)
    nuevo_y=np.array(nuevo_y)
    return nuevo_X,nuevo_y
        
def weighted_list_optimizada(X, y, vector_rangos_pesos):
    num_samples = len(X)
    # Generar índices aleatorios basados en los pesos
    weights = np.diff(np.concatenate(([0], vector_rangos_pesos)))
    idx = np.random.choice(np.arange(len(weights)), size=num_samples, p=weights)

    # Seleccionar elementos correspondientes de X e y usando la indexación de matrices
    nuevo_X = X[idx]
    nuevo_y = y[idx]

    return nuevo_X, nuevo_y



# creamos stumps de los datos (desicion tree con un solo nodo)
def stump(X, y):
    w=first_weights(X)
    #llamamos a nuestro desicion tree
    max_depth=1
    min_gini=0.1
    stump_data=dt.train_tree(X,y, max_depth, min_gini)
    #hacemos la prediccion para obtener el error
    #prediction=dt.prediccion(X_to_predict=X)
    prediction=dt.prediccion_optimized(X_to_predict=X)
    error=total_error(y,prediction,w)
    #calculamos el amount of say
    amount=amount_of_say(error)
    #actualizamos los pesos
    w_correct=update_weights_correct(w,amount)
    w_incorrect=update_weights_incorrect(w,amount)

    w=update_weights(w_correct,w_incorrect,prediction,y)
    vector_rangos_pesos=vector_rango_pesos(w)

    #new_X,new_y=weighted_list(X,y,vector_rangos_pesos)
    new_X,new_y=weighted_list_optimizada(X,y,vector_rangos_pesos)

    
    
    return new_X,new_y,amount,stump_data

def train_boosting(X, y, num_stumps):
    # creamos un vector de stumps
    sets_X=[]
    sets_y=[]
    sets_amount=[]
    stumps_data=[]
    for i in range(num_stumps):
        new_X,new_y,amount,stump_data =stump(X,y)
        sets_X.append(new_X)
        sets_y.append(new_y)
        sets_amount.append(amount)
        stumps_data.append(stump_data)
        X=new_X
        y=new_y
    #guardamos los datos de los stumps en un csv
    write_csv_stump(stumps_data,sets_amount)
           

    return sets_X,sets_y,sets_amount

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




def prediction(X_to_predict):
    #read the csv with the adaboost trained sin indices
    df=pd.read_csv('AdaBoost_stumps_data.csv', index_col=False)
    #get the number of stumps
    num_stumps=len(df['Stump'].unique())
    prediccion_final=[]
    for i in range(len(X_to_predict)):
        row_to_predict=X_to_predict[i]
        prediccion_stumps=[]
        amount_of_say_stumps=[]
        for stump in range(num_stumps):
            df_stump=df[df['Stump']==stump]
            caracteristica=df_stump['Caracteristica'].values[0]
            valor=df_stump['Valor'].values[0]
            if row_to_predict[caracteristica] < valor:
                prediccion_stumps.append(df_stump['Prediction div 1 menor que valor'].values[0])
                amount_of_say_stumps.append(df_stump['Amount of say'].values[0])
            else:
                prediccion_stumps.append(df_stump['Prediction div 2 mayor que valor'].values[0])
                amount_of_say_stumps.append(df_stump['Amount of say'].values[0])

        #sumamos los amount of say de acuerdo a la prediccion
        amount_of_say_clase_1=0
        amount_of_say_clase_2=0
        for i in range(len(prediccion_stumps)):
            if prediccion_stumps[i]==0:
                amount_of_say_clase_1+=amount_of_say_stumps[i]
            else:
                amount_of_say_clase_2+=amount_of_say_stumps[i]
        #comparamos los amount of say y hacemos la prediccion
        if amount_of_say_clase_1 > amount_of_say_clase_2:
            prediccion_final.append(0)
        else:
            prediccion_final.append(1)


    return prediccion_final

def prediction_optimized(X_to_predict,Y_labels=None):
    # lee el archivo CSV una sola vez al iniciar el programa
    df = pd.read_csv('AdaBoost_stumps_data.csv', index_col=False)
    num_stumps = len(df['Stump'].unique())

    # crea un diccionario para almacenar los datos de cada árbol
    tree_data = {}
    for stump in range(num_stumps):
        df_stump = df[df['Stump'] == stump]
        caracteristica = df_stump['Caracteristica'].values[0]
        valor = df_stump['Valor'].values[0]
        tree_data[stump] = {
            'caracteristica': caracteristica,
            'valor': valor,
            'div1_prediction': df_stump['Prediction div 1 menor que valor'].values[0],
            'div1_amount_of_say': df_stump['Amount of say'].values[0],
            'div2_prediction': df_stump['Prediction div 2 mayor que valor'].values[0],
            'div2_amount_of_say': df_stump['Amount of say'].values[0],
    }
        
    X_to_predict = pd.DataFrame(X_to_predict)

    # convierte características categóricas a una matriz numérica si es necesario
    # X_to_predict = pd.get_dummies(X_to_predict)

    # usa apply para aplicar la lógica de predicción a todas las filas
    def predict_row(row):
        amount_of_say_clase_1 = 0
        amount_of_say_clase_2 = 0

        for stump in range(num_stumps):
            data = tree_data[stump]
            if row[data['caracteristica']] < data['valor']:
                if data['div1_prediction'] == 0:
                    amount_of_say_clase_1 += data['div1_amount_of_say']
                else:
                    amount_of_say_clase_2 += data['div1_amount_of_say']
            else:
                if data['div2_prediction'] == 0:
                    amount_of_say_clase_1 += data['div2_amount_of_say']
                else:
                    amount_of_say_clase_2 += data['div2_amount_of_say']
        
        if amount_of_say_clase_1 > amount_of_say_clase_2:
            return 0
        else:
            return 1

    predictions = X_to_predict.apply(predict_row, axis=1)
    predictions=predictions.tolist()

    #calculamos metricas
    if Y_labels is not None:
        #metrics
        #convert to jnp
        Y_labels=jnp.array(Y_labels)
        predictions=jnp.array(predictions)

        precision=precision_jax(Y_labels, predictions)
        recall=recall_jax(Y_labels, predictions)
        accuracy=accuracy_jax(Y_labels, predictions)
        return predictions, precision, recall, accuracy
    else:
        return predictions


def main_AdaBoost(X, y, X_val, y_val, num_stumps):
    experiment_name = "AdaBoost"

      # Verificar si el experimento ya existe
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        # Si no existe, crear el experimento
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        # Si ya existe, obtener el experimento
        experiment_id = experiment.experiment_id

    
    with mlflow.start_run(experiment_id=experiment_id,run_name="AdaBoost") as run:
        # log parameters
        mlflow.log_param("num_stumps", num_stumps)
        numero_features=X.shape[1]
        mlflow.log_param("numero_features", numero_features)

        # train the model
        sets_X,sets_y,sets_amount=train_boosting(X, y, num_stumps)
        predictions, precision, recall, accuracy=prediction_optimized(X,y)

        print('----------------------Metricas con datos de entrenamiento----------------------')
        print("precision: ", precision)
        print("recall: ", recall)
        print("accuracy: ", accuracy)

        # log metrics
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("accuracy", accuracy)

        #metrics con datos de validacion. La validación se hace usando el csv con los datos de los stumps entrenados
        predictions_val, precision_val, recall_val, accuracy_val=prediction_optimized(X_val,y_val)

        print('----------------------Metricas con datos de validacion----------------------')
        print("precision_val: ", precision_val)
        print("recall_val: ", recall_val)
        print("accuracy_val: ", accuracy_val)

        # log metrics
        mlflow.log_metric("precision_validation", precision_val)
        mlflow.log_metric("recall_validation", recall_val)
        mlflow.log_metric("accuracy_validation", accuracy_val)


        # guardar csv con los datos de los stumps en mlflow
        mlflow.log_artifact('AdaBoost_stumps_data.csv')

    return predictions, precision, recall, accuracy



if __name__ == "__main__":
    sets_X,sets_y,sets_amount=train_boosting(X, y, num_stumps)
    prediction=prediction(X_to_predict)
    prediction=prediction_optimized(X_to_predict)
    predictions, precision, recall, accuracy=main_AdaBoost(X, y, num_stumps)


 




