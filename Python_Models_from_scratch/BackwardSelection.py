from itertools import combinations
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#cargamos los datos para meterlos a los modelos converimos a jax numpy array
Clean_df=pd.read_csv("training_data.csv")
Clean_df = Clean_df.drop(Clean_df.columns[0], axis=1)

#convertimos la columna de clase a 0 y 1
Clean_df.loc[Clean_df['categoria'] == -1, 'categoria'] = 0
Clean_df.loc[Clean_df['categoria'] == 1, 'categoria'] = 1

class_df=Clean_df.loc[:, "categoria"]
df_without_class = Clean_df.drop(["categoria"], axis=1)

X=df_without_class.to_numpy()
y=class_df.to_numpy()

#--------------separamos los datos de train en sus dos respectivas clases--------------------
df_x=pd.DataFrame(X)
df_y=pd.DataFrame(y, columns=['y'])
df=pd.concat([df_x, df_y], axis=1)
df1_X_class_1= df[df['y'] ==0 ]
df1_X_class_1 = df1_X_class_1.drop(df1_X_class_1.columns[-1],axis=1) #quitamos ultima columna relacionada a y
df1_X_class_2= df[df['y'] ==1 ]
df1_X_class_2 = df1_X_class_2.drop(df1_X_class_2.columns[-1],axis=1)

#reconvert to numpy array
X_class_1=df1_X_class_1.to_numpy()
X_class_2=df1_X_class_2.to_numpy()






#Defining some functions to use secuencial backward
#(split the data, agroup in all posible combinationsof features, find the max norm or separation between classes 
# for heach group, choose the best group and repeat)

def FirstSplit(X_toSplit,inicial_group_features_number):
    '''This function splits the data into small groups, and is only used in the first iteration of the algorithm 
    to reduce substantially the number of features when we have a lot of features.'''
    group=np.hsplit(X_toSplit, inicial_group_features_number)
    column_names=list(range(0,X_toSplit.shape[1]))
    column_names=np.array_split(column_names, inicial_group_features_number)
    return group,column_names

def CombinatorialGroups(X_toSplit,size, num_elements_in_group, column_names):
    lista_index=list(range(0,size)) 
    dimmension=X_toSplit.shape  #saple,numero de grupos  
    index_permutations=list(combinations(lista_index, num_elements_in_group)) 
    groups=[]
    new_columns_names=[]
    for group in range(size): #size=numero de grupos
        current_list_index=index_permutations[group] # tiene numero de features -1  
        group_k=np.zeros((dimmension[0],num_elements_in_group))
        for feature in range(num_elements_in_group):
            group_k[:,feature]=X_toSplit[:,current_list_index[feature]]         
        new_group_columns_names=[]
        for index in current_list_index:
            new_group_columns_names.append(column_names[index])
        new_columns_names.append(new_group_columns_names)
        groups.append(group_k)
    return groups,new_columns_names

def CombinatorialGroups_optimized(X_toSplit, size, num_elements_in_group, column_names):
    groups = []
    new_columns_names = []
    index_permutations = list(combinations(range(X_toSplit.shape[1]), num_elements_in_group))
    for group in range(size):
        current_list_index = index_permutations[group]
        group_k = X_toSplit[:, current_list_index]
        new_group_columns_names = [column_names[i] for i in current_list_index]
        new_columns_names.append(new_group_columns_names)
        groups.append(group_k)
    return groups, new_columns_names

def mu_calculation(Maingroup):
    number_of_groups=len(Maingroup[:])
    features=len(Maingroup[0][0,:]) #primer indice es el primer array luego de hsplit y luego los otros son para ingresar al array y sus features y datos
    mu=np.zeros((number_of_groups,features))
    for group in range(number_of_groups):
        for feature in range(features):
            mu[group,feature]=np.mean(Maingroup[group][:,feature])
    return mu    

def Norm_betweet_classes(mu_class_1,mu_class_2):
    number_of_groups=mu_class_1.shape[0]    #8grupos shape(13,8) 13 grupos con 8 features
    list_of_norm=[]
    for group in range(number_of_groups):
        Norm_betweet_classes=np.linalg.norm(np.abs(mu_class_1[group,:]-mu_class_2[group,:]))
        list_of_norm.append(Norm_betweet_classes)
    return list_of_norm

def GetNamesOfFeatures(column_names,df):
    list_of_names_df= list(df)
    real_names=[]
    for i in range(len(column_names)):
        real_names.append(list_of_names_df[column_names[i]])
    return real_names




#---------Funtion to call in in the correct order the previous functions

def feature_selection(X_class_1,X_class_2,first_split,DesireFeatures):
    count=0
    while True:
        if count==0:
            groups_class_1,column_names_class_1=FirstSplit(X_class_1,first_split) #nota: la relacion entre num features/inicial_group_features_number tiene que ser int 104/13=8
            groups_class_2,column_names_class_2=FirstSplit(X_class_2,first_split)
        mu_class_1=mu_calculation(groups_class_1)
        mu_class_2=mu_calculation(groups_class_2)

        list_of_norm=Norm_betweet_classes(mu_class_1,mu_class_2)
        #print('la lista de normas es',list_of_norm)
        max_norm=max(list_of_norm)
        #print('la norma maxima es',max_norm)
        index_max_norm=list_of_norm.index(max_norm)

        winner_group_class_1=groups_class_1[index_max_norm]
        winner_group_class_2=groups_class_2[index_max_norm]
        winner_name_class_1=column_names_class_1[index_max_norm]
        winner_name_class_2=column_names_class_2[index_max_norm]
        #print('el grupo ganador es',winner_name_class_1)
        
        number_features_in_group=len(groups_class_1[index_max_norm][0,:])
        
        if number_features_in_group==DesireFeatures: 
            return winner_group_class_1,winner_group_class_2,winner_name_class_1,winner_name_class_2    #funciona como el break
        
        #groups_class_1,column_names_class_1=CombinatorialGroups(winner_group_class_1,number_features_in_group,number_features_in_group-1, winner_name_class_1)
        #groups_class_2,column_names_class_2=CombinatorialGroups(winner_group_class_2,number_features_in_group,number_features_in_group-1, winner_name_class_2)
        groups_class_1,column_names_class_1=CombinatorialGroups_optimized(winner_group_class_1,number_features_in_group,number_features_in_group-1, winner_name_class_1)
        groups_class_2,column_names_class_2=CombinatorialGroups_optimized(winner_group_class_2,number_features_in_group,number_features_in_group-1, winner_name_class_2)
        
        count=count+1

def main(X_class_1=X_class_1,X_class_2=X_class_2, desire_features=3):
    winner_group_class_1,winner_group_class_2,column_name_class1,column_name_class2=feature_selection(X_class_1,X_class_2,1,DesireFeatures=desire_features)
    NamesInDataSet=GetNamesOfFeatures(column_name_class1,df_without_class)
    print('El nombre del grupo ganador es',NamesInDataSet)

    #creamos un solo dataframe con los datos de winner_group_class_1,winner_group_class_2 y NamesInDataSet
    training_data_backwards_selection=pd.DataFrame(winner_group_class_1,columns=NamesInDataSet)
    training_data_backwards_selection=pd.concat([training_data_backwards_selection,pd.DataFrame(winner_group_class_2,columns=NamesInDataSet)],axis=0)
    #agregamos la columna de categoria
    training_data_backwards_selection['categoria']=np.concatenate((np.zeros(len(winner_group_class_1)),np.ones(len(winner_group_class_2))),axis=0)
    #guardamos los datos de entrenamiento
    training_data_backwards_selection.to_csv("training_data_backwards_selection.csv")
    print('los datos de entrenamiento se guardaron en training_data_backwards_selection.csv')

    return winner_group_class_1,winner_group_class_2,NamesInDataSet


def plot3d(winner_group_class_1,winner_group_class_2):
    x_1=winner_group_class_1[:,0]
    y_1=winner_group_class_1[:,1]
    z_1=winner_group_class_1[:,2]
    x_2=winner_group_class_2[:,0]
    y_2=winner_group_class_2[:,1]
    z_2=winner_group_class_2[:,2]

    #3d plot
    fig = plt.figure() 
    ax = fig.add_subplot(111, projection='3d', elev=45, azim=90)
    ax.scatter(x_1,y_1,z_1)
    ax.scatter(x_2,y_2,z_2)
    plt.savefig('3dplot.png')
    plt.show()


def plot2d(winner_group_class_1,winner_group_class_2):
    plt.figure(figsize=(7,6))
    x_1=winner_group_class_1[:,0]
    y_1=winner_group_class_1[:,1]
    x_2=winner_group_class_2[:,0]
    y_2=winner_group_class_2[:,1]
    plt.scatter(x_1,y_1)
    plt.scatter(x_2,y_2)
    plt.savefig('2dplot.png')
    plt.show()


if __name__ == "__main__":
    winner_group_class_1,winner_group_class_2,NamesInDataSet=main(X_class_1=X_class_1,X_class_2=X_class_2, desire_features=3)
    plot3d(winner_group_class_1,winner_group_class_2)
    plot2d(winner_group_class_1,winner_group_class_2)
