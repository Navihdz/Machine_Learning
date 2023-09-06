import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from functools import partial
from jax import random
import os
import mlflow

# Switch off the cache 
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

class Logistic_Regression():
    """
    Basic Model + Quasi Newton Methods
    """
    def __init__(self, regularization='l2', method_opt='classic_model'):
        self.regularization = regularization
        self.method_opt = method_opt
        self.error_gradient = 0.001
        self.key = random.PRNGKey(0)
        self.cpus = jax.devices("cpu")
        # You need to add some variables
        self.W = None

    @staticmethod
    @jit
    def logistic_exp(W:jnp, X:jnp)->jnp:
        """
        Generate all the w^T@x values 
        args:
            W is a k-1 x d + 1
            X is a d x N
        """
        return jnp.exp(W@X)

    @staticmethod
    @jit
    def logistic_sum(exTerms: jnp)->jnp:        
        """
        Generate all the w^T@x values 
        args:
            W is a k-1 x d 
            X is a d x N
        """
        temp = jnp.sum(exTerms, axis=0)
        n = temp.shape[0]
        return jnp.reshape(1.0+temp, newshape=(1, n))

    @staticmethod
    @jit
    def logit_matrix(Terms: jnp, sum_terms: jnp)->jnp:
        """
        Generate matrix
        """
        divisor = 1/sum_terms
        n, _ = Terms.shape
        replicate = jnp.repeat(divisor, repeats=n, axis=0 )
        logits = Terms*replicate
        return jnp.vstack([logits, divisor])
    
    @partial(jit, static_argnums=(0,))
    def model(self, W:jnp, X:jnp, Y_hot:jnp,lamda=0)->jnp:
        """
        Logistic Model, and regularized model with lamda !=0
        """
        W = jnp.reshape(W, self.sh)
        terms = self.logistic_exp(W, X)
        sum_terms = self.logistic_sum(terms)
        matrix = self.logit_matrix(terms, sum_terms)
        #print("este es el logic matrix", matrix.shape)
        return jnp.sum(jnp.sum(jnp.log(matrix)*Y_hot, axis=0), axis=0) + lamda*jnp.trace(jnp.transpose(W)@(W))#devuelve el error total de la suma de las probabilidades y*log(x|w)
 
    @staticmethod
    def one_hot(Y: jnp):
        """
        One_hot matrix
        """
        numclasses = len(jnp.unique(Y))
        return jnp.transpose(jax.nn.one_hot(Y, num_classes=numclasses))
    
    def generate_w(self, k_classes:int, dim:int)->jnp:
        """
        Use the random generator at Jax to generate a random generator to instanciate
        the augmented values
        """
        key = random.PRNGKey(0)
        keys = random.split(key, 1)
        return jnp.array(random.normal(keys[0], (k_classes, dim)))

    @staticmethod
    def augment_x(X: jnp)->jnp:
        """
        Augmenting samples of a dim x N matrix
        """
        N = X.shape[1]
        return jnp.vstack([X, jnp.ones((1, N))])
     
   
    def fit(self, X: jnp, Y:jnp,lr,tol,lamda,n_max_steps)->None:
        """
        The fit process
        """
        nclasses = len(jnp.unique(Y))
        X = self.augment_x(X)
        dim = X.shape[0]
        W = self.generate_w(nclasses-1, dim)
        Y_hot = self.one_hot(Y)
        #print(Y_hot)
        self.W = getattr(self, self.method_opt, lambda W, X, Y_hot, lr,tol,lamda, n_max_steps: self.error() )(W, X, Y_hot,lr,tol,lamda, n_max_steps)
        return self.W
    
    @staticmethod
    def error()->None:
        """
        Only Print Error
        """
        raise Exception("Opt Method does not exist")
    
    def classic_model(self, W:jnp, X:jnp, Y_hot:jnp, lr:float=1e-9,  tol:float=1e-3, lamda=0, n_max_steps=200)->jnp:
        """
        The naive version of the logistic regression
        """
        n, m = W.shape 
        self.sh = (n, m)
        Grad = jax.grad(self.model, argnums=0)(jnp.ravel(W), X, Y_hot,lamda)
        
        loss = self.model(jnp.ravel(W), X, Y_hot,lamda)
        cnt = 0
        while True:
            Hessian = jax.hessian(self.model, argnums=0)(jnp.ravel(W), X, Y_hot,lamda)
            W = W - lr*jnp.reshape((jnp.linalg.inv(Hessian)@Grad), self.sh)
            Grad =  jax.grad(self.model, argnums=0)(jnp.ravel(W), X, Y_hot,lamda)
            old_loss = loss
            loss = self.model(jnp.ravel(W), X, Y_hot,lamda)
            if cnt%200 == 0:
                print('step:',cnt,'loss:',f'{self.model(jnp.ravel(W), X, Y_hot,lamda)}')
            #if  jnp.abs(old_loss - loss) < tol:
                #break
            cnt +=1
            if cnt > n_max_steps:
                break
        return W
    
    def estimate(self, X:jnp)->jnp:
        """
        Estimation
        """
        X = self.augment_x(X)
        terms = self.logistic_exp(self.W, X)
        sum_terms = self.logistic_sum(terms)
        matrix = self.logit_matrix(terms, sum_terms)
        return jnp.argmax(matrix, axis=0)
    
    def precision(self, y, y_hat):
        """
        Precision
        args:
            y: Real Labels
            y_hat: estimated labels
        return TP/(TP+FP)
        """
        TP = sum(y_hat == y)
        FP = sum(y_hat != y)
        return (TP/(TP+FP)).tolist()
    
    def precision_jax(self, y, y_hat):
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
    
    def recall_jax(self,y, y_hat):
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
    
    def accuracy_jax(self,y, y_hat):
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
            accuracy_cpu = jit(lambda x: x, device=self.cpus[0])((TP+TN)/(TP+FP+TN+FN))
            return float(accuracy_cpu)
    
    def normalize_columns(self,arr):
        """
        Normalizes an array of shape (n, m) by column using the min-max scaling method.
        """
        mins = jnp.min(arr, axis=0)
        maxs = jnp.max(arr, axis=0)
        return (arr - mins) / (maxs - mins)
    


def LogisticRegression(X, Y, X_val,Y_val,lr,tol,lamda,n_max_steps)->jnp:
    """
    Logistic Regression
    args:
        X: dataset de dimension (samples, features)
        Y: labels (entre 0 y 1) de dimension (samples,)
    return:
        Y_hat: estimated labels de dimension (samples,)
        precision: precision of the model
    """
    #----------------------------mlflow--------------------------------
    experiment_name = "LogisticRegression"
    # Verificar si el experimento ya existe
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        # Si no existe, crear el experimento
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        # Si ya existe, obtener el experimento
        experiment_id = experiment.experiment_id
    

    with mlflow.start_run(experiment_id=experiment_id,run_name="LogisticRegression") as run:
        numero_features=X.shape[1]
        #log params
        mlflow.log_param("lr", lr)
        mlflow.log_param("tol", tol)
        mlflow.log_param("lamda", lamda)
        mlflow.log_param("n_max_steps", n_max_steps)
        mlflow.log_param("NumberOfFeatures", numero_features)
    #------------------------------------------------------------------     
        #garantizamos que sean jax arrays
        X = jnp.array(X)
        Y = jnp.array(Y)
        model = Logistic_Regression()
        X= model.normalize_columns(X)
        #trasnponemos X ya que siguientes funciones asumen que es de dimension (features, samples)
        X = jnp.transpose(X)
        w_logistic=model.fit(X, Y, lr,tol,lamda,n_max_steps)

        Y_hat = model.estimate(X)
        precision=model.precision_jax(Y, Y_hat)
        recall=model.recall_jax(Y, Y_hat)
        accuracy=model.accuracy_jax(Y, Y_hat)

        print('----------------------Metricas con datos de entrenamiento----------------------')
        print(f'precision: {precision}')
        print(f'recall: {recall}')
        print(f'accuracy: {accuracy}')


        #log metrics
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("accuracy", accuracy)

        #-----------------------------validacion--------------------------------

        Y_hat_val, precision_val, recall_val, accuracy_val=prediction_trained_logistic(X_val,w_logistic, Y_val)


        print('----------------------Metricas con datos de validacion----------------------')
        print(f'precision: {precision_val}')
        print(f'recall: {recall_val}')
        print(f'accuracy: {accuracy_val}')
        
        #log metrics
        mlflow.log_metric("precision_validation", precision_val)
        mlflow.log_metric("recall_validation", recall_val)
        mlflow.log_metric("accuracy_validation", accuracy_val)

        


        return w_logistic,Y_hat, precision, recall, accuracy
    
def prediction_trained_logistic(X,w_logistic, Y=None):
    """
    Prediction with trained logistic regression
    args:
        X_val: dataset de dimension (samples, features)
        w_logistic: trained weights
    return:
        Y_hat: estimated labels de dimension (samples,)
        precision: precision of the model
    """
    #garantizamos que sean jax arrays
    X = jnp.array(X)
    Y = jnp.array(Y)
    model_prediction = Logistic_Regression()
    X= model_prediction.normalize_columns(X)
    #trasnponemos X ya que siguientes funciones asumen que es de dimension (features, samples)
    X = jnp.transpose(X)
    model_prediction.W = w_logistic
    Y_hat = model_prediction.estimate(X)


    if Y is not None:
        precision=model_prediction.precision_jax(Y, Y_hat)
        recall=model_prediction.recall_jax(Y, Y_hat)
        accuracy=model_prediction.accuracy_jax(Y, Y_hat)
        return Y_hat, precision, recall, accuracy
    
    else:
        return Y_hat

if __name__ == "__main__":
    w_logistic,Y_hat, precision, recall, accuracy = LogisticRegression(X, Y,X_val,y_val, lr,tol,lamda,n_max_steps)
    prediction_trained_logistic(X,w_logistic)
