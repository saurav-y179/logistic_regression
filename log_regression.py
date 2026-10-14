#  logistic  regression  
import numpy as  np 
import matplotlib.pyplot as plt 

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris = load_iris()

X = iris.data
y = iris.target  

mask = (y <= 1 )
X = X[mask]
y = y[mask]

X  =  X[:,[2,3]]  #   order is here  though   for choosing the columns  


X_train , X_test , y_train ,y_test = train_test_split(X,y,test_size = 0.2, random_state = 42 , stratify = y )

#Splitting is random. Every time you run the script, you'd get a different train/test split, and your metrics would jump around slightly. Setting random_state=42 freezes the shuffle so the split is identical every run. This makes your results reproducible — you and I both get the same numbers.


# stratify  is to reserve the class ratio  in both the  splits  in both test and train 


mean = X_train.mean(axis = 0 )
std = X_train.std(axis = 0 )

# here mean becomes    a  single value for each collumn as we collapsed the  rows 
# same  thing for  both std and mean 

X_train =  (X_train-mean)/std
X_test = (X_test-mean)/std 

#standardization above


def sigmoid(z):
    p =  1 /(1+np.exp(-z))
    return p

def bce_loss(y_true,p):
    eps = 1e-15
    p = np.clip(p,eps,1-eps)
    return -np.mean((y_true*np.log(p))+ ((1-y_true)*np.log(1-p)))


def train_logistic_regression(X,y,lr = 0.1 ,epochs = 1000):
    n,d = X.shape 
    w = np.zeros(d)
    b = 0.0 
    losses = []
    
    for _  in range(epochs):
        z  = X@w +  b  
        p =sigmoid(z)
        
        loss = bce_loss(y,p)
        losses.append(loss)
        
        dw   =  (X.T @(p-y))/  n 
        db =  np.sum(p-y) / n  
        
        w  -= lr*dw
        b -= lr* db
    return  w ,b , losses 
w ,b  , losses  =  train_logistic_regression(X_train,y_train,lr = 0.1 , epochs = 10000)
print("weights:", w)
print("bias:",b )


def predict(X,w,b , thresold = 0.5 ):
    p = sigmoid(X@w+ b )
    return (p >= thresold).astype(int), p 
y_pred , y_prob = predict(X_test,w,b)


def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp, tn, fp, fn


tp, tn , fp , fn =  confusion_matrix(y_test , y_pred )


accuracy  =  (tp +tn ) /  (tp+tn +fn +fp )

recall  =  tp/(tp+ fn ) if (tp+fn) > 0 else 0 
precision = tp/(tp+fp)    if  (tp+fp) > 0 else  0  
f1  =  2 *  precision * recall /(precision + recall) if (precision + recall )  > 0 else 0 

print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1:        {f1:.4f}")
    
    
def  plot_decision_boundary(X_train,y_train,X_test,y_test,w,b):
    plt.figure(figsize=(8,6))
    
    plt.scatter(X_train[y_train == 0][:,0],X_train[y_train == 0][:,1],
                color = "blue",marker = "o",alpha = 0.6 , label = "Train Setosa")
    plt.scatter(X_train[y_train == 1][:,0],X_train[y_train == 1][:,1],
                color = "red",marker = "o",alpha = 0.6 , label = "Train Versicolor")
    
    
    plt.scatter(X_test[y_test == 0][:,0],X_test[y_test == 0][:,1],
                    color = "blue",marker = "x",s = 100  , label = "Test  Setosa")
    plt.scatter(X_test[y_test == 1][:,0],X_test[y_test == 1][:,1],
                    color = "red",marker = "x",s =  100  , label = "Test  Versicolor")
    
    x1_vals  = np.linspace(X_train[:,0].min()-0.5,
                           X_train[:,0].max()+0.5, 100)
    x2_vals = -(w[0]* x1_vals + b ) / w[1]
    
    plt.plot(x1_vals, x2_vals, color = "green", linewidth = 2 ,label = "Decision Boundary")
    
    plt.xlabel("Petal length(Standardized)")
    plt.ylabel("Petal width(Standardized)")
    plt.title("LOgistic regression - Setosa vs Versicolor ")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_decision_boundary(X_train , y_train , X_test , y_test , w, b )

plt.figure(figsize=(8,6))
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Binary cross entropy Loss")
plt.title("Loss Curve ")
plt.grid(True)
plt.show()
        
    
                