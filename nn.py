import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
digit_data = pd.read_csv("digitdataset.csv")
data = np.array(digit_data)

m,n  = data.shape
#print(m,n)

np.random.shuffle(data)

test_data = data[0:1000].T
y_test = test_data[0]
X_test = test_data[1:n]
X_test = X_test/255

train_data = data[1000:m].T
y_train = train_data[0]
X_train = train_data[1:n]
X_train = X_train/255

n_train,m_train = X_train.shape
#print(y_train)

def start_param():
    w1 = np.random.rand(10,784) - 0.5
    b1 = np.random.rand(10,1) - 0.5

    w2 = np.random.rand(10,10) - 0.5
    b2 = np.random.rand(10,1) - 0.5

    return w1,b1,w2,b2

def relu(Z):
    return np.maximum(Z,0)

def softmax(Z):
    A = np.exp(Z)/sum(np.exp(Z))
    return A

def forward_prop(w1,b1,w2,b2,X):
    z1 = w1.dot(X) +b1
    a1 = relu(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)

    return z1,a1,z2,a2

def relu_der(Z):
    return Z>0

def ohe(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y



def update_params(w1,b1,w2,b2,dw1,db1,dw2,db2,alpha):
    w1 = w1 - alpha*dw1
    b1 = b1 - alpha*db1
    w2 = w2 - alpha*dw2
    b2 = b2 - alpha*db2

    return w1,b1,w2,b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def back_prop(z1, a1, z2, a2, w1, w2, X, Y, m):
    one_hot_y = ohe(Y)
    dz2 = a2 - one_hot_y
    dw2 = (1 / m) * dz2.dot(a1.T)
    db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)
    dz1 = w2.T.dot(dz2) * relu_der(z1)
    dw1 = (1 / m) * dz1.dot(X.T)
    db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

    return dw1, db1, dw2, db2

def gradient_desc(X, Y, alpha, iterations):
    w1, b1, w2, b2 = start_param()
    m = X.shape[1]  # Number of examples
    for i in range(iterations):
       
        z1, a1, z2, a2 = forward_prop(w1, b1, w2, b2, X)
        dw1, db1, dw2, db2 = back_prop(z1, a1, z2, a2, w1, w2, X, Y, m)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)
        if i % 10 == 0:
            print(f"Iteration {i}")
            predictions = get_predictions(a2)
            accuracy = get_accuracy(predictions, Y)
            print(f"Accuracy: {accuracy:.4f}")
    return w1, b1, w2, b2
w1, b1, w2, b2 = gradient_desc(X_train, y_train, 0.10, 500)

def make_prediction(X,w1,b1,w2,b2):
  _,_,_,a2 = forward_prop(w1,b1,w2,b2,X)
  prediction = get_predictions(a2)
  return prediction 
def test_prediction(index,w1,b1,w2,b2):
  current_image = X_train[:,index,None]
  prediction = make_prediction(X_train[:,index,None],w1,b1,w2,b2)
  label = y_train[index]
  print("Prediction: ", prediction)
  print("Label: ", label)

  current_image = current_image.reshape((28, 28)) * 255
  plt.gray()
  plt.imshow(current_image, interpolation='nearest')
  plt.show()

test_prediction(0, w1, b1, w2, b2)
test_prediction(1, w1, b1, w2, b2)
test_prediction(2, w1, b1, w2, b2)
test_prediction(3, w1, b1, w2, b2)
test_prediction(4,w1,b1,w2,b2)

test_prediction = make_prediction(X_test,w1,b1,w2,b2)
get_accuracy(test_prediction,y_test)