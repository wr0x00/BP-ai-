import numpy as np
def sigmoid(x):
  #激活函数: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    #激活函数的导函数
  return  sigmoid(x)* (1 - sigmoid(x))

def mse_loss(y_true, y_pred):
  # y_true正确值，y_pred预测值
  return ((y_true - y_pred) ** 2).mean()
class ourworld:
  def __init__(self):
    # 权重，Weights
    self.w1 = np.random.normal()
    self.w2 = np.random.normal()
    self.w3 = np.random.normal()
    self.w4 = np.random.normal()
    self.w5 = np.random.normal()
    self.w6 = np.random.normal()
    self.w7 = np.random.normal()
    self.w8 = np.random.normal()
    self.w9 = np.random.normal()
    self.w10 = np.random.normal()
    self.w11 = np.random.normal()
    self.w12 = np.random.normal()
    # 截距项，Biases
    self.b1 = np.random.normal()
    self.b2 = np.random.normal()
    self.b3 = np.random.normal()
    self.b4 = np.random.normal()
  def feedforward(self, x):
    # x is a numpy array with 2 elements.
    h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.w3 * x[2]+self.b1)
    h2 = sigmoid(self.w4 * x[0] + self.w5 * x[1] + self.w6 * x[2]+self.b2)
    h3 = sigmoid(self.w7 * x[0] + self.w8 * x[1] + self.w9 * x[2]+self.b3)
    o1 = sigmoid(self.w10 * h1 + self.w11 * h2 + self.w12 * h3+self.b3)
    return o1
  def train(self, data, all_y_trues):
    '''
    - data is a (n x 2) numpy array, n = # of samples in the dataset.
    - all_y_trues is a numpy array with n elements.
      Elements in all_y_trues correspond to those in data.
    '''
    learn_rate = 0.1
    i=0
    while True:
      loss2=mse_loss(all_y_trues,np.apply_along_axis(self.feedforward, 1, data))
      i+=1
      for x, y_true in zip(data, all_y_trues):
       h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.w3 * x[2]+self.b1)
       h2 = sigmoid(self.w4 * x[0] + self.w5 * x[1] + self.w6 * x[2]+self.b2)
       h3 = sigmoid(self.w7 * x[0] + self.w8 * x[1] + self.w9 * x[2]+self.b3)
       y_pred = sigmoid(self.w10 * h1 + self.w11 * h2 + self.w12 * h3+self.b4)
       d_L_d_ypred = -2 * (y_true - y_pred)
       # Neuron o1
       d_ypred_d_w10= h1 * deriv_sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.w3 * x[2]+self.b1)
       d_ypred_d_w11= h2 * deriv_sigmoid(self.w4 * x[0] + self.w5 * x[1] + self.w6 * x[2]+self.b2)
       d_ypred_d_w12= h3 * deriv_sigmoid(self.w7 * x[0] + self.w8 * x[1] + self.w9 * x[2]+self.b3)
       d_ypred_d_b4 = deriv_sigmoid(self.w10 * h1 + self.w11 * h2 + self.w12 * h3+self.b3)

       d_ypred_d_h1 = self.w10 * deriv_sigmoid(self.w10 * h1 + self.w11 * h2 + self.w12 * h3+self.b3)
       d_ypred_d_h2 = self.w11 * deriv_sigmoid(self.w10 * h1 + self.w11 * h2 + self.w12 * h3+self.b3)
       d_ypred_d_h3 = self.w12 * deriv_sigmoid(self.w10 * h1 + self.w11 * h2 + self.w12 * h3+self.b3)

       # Neuron h1
       d_h1_d_w1 = x[0] * deriv_sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.w3 * x[2]+self.b1)
       d_h1_d_w2 = x[1] * deriv_sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.w3 * x[2]+self.b1)
       d_h1_d_w3 = x[2] * deriv_sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.w3 * x[2]+self.b1)
       d_h1_d_b1 = deriv_sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.w3 * x[2]+self.b1)
       # Neuron h2
       d_h2_d_w4 = x[0] * deriv_sigmoid(self.w4 * x[0] + self.w5 * x[1] + self.w6 * x[2]+self.b2)
       d_h2_d_w5 = x[1] * deriv_sigmoid(self.w4 * x[0] + self.w5 * x[1] + self.w6 * x[2]+self.b2)
       d_h2_d_w6 = x[2] * deriv_sigmoid(self.w4 * x[0] + self.w5 * x[1] + self.w6 * x[2]+self.b2)
       d_h2_d_b2 = deriv_sigmoid(self.w4 * x[0] + self.w5 * x[1] + self.w6 * x[2]+self.b2)
       #Neuron h3
       d_h3_d_w7 = x[0] * deriv_sigmoid(self.w7 * x[0] + self.w8 * x[1] + self.w9 * x[2]+self.b3)
       d_h3_d_w8 = x[1] * deriv_sigmoid(self.w7 * x[0] + self.w8 * x[1] + self.w9 * x[2]+self.b3)
       d_h3_d_w9 = x[2] * deriv_sigmoid(self.w7 * x[0] + self.w8 * x[1] + self.w9 * x[2]+self.b3)
       d_h3_d_b3 = deriv_sigmoid(self.w7 * x[0] + self.w8 * x[1] + self.w9 * x[2]+self.b3)

       # --- Update weights and biases
       # Neuron h1
       self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
       self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
       self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w3
       self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1
       # Neuron h2
       self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
       self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w5
       self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w6
       self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2
       # Neuron h3
       self.w7 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h3_d_w7
       self.w8 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h3_d_w8
       self.w9 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h3_d_w9
       self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h3_d_b3
       # Neuron o1
       self.w10 -= learn_rate * d_L_d_ypred * d_ypred_d_w10
       self.w11 -= learn_rate * d_L_d_ypred * d_ypred_d_w11
       self.w12 -= learn_rate * d_L_d_ypred * d_ypred_d_w12
       self.b4 -= learn_rate * d_L_d_ypred * d_ypred_d_b4
       loss1=mse_loss(all_y_trues,np.apply_along_axis(self.feedforward, 1, data))
       if i % 10 == 0 :
         print("i loss: ",i,loss1)
      if loss2<loss1:
        break 
data = np.array([
  #M,P,C
  [2,1,1],  # Alice
  [2,0,1],   # Bob
  [8,9,1],   # Charlie
  [7,1,1], # Diana
  [1,1,1],  # Alice
  [0,1,1],   # Bob
  [8,9,1],   # Charlie
  [4,5,1], # Diana
])
all_y_trues = np.array([
  0.04, 
  0.03, 
  0.18,
  0.09, 
  0.03,
  0.02,
  0.18,
  0.1,
])
# Train our neural network!
network = ourworld()
network.train(data, all_y_trues)
print(network.w1,network.w2,network.w3,network.w4,network.w5,network.w6,network.w7,network.w8,network.w9,network.w10,network.w11,network.w12) #打印各权重 0.951 - F
print(network.b1,network.b2,network.b3)#打印各截距项
while True:
    a=int(input("a:"))
    b=int(input("b:"))
    c=int(input("c:"))
    print("is: ", network.feedforward(np.array([a,b,c]))*10) # 0.951 - F
''''''