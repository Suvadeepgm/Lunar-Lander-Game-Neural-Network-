import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import random, dot
import pickle
from FinalNeural import LunarLanderFinalNew

class NeuralNetHolder:

    def __init__(self):
        super().__init__()
        
        with open("./p1_obj.pickle", "rb") as f:
            self.p1_obj=pickle.load(f)
       
        '''self.weight_xh= np.array([[-0.61010274, -1.48763591],
         [ 0.19215344,  3.61288443],
         [-0.46050105,  1.72780642],
         [-4.79754254,  0.31784441],
         [ 3.52107063,  0.57080336],
         [10.24490226,  0.78743965],
         [ 0.59374086, -8.3987323 ],
         [-0.57776846, -2.25893814],
         [ 4.20473552,  1.73805939],
         [ 1.13200342,  2.89823757],
         [-0.85044884, -3.52741466],
         [ 5.67864389, -0.76787297],
         [ 1.70639221, -1.17774654],
         [-0.16329573, -1.2384815 ],
         [ 1.35335974, -1.46465976],
         [ 0.27631965, -2.03616681],
         [ 0.1335792 , -6.114786  ],
         [-0.45247534, -5.03516975]])
        self.weight_hy= np.array([[-0.0661191,  -1.34221526,  1.04716919, -0.19809918, -1.47585272, -0.83578004,
           3.80251568,  0.31013956,  1.97567466, -0.14431481,  0.89771951,  2.18489796,
          -1.59092352,  0.37776123, -1.5828866,   0.6759124,   2.66577622,  2.07035289],
         [ 0.44994728,  0.64558299, -0.62713184, -0.12211058, -2.45829035,  3.49137068,
           0.5556827,  -0.5540161,  -0.6785849,   0.46712775, -0.83696936, -1.44580875,
          -0.72232376, -0.22708979, -0.20441635,  0.35427232, -0.14984041, -0.8946756 ]])
        self.bias_xh=np.array([[-0.50692555,  2.31110635, -0.40170075,  1.43567846, -2.71624131, -5.51090572,
           0.04994598, -2.13695245, -1.1099473,   1.41050319, -1.79225401, -1.93223113,
          -0.48109126, -2.27159358, -0.14077443, -2.60652078, -1.14469714, -1.45085461]])
        self.bias_hy=np.array([[-0.00057718,  0.33914417]])'''
        self.MAX = [799.2745374,908, 8, 6.617275317]
        self.MIN = [-810.7223174,65.00015069,-6.9,-7.486346828]
        
   
        
    def sigmoidFunc(self,x):
        return 1.0 / (1.0 + np.exp(-x))
    
      
    def normalized(self, x,minimum, maximum):
        normalizedData =((x - minimum)/(maximum - minimum)) 
        return normalizedData
    
    def denormalized(self, x_norm, maximum, minimum):
        denormalized_val = (x_norm * (maximum - minimum)) + minimum 
        return denormalized_val
    
    
    
    
    def forwardPropagation(self,x):  #Forward Propagation
       
        
        self.n1=np.dot(x,self.weight_xh.T) + self.bias_xh
        #print(np.shape(self.z1))
        self.h1=self.sigmoidFunc(self.n1)
        #print(np.shape(self.h1))
        self.n2=np.dot(self.h1,self.weight_hy.T) + self.bias_hy
        #print(np.shape(self.z2))
        self.h2=self.sigmoidFunc(self.n2)
        #print(np.shape(self.h2))
        return self.h2
    
    def predict(self, input_row):
        lst = input_row.split(',')
        input_row = [float(i) for i in lst]

        # WRITE CODE TO PROCESS INPUT ROW AND PREDICT X_Velocity and Y_Velocity
        print('actual input',input_row)
        X1 = float(self.normalized(input_row[0],self.MIN[0],self.MAX[0])) 
        X2 = float(self.normalized(input_row[1],self.MIN[1],self.MAX[1]))
        
        #X1 = int(input_row[0])
        #X2 = int(input_row[1])
        input_row = np.array([[X1,X2]])
        self.input_row=input_row
        print('input',input_row)
        prediction =  self.p1_obj.forwardPropagation(input_row)
        
        #Y1 = self.denormalized(prediction[0,0],self.MAX[2],self.MIN[2])
        Y2 = self.denormalized(prediction[0,1],self.MAX[3],self.MIN[3])
        print('output',prediction)
        Y1 = prediction[0,0]
        #Y2 = prediction[0,1]   
        print('Y',Y2,Y1)
        return Y2,Y1
        #pass
