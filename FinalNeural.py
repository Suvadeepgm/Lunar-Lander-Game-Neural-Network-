# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 13:46:49 2022

@author: suvadeep
"""

import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from numpy import random, dot
import pickle

class LunarLanderFinalNew:
    
    
    def __init__(self):
        random.seed(48)
        self.inputN=2 #Number of Input Neurons
        self.outputN=2 #Number of Output Neurons
        self.hiddenN=18 #Number of Hidden Neurons
        self.weight_xh= np.random.randn(self.hiddenN,self.inputN) *np.sqrt(2/self.inputN)  #input to hidden layer weight initialization
        self.weight_hy= np.random.randn(self.outputN,self.hiddenN) *np.sqrt(2/self.hiddenN)  #hidden to output layer weight initialization
        self.bias_xh=np.zeros((1,self.hiddenN)) #bias to hidden layer initialization
        self.bias_hy=np.zeros((1,self.outputN)) #bias to hidden layer initialization
        self.alpha = 0.2 #Momentum
        self.Lambda = 0.6 
        self.learningRate=0.3 #learning rate initialization 
        self.epoch=200 # number of epoch
        self.dWeight_xh = 0 
        self.dWeight_hy = 0
        self.dBias_xh = 0
        self.rmseList=[]
        self.RMSE_validation=[]
        self.eiList=[]
        self.error = 0
        
        
    def read(self):
        self.df = pd.read_csv(r"D:\EssexFiles\NeuralNetwork\AssignmentCode\ce889_dataCollection.csv") #using pandas to read the data
        self.df.drop_duplicates() #checking for duplicate data
        self.df.dropna()  # checking for missing values
        #self.df = self.df.sample(frac = 1) #taking a random sample of data
        LunarData_Normalized = (self.df - self.df.min())/(self.df.max() - self.df.min()) #Normalizing entire dataset
        LunarData_Normalized.columns = ['x1','x2','y1','y2'] #adding headers to the dataset 
        self.x = LunarData_Normalized[['x1','x2']].values
        self.y = LunarData_Normalized[['y1','y2']].values
        
    def sigmoidFunc(self,x):  #Sigmoid function
        return 1.0 / (1.0 + np.exp(-x))
    
    def derivative_sigmoid(self,x): #Sigmoid Derivative Function
        return x * (1 - x)
    
    def writeToCSV(self,row): #function to write weights into csv
        with open('D:\\NeuralNetwork\\AssignmentCode\\Weights.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(row)
       
    
    def denormalized(self, x_norm):
        denormalized_val = x_norm * (self.df.max() - self.df.min()) + self.df.min()
        return denormalized_val
    
    ########################''''''''''Forward Propagation'''''''''''#################
    
    def forwardPropagation(self,x):   
        
        self.n1=np.dot(x,self.weight_xh.T) + self.bias_xh 
        #print(np.shape(self.z1))
        self.h1=self.sigmoidFunc(self.n1)
        #print(np.shape(self.h1))
        self.n2=np.dot(self.h1,self.weight_hy.T) + self.bias_hy
        #print(np.shape(self.z2))
        self.h2=self.sigmoidFunc(self.n2)
        return self.h2
        
     #########################''''''''''''Backward Propagation''''''''''##################
    def backPropagation(self,xTrain,yTrain,):
        
            self.error=self.h2 - yTrain     #error calculation = predicted - actual      
            
     #######################'''''''''''''''Delta at Output'''''''''''''''''''''############
                        
            delta_at_output=self.error*(self.derivative_sigmoid(self.h2)*self.Lambda)
            self.dWeight_hy = self.learningRate * np.dot(delta_at_output.T,self.h1) + self.alpha * self.dWeight_hy
            
            
      ######################'''''''''''''''Deltas at Hidden Layer ''''''''''''''''#####################
            
            errorHiddenLayer = np.dot( delta_at_output, self.weight_hy)
            delta_at_hidden = errorHiddenLayer*(self.derivative_sigmoid(self.h1)*self.Lambda)
             
            self.dWeight_xh=self.learningRate * dot(delta_at_hidden.T,xTrain)+self.alpha*self.dWeight_xh
           
            
       ###################### Weight and Bias update at Output ###########################
            
            self.weight_hy -= self.learningRate*self.dWeight_hy 
           
            self.bias_hy -=np.sum(delta_at_output,axis = 0) *self.learningRate
            
            ##################### Weight and Bias update at Hidden Layer##########################
           
            self.weight_xh -= self.dWeight_xh * self.alpha
            self.bias_xh -= np.sum(delta_at_hidden,axis=0) *self.learningRate
            
              #####'''''''''''''' Training - Validation - Test ''''''''############
    def train(self):    
            x_train, x_validate_test , y_train, y_validate_test = train_test_split(self.x, self.y, test_size=0.6, random_state=1)
            x_validate,x_test,y_validate,y_test=train_test_split(self.x, self.y, test_size=0.6, random_state=1)
            self.nTrain=len(x_train)
            self.nValidate=len(x_validate)
            self.nTest=len(x_test)
            for i in range(self.epoch):  #Epoch loop
                rmse_list = []
                rmse_listValidation=[]
                #np.random.shuffle(x_train)
                for j in range(self.nTrain):                 #Training Loop
                    self.forwardPropagation(x_train[j]) 
                    self.backPropagation(x_train[j].reshape(1,2),y_train[j])
                    t_error=np.sum(np.abs(self.error))/2
                    rmse_list.append(t_error)
                    
                for j in range(self.nValidate):               #Validation Loop
                    self.forwardPropagation(x_validate[j].reshape(1,2))
                    errorValidation  = self.h2 - y_validate[j]
                    tErrorValidation= np.sum(np.abs(errorValidation))/2
                    rmse_listValidation.append(tErrorValidation)
                    
                #######'''''''''RMSE Calculation - Validation''''''########
                    
                errorSumValidate = sum([k**2 for k in rmse_listValidation])
                RmseEpochValidate = np.sqrt(errorSumValidate/self.nValidate)
                print('RMSE Value Validation =================>>>>',RmseEpochValidate )
                self.RMSE_validation.append(RmseEpochValidate)
                        
                #######'''''''''RMSE Calculation - Train''''''########
                
                errorSqSummation=sum([k**2 for k in rmse_list])
                rmsePerEpoch=np.sqrt((errorSqSummation)/self.nTrain)                 
                self.rmseList.append(rmsePerEpoch)
                print ('Loss in epoch '+str(i)+': ',rmsePerEpoch) 
                
                
                
                ########'''''''''''''''''Early Stopping Criteria'''''''''''########
                '''earlyStopVal = abs(RmseEpochValidate - rmsePerEpoch)/RmseEpochValidate
                if earlyStopVal<0.00000001: 
                    
                    break;'''
                if i>=2 and self.RMSE_validation[i]>self.RMSE_validation[i-1]:
                        print("Early Stopping Criteria Met")
                        break;
                    
            
               
            plt.figure(1)  
            plt.plot(self.rmseList)                    #RMSE Train
            plt.plot(self.RMSE_validation)             #RMSE Validate
            
            for i in range(self.nTest):                #Test Loop
                self.forwardPropagation(x_test[i])
                
            errorTest = self.h2 - y_test[j]            #Calculating RMSE Test
            rmseTest = np.sqrt((np.sum(np.abs(errorTest))**2))
            print('rmse test--->',rmseTest)    
            
            print("input to hidden weight: ",self.weight_xh)
            print("hidden to output weight: ",self.weight_hy)
            print("input to hidden bias: ",self.bias_xh)
            print("hidden to output bias: ",self.bias_hy)
            self.writeToCSV([self.weight_xh, self.weight_hy, self.bias_xh, self.bias_hy])    #Storing the final weights and biases
     
if __name__=='__main__':       
    p1  = LunarLanderFinalNew() #Creating object of the class
    p1.read() #Invoking the function to read the game data
    p1.train() #Training/Validation/Test 
    
    with open("p1_obj.pickle", "wb") as f:    #Using pickle to store the final weights and biases
        pickle.dump(p1,f)

        
        