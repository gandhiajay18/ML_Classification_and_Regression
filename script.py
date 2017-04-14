
# coding: utf-8

# In[1]:

import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys


# In[2]:


def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 

    #print(X)
    #print(y)
    #To get variables of input
    Data = X
    
    Data_rows = len(Data)
    Data_cols = len(Data[0])
    Y_rows = len(y)
    Y_cols = len(y[0])
    #Convert y into usable format
    Y_Labels = y.reshape(Y_rows)
    #Check Label Values
    #print(Y_Labels)      #We can see that Labels are in range from 1-5
    max_Label = max(Y_Labels) #Float value
    #print(max_Label)   
    max_Label = int(max_Label) #Converted to int
    #print(max_Label)   
    #true_Labels = np.arange(1, max_Label+1, 1)
    true_Labels = np.unique(Y_Labels)  #Create a unique list of Labels
    #print(true_Labels)
    #Create a means matrix of size d*k ; Here d = X_cols and k = max_Label
    means = np.zeros((Data_cols, max_Label))  #Empty means matrix created
    #print(means)
    
    for i in range(max_Label):
        means[:,i] = np.mean(Data[Y_Labels==true_Labels[i]],0) 
    
    
    #covmat = []
    covmat = np.cov(Data,rowvar=0)
    #print(means)
    #print(covmat)
    # IMPLEMENT THIS METHOD 
    return means,covmat


# In[3]:


def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    Data = X
    
    Data_rows = len(Data)
    Data_cols = len(Data[0])
    Y_rows = len(y)
    Y_cols = len(y[0])
    #Convert y into usable format
    Y_Labels = y.reshape(Y_rows)
    #Check Label Values
    #print(Y_Labels)      #We can see that Labels are in range from 1-5
    max_Label = max(Y_Labels) #Float value
    #print(max_Label)   
    max_Label = int(max_Label) #Converted to int
    #print(max_Label)   
    #true_Labels = np.arange(1, max_Label+1, 1)
    true_Labels = np.unique(Y_Labels)  #Create a unique list of Labels
    #print(true_Labels)
    #Create a means matrix of size d*k ; Here d = X_cols and k = max_Label
    
    means = np.zeros((Data_cols, max_Label))  #Empty means matrix created
    #print(means)    

    covmats = [np.zeros((Data_cols,Data_cols))] * max_Label
    for i in range(max_Label):
        means[:,i] = np.mean(Data[Y_Labels==true_Labels[i]],0) 
        covmats[i] = np.cov(Data[Y_Labels==true_Labels[i]].T)

        
    # IMPLEMENT THIS METHOD
    return means,covmats


# In[4]:

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    
    test_Data = Xtest
    test_Data_rows = len(test_Data)
    test_Data_cols = len(test_Data[0])
    Y_test_rows = len(ytest)
    Y_test_cols = len(ytest[0])
    Y_test_Labels = ytest.reshape(Y_test_rows)
    means_rows = len(means)
    means_cols = len(means[0])
    covmat_inv = np.linalg.inv(covmat)
    covmat_det = np.linalg.det(covmat)
    pdf= np.zeros((test_Data_rows,means_cols));
    covmat_det_sq = np.square(covmat_det)
    twopi = 2*np.pi
    denom = np.sqrt(twopi)*(covmat_det_sq)
    for i in range(means_cols):
        var = (test_Data - means[:,i])
        var2 = np.sum(var*np.transpose(np.dot(covmat_inv, np.transpose(var))),1)
        num = np.exp(-0.5*var2)
        pdf[:,i] = num/denom
        #pdf[:,i]=var2
    max_indice = np.argmax(pdf,1)
    ypred = max_indice + 1
    acc = np.mean(ypred == Y_test_Labels)*100
    
    return acc,ypred


# In[5]:

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    
    test_Data = Xtest
    test_Data_rows = len(test_Data)
    test_Data_cols = len(test_Data[0])
    Y_test_rows = len(ytest)
    Y_test_cols = len(ytest[0])
    Y_test_Labels = ytest.reshape(Y_test_rows)
    means_rows = len(means)
    means_cols = len(means[0])
    
    twopi = 2*np.pi
    pdf= np.zeros((test_Data_rows,means_cols));

    for i in range(means_cols):
        covmat_inv = np.linalg.inv(covmats[i])
        covmat_det = np.linalg.det(covmats[i])
        #covmat_det_sq = np.square(covmat_det)
        denom = np.power(twopi,test_Data_cols/2)*np.power(covmat_det,0.5)
        var = (test_Data - means[:,i])
        var2 = np.sum(var*np.transpose(np.dot(covmat_inv, np.transpose(var))),1)
        num = np.exp(-0.5*var2)
        pdf[:,i] = num/denom
        
    max_indice = np.argmax(pdf,1)
    ypred = max_indice + 1
    acc = np.mean(Y_test_Labels == ypred)*100
    return acc,ypred


# In[10]:

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 
	# IMPLEMENT THIS METHOD      
    var1 = np.dot(X.T,X)
    var1_inv = inv(var1)
    var2 = np.dot(X.T,y)
    w = np.dot(var1_inv,var2)                                     
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1              
    # IMPLEMENT THIS METHOD                             
    Data = X
    Data_rows = len(Data)
    Data_cols = len(Data[0])
    id_x = np.identity(Data_cols)
    #var = Data_rows*id_x
    var1 = np.dot(Data.T,Data)
    var2 = lambd*id_x
    var3 = inv(var1+var2)
    var4 = np.dot(var3,Data.T)
    w = np.dot(var4,y)
    
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    Data = Xtest
    N = len(Data)
    

    var1 = np.dot(Data,w)
    var2 = (ytest - var1)
    var3 = np.power(var2,2)
    var4 = np.sum(var3)
    mse = var4/N

    return mse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda    
   
    # IMPLEMENT THIS METHOD     
    Data = X
    Data_rows = len(Data)
    Data_cols = len(Data[0])
    w_rows = len(w)
    w_new = w.reshape((w_rows, 1))
    var = np.dot(X,w_new)
    var1 = np.subtract(y,var)
    var2 = np.dot(var1.T,var1)
    part1 = 0.5*var2
    var3 = np.dot(w_new.T,w_new)
    part2 = (0.5*lambd)*var3
    error = (part1+part2).flatten()
 
    var4 = np.dot(Data.T,Data)
    part3 = np.dot(var4,w_new)
    part4 = np.dot(Data.T,y)
    part5 = lambd*w_new
    interm = np.subtract(part3,part4)
    interm2= np.add(interm,part5)
    error_grad = (interm2).flatten()
                
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1)) 
    inp = x
    inp_rows = len(inp)
    Xd = np.zeros((inp_rows, p+1))
    for i in range(0, p+1):
        Xd[:,i] = np.power(x, i)

#     print(Xd)
    # IMPLEMENT THIS METHOD
    return Xd


# In[11]:

# Main script

# # Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA

means,covmat = ldaLearn(X,y)
ldaLearn(X,y)
#     print("Means")
#     print(means)
#     print("Covmat")
#     print(covmat)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
#print(means)
#print("COVMATS: ")
#print(covmats)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('QDA')

plt.show()
# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)
mle_tr = testOLERegression(w,X,y)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)
mle_i_tr = testOLERegression(w_i,X_i,y)

# print('Weight learnt w/o Intercept '+str(w))
# print('Weight learnt with Intercept '+str(w_i))
print('MSE without intercept test '+str(mle))
print('MSE with intercept test '+str(mle_i))
print('MSE without intercept train '+str(mle_tr))
print('MSE with intercept train '+str(mle_i_tr))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
#     print(lambd,mses3_train[i],mses3[i])
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
# plt.plot(lambdas,w_l)
# plot.title('Lambda vs Weight')
print('Printing Weights for Ridge Regression')
# print(w_l)
print('End print weights for Ridge Regression')
# print (lambdas,mses3_train,mses3)
# print(lambdas)
# print(mses3_train)
# print(mses3)
plt.show()

# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
print(lambdas)
print(mses4_train)
print(mses4)
plt.show()


# Problem 5
pmax = 7
# lambda_opt = 0 # REPLACE THIS WITH lambda_opt estimated from Problem 3
lamb1=lambdas[np.argmin(mses3)]
print('Optimal Lambda is'+str(lamb1))
lambda_opt = lamb1
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    print('MSE Train for Lambda 0 when p = '+str(p)+'is '+str(mses5_train[p,0]))
    print('MSE Test for Lambda 0 when p = '+str(p)+'is '+str(mses5[p,0]))
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
    print('MSE Train for Lambda Optimal when p = '+str(p)+'is '+str(mses5_train[p,1]))
    print('MSE Test for Lambda Optimal when p = '+str(p)+'is '+str(mses5[p,1]))

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
print(mses5_train)
print(mses5)
plt.show()


# In[ ]:



