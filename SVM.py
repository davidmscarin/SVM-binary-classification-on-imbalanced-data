import numpy as np
import cvxopt as cvxopt
from cvxopt import solvers
import matplotlib.pyplot as plt
import pylab as pl
from sklearn.metrics import accuracy_score
import seaborn as sns
import pandas as pd
import itertools

def plotD(df,pred,att1=0,att2=1):
	frame=pd.DataFrame(df)
	frame['class']=pd.DataFrame(pred)
	sns.scatterplot(data=frame, x=att1, y=att2, hue='class')
    
def score(y_true, y_pred):
    print(f"accuracy: {accuracy_score(getPut(y_true),getPut(y_pred)):.3f}")

#hyperparameter tuning was taking too long to find the parameter combination, most likely because of some quirk of the SVM, so we did not include it
def tuning(frame, max_iters=5):
    hp = find_hyperparameters(frame, max_iters)
    y_pred, y_true, _ = sequence(frame, k=hp.get('kernel'), c_=hp.get('C'), sigma=hp.get('sigma'), dg=hp.get('degree'), threshold=hp.get('threshold'))
    print("best parameter combination found: ", hp)
    score(y_true, y_pred)
    err_min_cls(y_true, y_pred)

def find_hyperparameters(frame, max_iters=5):
	kernels = ['poly']
	Cs = [0.1, 1, 10]
	sigmas = [0.5, 1, 2]
	degrees = [1, 2, 3]
	thresholds = [1e-4, 1e-5, 1e-6]

	iters=0

	best_accuracy = 0
	best_hyperparameters = {}

	for kernel, C, sigma, degree, threshold in itertools.product(kernels, Cs, sigmas, degrees, thresholds):
		y_pred, y_true, _ = sequence(frame, c_=C, k = kernel, dg = degree, threshold = threshold, sigma = sigma)
		cur_accuracy = accuracy_score(getPut(y_true),getPut(y_pred))
		iters+=1
		if(cur_accuracy>best_accuracy):
			best_accuracy = cur_accuracy
			best_hyperparameters = {'kernel': kernel, 'C': C, 'sigma': sigma, 'degree': degree, 'threshold': threshold}
		if(iters==max_iters):
			return best_hyperparameters

	return best_hyperparameters

#Target matrix cration for classification
def target_normalization(frame,n_class=2):
    target=-np.ones((np.shape(frame)[0],n_class),dtype=float)
    for i in range(n_class):
        indices=np.where(frame[:,len(frame[0])-1]==i)
        target[indices,i]=1
    return target

def noneCgen(n_class):
	return [1 for i in range(n_class)]

#SVM pipeline
def sequence(frame,c_=noneCgen(2),n_class=2,k='linear',dg=1,wc=False, threshold = 1e-5, sigma=1):
	if not wc:
		c_=noneCgen(n_class)
	target=target_normalization(frame,n_class)
	Xtrain = frame[::2,0:len(frame[0])-1]
	Ytrain = target[::2]
	Xtest = frame[1::2,0:len(frame[0])-1]
	Ytest = target[1::2]
	output = np.zeros((np.shape(Xtest)[0],n_class))
	for i in range(n_class):
		svc=svm(kernel=k,C=c_[i],degree=dg)
		svc.fit(Xtrain,Ytrain,i)
		output=svc.predict(output,Xtest,i)
	return output,Ytest,Xtest

def getPut(output):
    return np.argmax(output,axis=1)

def err_min_cls(mby_test, mby_pred): 
	
	predlist = getPut(mby_pred)
	true_list = getPut(mby_test)
	count=0

	for i in range(len(true_list)):
		if(true_list[i]==1 and predlist[i]!=1):
			count+=1
		
	print("Misslabeled minority class examples: ", count)

class svm:

	def __init__(self,kernel='linear',C=None,sigma=1.,degree=1.,threshold=1e-5):
		self.kernel = kernel
		if self.kernel == 'linear':
			self.kernel = 'poly'
			self.degree = 1.
		self.C = C
		self.sigma = sigma
		self.degree = degree
		self.threshold = threshold

	def build_kernel(self,X):
		self.K = np.dot(X,X.T)
		if self.kernel=='poly':
			self.K = (1. + 1./self.sigma*self.K)**self.degree
		elif self.kernel=='rbf':
			self.xsquared = (np.diag(self.K)*np.ones((1,self.N))).T
			b = np.ones((self.N,1))
			self.K -= 0.5*(np.dot(self.xsquared,b.T) + np.dot(b,self.xsquared.T))
			self.K = np.exp(self.K/(2.*self.sigma**2))

	def fit(self,Xtrain,Ytrain,i_class):
		self.train_svm(Xtrain,np.reshape(Ytrain[:,i_class],(np.shape(Xtrain[:,:len(Xtrain-2)])[0],1)),i_class)

	def predict(self,output,test,i_class):
		output[:,i_class]=self.classifier(test,soft=True).T
		return output

	def train_svm(self,X,targets,i_c,print_info=False):
		#initial_targets.replace({0:-1}, inplace=True)
		#X = initial_X.to_numpy().reshape((-1, initial_X.shape[1]))
		#targets = initial_targets.to_numpy().reshape((-1, 1)) + 0.
		self.N = np.shape(X)[0]
		self.build_kernel(X)
		# Assemble the matrices for the constraints
		P = targets*targets.transpose()*self.K
		q = -np.ones((self.N,1))
		if self.C is None:
			G = -np.eye(self.N)
			h = np.zeros((self.N,1))
		else:
			G = np.concatenate((np.eye(self.N),-np.eye(self.N)))
			h = np.concatenate((self.C*np.ones((self.N,1)),np.zeros((self.N,1))))
		A = targets.reshape(1, self.N)
		b = 0.0
		
		# Call the quadratic solver
		if (not print_info):
			solvers.options["show_progress"]=False
		else:
			solvers.options["show_progress"]=True
		sol = cvxopt.solvers.qp(cvxopt.matrix(P),cvxopt.matrix(q),cvxopt.matrix(G),cvxopt.matrix(h), cvxopt.matrix(A), cvxopt.matrix(b))
		
		# Get the Lagrange multipliers out of the solution dictionary
		lambdas = np.array(sol['x'])
		
		# Find the (indices of the) support vectors, which are the vectors with non-zero Lagrange multipliers
		self.sv = np.where(lambdas>self.threshold)[0]
		self.nsupport = len(self.sv)
		print (self.nsupport, "support vectors found") 
		
		# Just retain the data corresponding to the support vectors
		self.X = X[self.sv,:]
		self.lambdas = lambdas[self.sv]
		self.targets = targets[self.sv]
    	
		#self.b = np.sum(self.targets)
    	#for n in range(self.nsupport):
		#self.b -= np.sum(self.lambdas*self.targets.T*np.reshape(self.K[self.sv[n],self.sv],(self.nsupport,1)))
        #self.b /= len(self.lambdas)
		#print ("b=",self.b)
		self.b = np.sum(self.targets)
		for n in range(self.nsupport):
			self.b -= np.sum(self.lambdas*self.targets*np.reshape(self.K[self.sv[n],self.sv],(self.nsupport,1)))
			self.b /= len(self.lambdas)
		#print "b=",self.b

		#bb = 0
		#for j in range(self.nsupport):
			#tally = 0	
			#for i in range(self.nsupport):
				#tally += self.lambdas[i]*self.targets[i]*self.K[self.sv[j],self.sv[i]]
			#bb += self.targets[j] - tally
		#self.bb = bb/self.nsupport
		#print self.bb
				
		if self.kernel == 'poly':
			def classifier(Y,soft=False):
				K = (1. + 1./self.sigma*np.dot(Y,self.X.T))**self.degree
				self.y = np.zeros((np.shape(Y)[0],1))
				for j in range(np.shape(Y)[0]):
					for i in range(self.nsupport):
						self.y[j] += self.lambdas[i]*self.targets[i]*K[j,i]
					self.y[j] += self.b
				if soft:
					return self.y
				else:
					return np.sign(self.y)
	
		elif self.kernel == 'rbf':
			def classifier(Y,soft=False):
				#Y = Y_initial.to_numpy().reshape((-1, Y_initial.shape[1]))
				K = np.dot(Y,self.X.T)
				c = (1./self.sigma * np.sum(Y**2,axis=1)*np.ones((1,np.shape(Y)[0]))).T
				c = np.dot(c,np.ones((1,np.shape(K)[1])))
				aa = np.dot(self.xsquared[self.sv],np.ones((1,np.shape(K)[0]))).T
				K = K - 0.5*c - 0.5*aa
				K = np.exp(K/(2.*self.sigma**2))

				self.y = np.zeros((np.shape(Y)[0],1))
				for j in range(np.shape(Y)[0]):
					for i in range(self.nsupport):
						self.y[j] += self.lambdas[i]*self.targets[i]*K[j,i]
					self.y[j] += self.b

				if soft:
					return self.y
				else:
					return np.sign(self.y)
		else:
			print ("Error -- kernel not recognised")
			return

		self.classifier = classifier