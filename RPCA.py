import numpy as np
from scipy.sparse.linalg import svds

def RPCA(X, n_iter=1000):
    
    def shrink(X,tau):
        Y = np.abs(X)-tau
        return np.sign(X) * np.maximum(Y,np.zeros_like(Y))
    
    def SVT(X,tau):
        U,S,VT = np.linalg.svd(X,full_matrices=0)
        out = U @ np.diag(shrink(S,tau)) @ VT
        return out
    
    n1,n2 = X.shape
    mu = n1*n2/(4*np.sum(np.abs(X.reshape(-1))))
    lambd = 1/np.sqrt(np.maximum(n1,n2))
    thresh = 10**(-7) * np.linalg.norm(X)
    
    S = np.zeros_like(X)
    Y = np.zeros_like(X)
    L = np.zeros_like(X)
    count = 0
    while (np.linalg.norm(X-L-S) > thresh) and (count < n_iter):
        L = SVT(X-S+(1/mu)*Y,1/mu)
        S = shrink(X-L+(1/mu)*Y,lambd/mu)
        Y = Y + mu*(X-L-S)
        count += 1
        
    return L,S

def RPCA_partial(X, rank=10, n_iter=1000):
    
    def shrink(X,tau):
        Y = np.abs(X)-tau
        return np.sign(X) * np.maximum(Y,np.zeros_like(Y))
    
    def SVT(X,tau):
        U,S,VT = svds(X.astype(float),k=rank)
        out = U @ np.diag(shrink(S,tau)) @ VT
        return out
    
    n1,n2 = X.shape
    mu = n1*n2/(4*np.sum(np.abs(X.reshape(-1))))
    lambd = 1/np.sqrt(np.maximum(n1,n2))
    thresh = 10**(-7) * np.linalg.norm(X)
    
    S = np.zeros_like(X)
    Y = np.zeros_like(X)
    L = np.zeros_like(X)
    count = 0
    while (np.linalg.norm(X-L-S) > thresh) and (count < n_iter):
        L = SVT(X-S+(1/mu)*Y,1/mu)
        S = shrink(X-L+(1/mu)*Y,lambd/mu)
        Y = Y + mu*(X-L-S)
        count += 1
        
    return L,S
