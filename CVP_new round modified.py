# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 12:04:24 2020

@author: HyewonSung
"""

from fractions import Fraction
from typing import Sequence
from typing import List
import numpy as np
import math
from functools import reduce
from random import randrange, getrandbits





class Vector(list):
            
    def sdot(self) :  
       
        return self.dot(self)
    
    def dot(self, rhs: "Vector") :
        """
        >>> Vector([1, 2, 3]).dot([4, 5, 6])
        Fraction(32, 1)
        """
        rhs = Vector(rhs)
        assert len(self) == len(rhs)
        return sum(map(lambda x: x[0] * x[1], zip(self, rhs)))
    def proj_coff(self, rhs: "Vector") :
        """
        >>> Vector([1, 1, 1]).proj_coff([-1, 0, 2])
        Fraction(1, 3)
        """
        rhs = Vector(rhs)
        assert len(self) == len(rhs)
        return self.dot(rhs) / self.sdot()
    def proj(self, rhs: "Vector") -> "Vector":
        """
        >>> Vector([1, 1, 1]).proj([-1, 0, 2])
        [1/3, 1/3, 1/3]
        """
        rhs = Vector(rhs)
        assert len(self) == len(rhs)
        return self.proj_coff(rhs) * self
    def __sub__(self, rhs: "Vector") -> "Vector":
        """
        >>> Vector([1, 2, 3]) - [6, 5, 4]
        [-5, -3, -1]
        """
        rhs = Vector(rhs)
        assert len(self) == len(rhs)
        return Vector(x - y for x, y in zip(self, rhs))
    def __mul__(self, rhs) -> "Vector":
        """
        >>> Vector(["3/2", "4/5", "1/4"]) * 2
        [3, 8/5, 1/2]
        """
        return Vector(x * rhs for x in self)
    def __rmul__(self, lhs) -> "Vector":
        """
        >>> 2 * Vector(["3/2", "4/5", "1/4"])
        [3, 8/5, 1/2]
        """
        return Vector(x * lhs for x in self)
    def __repr__(self) -> str:
        return "[{}]".format(", ".join(str(x) for x in self))
    
def gramschmidt(v: Sequence[Vector]) -> Sequence[Vector]:
    """
    >>> gramschmidt([[3, 1], [2, 2]])
    [[3, 1], [-2/5, 6/5]]
    >>> gramschmidt([[4, 1, 2], [4, 7, 2], [3, 1, 7]])
    [[4, 1, 2], [-8/7, 40/7, -4/7], [-11/5, 0, 22/5]]
    """
    u: List[Vector] = []
    for vi in v:
        ui = Vector(vi)
        for uj in u:
            ui = ui - uj.proj(vi)
        if any(ui):
            u.append(ui)
    return u


def dot(a,b):
    a=np.array(a)
    b=np.array(b)
    u=a*b
    #print(u)
    C=u[0]
    i=1
    while i<len(a):
       C=C+u[i]
       i=i+1
    return C

def mult(v,B):
    B=np.matrix(B)
    v=np.array(v)
    #print(v,B)
    C=v*(B)
    #print(C)
    return sum(map(lambda x: x , C))

def new_round(num) :
    if (num - math.floor(num) <= 1/2) :
        return math.floor(num) 
    else :
        return math.ceil(num)  

def LLLreduction(basis: Sequence[Sequence[int]], delta: float):
 #-> Sequence[Sequence[int]]:
    
    n = len(basis)
    basis = list(map(Vector, basis))
    ortho = gramschmidt(basis)
    def mu(i: int, j: int) :
        return ortho[j].proj_coff(basis[i])
    k = 1
    while k < n:
        #print("(k=", k,")",)
        for j in range(k - 1, -1, -1):
            mu_kj = mu(k, j)
            if abs(mu_kj) > 0.5:
                basis[k] = basis[k] - basis[j] * new_round(mu_kj)
                ortho = gramschmidt(basis)
                #print("(k,j)=",k,j, basis)
                #print("(k,j)=",k,j, ortho)
        #print("size reduced basis k=", k, basis)
        #print("GSO of size reduced basis k=", k, ortho)
        #print("*****************")
        if ortho[k].dot(ortho[k])>= (delta-((mu(k, k-1))**2))*(ortho[k-1].dot(ortho[k-1])):
            k += 1
            #print("Lovasz holds")
        else:
            #print("Lovasz fails")
            basis[k], basis[k - 1] = basis[k - 1], basis[k]
            ortho = gramschmidt(basis)
            k = max(k - 1, 1)
        #print("updated basis=", basis, "updated GSO=",ortho)
        #print("*****************")
    return basis
    #return [list(map(int, b)) for b in basis]

    


def Babai_Nearst(B, w):
    #print(w,B)      
    T= gramschmidt(B)
    i=len(B)-1
    v=np.zeros(len(B))
    while i>=0:       
        y1=dot(w[0],T[i])
        y2=dot(T[i],T[i])   
        y=y1/y2
        a=new_round(y)        
        v=v+a*B[i]
        w=w-a*B[i]-(a-new_round(a))*T[i]
        i=i-1    
    return v
    #return sum(map(lambda x: x, h)) 

    
def maxL2(B):
    h=dot(B[0],B[0]) 
    #print(h)
    i=1
    while i <len(B):        
        k=dot(B[i],B[i])
        h=max(h,k)
        #print(i,h)
        i=i+1
        
    return h**0.5
    
def CVP_embed(B, w):
    #print("input: ",w, B, len(B))
    n=len(B)
    M=maxL2(B)
    M=int(M)+1
    #M=1
    #print("M=",M)
    w=np.append(w,M)
    #print(w, type(w))
    i=0
    h=[]
    while i <len(B):
        x=np.append(B[i],0)
        h.append(x)
        i=i+1
        #print(i,h)
    h.append(w)
    #print(h, type(h))
    h=np.array(h)
    #print(h, type(h))
    W=LLLreduction(h,0.75)
    #print(W)
    x=W[n]
    x=x[:n]
    v=w[:n]-x
    return v
        
def Babai_round(B, w):
    #print("input: ",w, B)
    Q=np.linalg.inv(B)
    Q=np.array(Q)    
    y=mult(w,Q)
    #print("coefficient vector =",y)
    #print("correct coefficient?", mult(y,B)-w)
    h=[]
    for i in range(len(B)):
      h.append(new_round(y[0,i]) * B[i])
    #print("h array :", np.array(h))               
    return sum(map(lambda x: x, h))


def main(): 
  n=3
  #B=np.random.randint(low=-2**5,high=2**5,size=(n,n))  
  #w=np.random.randint(low=-2**5,high=2**5, size=(1,n)) 
  B=np.array([[-265,287,56],[-460,448,72],[-50,49,8]])
  w=np.array([[100,80,100]])
  print("basis: \n", B, "\n", "target vector",w)
  C=LLLreduction(B,0.75)
  print("LLL reduced basis",C)
  C=np.array(C) 
  v1=CVP_embed(B, w)
  v2=Babai_Nearst(C, w)  
  v3=Babai_Nearst(B, w)
  v4=Babai_round(B, w)
 
   
  print("Babai Nearest output", Babai_Nearst(B, w),np.linalg.norm(v3-w))
   
  print("Babai Rounding output", Babai_round(B, w),np.linalg.norm(v4-w))  
  
  print("******************")
  print("Embedding output", CVP_embed(B, w), np.linalg.norm(v1-w))
  print("Babai LLL-Nearest output", Babai_Nearst(C, w), np.linalg.norm(v2-w))
  
  
 
  
  
        
        
        
   
  
if __name__ == '__main__': 
    main() 
    

