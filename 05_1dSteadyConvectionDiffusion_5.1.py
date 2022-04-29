#[1]An Introduction to Computational Fluid Dynamics: Versteeg & Malalasekera 
#Example 4.2

import numpy as np
import matplotlib.pyplot as plt

def tdma(a,b,c,d,T):
    '''solves tri-diagonal matrix. a:main diagonal, above is b and below is c. d is rhs.
    T is updated with solution
     '''
    N=len(a)
    P=np.ones(N)
    Q=np.ones(N)
    P[0]=-b[0]/a[0]
    Q[0]= d[0]/a[0]
    #denom=1.0
    for i in range(1,N):
        denom=1/(a[i]+c[i]*P[i-1])
        P[i] = -b[i]*denom
        Q[i] = (d[i]-c[i]*Q[i-1])*denom
    T[N-1]=Q[N-1]
    for i in range(N-2,-1,-1):
        T[i]=P[i]*T[i+1]+Q[i]
#geometry
length = 1.0 #m
area = 1 #m2  

#material property
k = 0.1
rho = 1

#bc: End temperatures
Ta = 1
Tb = 0
 
nx = 20# case1: 5, ase 2: 20
dx = length/nx

velocity = 2.5 #case 1: 0.1, case 2&3: 2.5
D = k/dx
F = rho*velocity

#define arrays
u=np.ones(nx)*velocity 
T = np.zeros(nx)
aw = np.ones(nx)*(D+F/2)
ae = np.ones(nx)*(D-F/2)
Sp = np.zeros(nx)
Su = np.zeros(nx)
#Left most node
aw[0]=0
Sp[0] = -(2*D+F)
Su[0] = (2*D+F)*Ta
#Right most node
ae[nx-1]=0
Sp[nx-1] = -(2*D-F)
Su[nx-1] = (2*D-F)*Tb

ap = ae+aw-Sp

print("ap:")
print(ap)
print("ae:")
print(ae)
print("aw:")
print(aw)
print("Su:")
print(Su)
#-ve sign for ae and aw as tdma expects Ax=b matrix
#whereas we made like apTp = awTw+aeTe+Su
tdma(ap,-ae,-aw,Su,T)
print("T:")
print(T)

x=np.linspace(0,length,nx+1)+dx/2
x=x[:-1]
Texact=np.ones(nx)
for i in range(nx):
    Texact[i] = 1- (np.exp(rho*velocity*x[i]/k) -1)/(np.exp(rho*velocity*length/k) -1)

fig = plt.figure(figsize=(10,10))
plt.plot(x,Texact,linestyle='--', marker='o', color='b', label='Exact Solution')
plt.plot(x,T,'-gD', label='Numerical Solution(Central)')
plt.grid()
plt.legend()
plt.title("Comparison of distribution of property along length")
plt.xlabel("Position (m)")
plt.ylabel("Property")
plt.show()
plt.savefig("compare")
