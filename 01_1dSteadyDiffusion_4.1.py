#[1]An Introduction to Computational Fluid Dynamics: Versteeg & Malalasekera 
#Example 4.1

import numpy as np

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
length = 0.5 #m
area = 1e-2 #m2

#material property
k = 1000

#bc: End temperatures
Ta = 500
Tb = 500

nx = 5
dx = length/nx

#define arrays
T = np.zeros(nx)
aw = np.ones(nx)*k*area/dx
ae=np.ones(nx)*k*area/dx
ap = ae+aw
d=np.zeros(nx) # no heat source 

#end nods need special care:

#first node
#[1] pg.No. 120: the fixed temperature boundary condition enters the
#calculation as a source term (Su + SpTp) with Su = (2kA/δx)Ta and Sp = −2kA/δx
aw[0]=0 #no west node for first node
ap[0] = aw[0]+ae[0]+2*k*area/dx
d[0] = (2*k*area/dx)*Ta

#last node
ae[nx-1]=0 #no east node for last node
ap[nx-1] = aw[nx-1]+ae[nx-1]+2*k*area/dx
d[nx-1] = (2*k*area/dx)*Tb

#-ve sign for ae and aw as tdma expects Ax=b matrix
#whereas we made like apTp = awTw+aeTe+Su
tdma(ap,-ae,-aw,d,T)
print(T)