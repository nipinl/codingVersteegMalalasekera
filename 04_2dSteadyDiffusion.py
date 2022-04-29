#[1]An Introduction to Computational Fluid Dynamics: Versteeg & Malalasekera 
#2D conduction

import numpy as np
import copy

def tdma(a_p,a_w,a_e,a_s,a_n,Su,T_,j):
    '''solves tri-diagonal matrix for 2D case. a:main diagonal, above is b and below is c. d is rhs.
    T is updated with solution, j is y index
     '''
    a=a_p[:,j]
    b=-a_e[:,j]
    c=-a_w[:,j]
    if(j==0):
        d=Su[:,j]+a_n[:,j]*T_[:,j+1]
    elif(j==ny-1):
        d=Su[:,j]+a_s[:,j]*T_[:,j-1]
    else:
        d=Su[:,j]+a_s[:,j]*T_[:,j-1]+a_n[:,j]*T_[:,j+1]
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
    T_[N-1,j]=Q[N-1]
    for i in range(N-2,-1,-1):
        T_[i,j]=P[i]*T_[i+1,j]+Q[i]
#geometry
length = 3.0 #m
width = 3.0 #m

nx = 5
ny = 5


#material property
k = 100

#Constants
sigma = 5.67e-8


#bc
#=======================LEFT BONDARY CONDITION=====================
lbc=4   #select appropriate bondary conditions here
#1 const temp bc
TLeft=100        

#2 const heat flux bc
qDotLeft=1000    

#3 convection
TinfLeft=20
hLeft=2

#4 radiation
TambLeft=1000
epsilonLeft=0.7

#=======================RIGHT BONDARY CONDITION=====================
rbc=3   #select appropriate bondary conditions here
#1 const temp bc
TRight=100       

#2 const heat flux bc
qDotRight=1000   

#3 convection
TinfRight=20
hRight=2

#4 radiation
TambRight=24
epsilonRight=0.7
#=======================BOTTOM BONDARY CONDITION=====================
bbc=2   #select appropriate bondary conditions here
#1 const temp bc
TBottom=100        

#2 const heat flux bc
qDotBottom=10

#3 convection
TinfBottom=30
hBottom=2

#4 radiation
TambBottom=24
epsilonBottom=0.7

#=======================TOP BONDARY CONDITION=====================
tbc=1 #select appropriate bondary conditions here
#1 const temp bc
TTop=100        

#2 const heat flux bc
qDotTop=24    

#3 convection
TinfTop=30
hTop=2

#4 radiation
TambTop=24
epsilonTop=0.7

#heat generation per unit volume
qDotV=0

#initial condition
Tinit = 100



dx = length/nx
dy = width/ny

#qdot.dV = Su+SpTp
#-h(Tp-Tinf).Adx = Su+SpTp
#Sp = -hAdx


#define arrays
T = np.ones((nx,ny))*Tinit

a_w =   np.ones((nx,ny))*k*(dy*1)/dx
a_e =   np.ones((nx,ny))*k*(dy*1)/dx
a_n =   np.ones((nx,ny))*k*(dx*1)/dy
a_s =   np.ones((nx,ny))*k*(dx*1)/dy #sadly, 'as' is a builtin word so 'a_s'

# add heat generation source in Su
Su = np.ones((nx,ny))*qDotV*dx*dy*1
Sp = np.zeros((nx,ny))

print("aw:")
print(a_w)
print("ae:")
print(a_e)
print("as:")
print(a_s)
print("an:")
print(a_n)
print("Su:")
print(Su)
print("Sp:")
print(Sp)
print("going to add bc contrib")
# Adding bcs into Su and Sp
## Left boundary
a_w[0,:] = 0    #no west node left of left bounary
if(lbc==1):
    Sp[0,:] += -2*k*(dy*1)/dx
    Su[0,:] += (2*k*(dy*1)/dx)*TLeft
elif(lbc==2):
    Su[0,:] += qDotLeft*dy*1
elif(lbc==3):
    Sp[0,:] += -hLeft*(dy*1)
    Su[0,:] += hLeft*(dy*1)*TinfLeft
elif(lbc==4):
    Sp[0,:] += -4*sigma*epsilonLeft*(dy*1)*T[0,:]**3 #refer [2]Exmpl#1 pg. 537
    Su[0,:] += sigma*epsilonLeft*(dy*1)*(TambLeft**4+3*T[0,:]**4)
else:
    print("Wrong bc on left!!")

## Right boundary
a_e[nx-1,:] = 0 #no east node right of right bounary
if(rbc==1):
    Sp[nx-1,:] += -2*k*(dy*1)/dx
    Su[nx-1,:] += (2*k*(dy*1)/dx)*TRight
elif(rbc==2):
    Su[nx-1,:] += qDotRight*dy*1
elif(rbc==3):
    Sp[nx-1,:] += -hRight*(dy*1)
    Su[nx-1,:] += hRight*(dy*1)*TinfRight
elif(rbc==4):
    Sp[nx-1,:] += -4*sigma*epsilonRight*(dy*1)*T[nx-1,:]**3 #refer [2]Exmpl#1 pg. 537
    Su[nx-1,:] += sigma*epsilonRight*(dy*1)*(TambRight**4+3*T[nx-1,:]**4)
else:
    print("Wrong bc on right!!")

## Bottom boundary
a_s[:,0] = 0    #no node below bottom bounary
if(bbc==1):
    Sp[:,0] += -2*k*(dx*1)/dy
    Su[:,0] += (2*k*(dx*1)/dy)*TBottom
elif(bbc==2):
    Su[:,0] += qDotBottom*dx*1
elif(bbc==3):
    Sp[:,0] += -hBottom*(dx*1)
    Su[:,0] += hBottom*(dx*1)*TinfBottom
elif(bbc==4):
    Sp[:,0] += -4*sigma*epsilonBottom*(dx*1)*T[:,0]**3 #refer [2]Exmpl#1 pg. 537
    Su[:,0] += sigma*epsilonBottom*(dx*1)*(TambBottom**4+3*T[:,0]**4)
else:
    print("Wrong bc on left!!")

## Top boundary
a_n[:,ny-1] = 0 #no node above top bounary
if(tbc==1):
    Sp[:,ny-1] += -2*k*(dx*1)/dy
    Su[:,ny-1] += (2*k*(dx*1)/dy)*TTop
elif(tbc==2):
    Su[:,ny-1] += qDotTop*dx*1
elif(tbc==3):
    Sp[:,ny-1] += -hTop*(dx*1)
    Su[:,ny-1] += hTop*(dx*1)*TinfTop
elif(tbc==4):
    Sp[:,ny-1] += -4*sigma*epsilonTop*(dx*1)*T[:,ny-1]**3 #refer [2]Exmpl#1 pg. 537
    Su[:,ny-1] += sigma*epsilonTop*(dx*1)*(TambTop**4+3*T[:,ny-1]**4)
else:
    print("Wrong bc on left!!")

a_p = a_e + a_w + a_s + a_n - Sp

iterations=1000
for t in range(iterations):
    if(t==iterations-1):
        Told = copy.deepcopy(T)
    for j in range(ny):
        tdma(a_p,a_w,a_e,a_s,a_n,Su,T,j)
    if(t==iterations-1):
        err = np.abs((Told-T)/T).max()
        print("maximum error is ",err*100,"%")
    

print("T:")
print(T)
