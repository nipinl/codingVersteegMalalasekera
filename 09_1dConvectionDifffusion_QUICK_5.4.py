#Upwind for convection term. CD for diffusion term

import numpy as np
import matplotlib.pyplot as plt


def tdma(a,b,c,d,T):
    """ a: ap(main diagonal), b: -ae, c: -aw, d: Su, T: variable vector """
    N=len(a)
    P=np.ones(N)
    Q=np.ones(N)
    P[0]=-b[0]/a[0]
    Q[0]= d[0]/a[0]
    for i in range(1,N):
        denom=1/(a[i]+c[i]*P[i-1])
        P[i] = -b[i]*denom
        Q[i] = (d[i]-c[i]*Q[i-1])*denom
    T[N-1]=Q[N-1]
    for i in range(N-2,-1,-1):
        T[i]=P[i]*T[i+1]+Q[i]



#geometric
length = 1 #m

#fluid properties
rho=1
k=0.1


nx=5
velocity=-0.2


dx = length/nx

#bc and initial conditions

Ta=1
Tb=0

F = rho*velocity
D = k/dx

#declaring and initialising arrays
u = np.ones(nx)*velocity
T=np.ones(nx)
aww=np.ones(nx)*(-F/8)
aw=np.ones(nx)*(D+7*F/8)
ae=np.ones(nx)*(D-3*F/8)
Sp=np.zeros(nx)
Su=np.zeros(nx)

#node 0
aww[0]=0
aw[0]=0
ae[0]=4/3*D-3/8*F
Sp[0]=-(8/3*D+10/8*F)
Su[0]=(8/3*D+10/8*F)*Ta
#node 1
aww[1]=0
aw[1]=D+F
ae[1]=D-3/8*F
Sp[1]=F/4
Su[1]=-F/4*Ta
#rightmost node
aww[nx-1]=-F/8
aw[nx-1]=4/3*D+6/8*F
ae[nx-1]=0
Sp[nx-1]=-(8/3*D-F)
Su[nx-1]=(8/3*D-F)*Tb
ap=aww+aw+ae-Sp
print("aww =",aww)
print("aw =",aw)
print("ae =",ae)
print("Sp =",Sp)
print("ap =",ap)
print("Su =",Su)

A = np.empty(shape=(nx,nx))
A.fill(0)
for i in range(nx):
    A[i][i]=ap[i]
for i in range(nx-1):
    A[i][i+1]=-ae[i]
for i in range(1,nx):
    A[i][i-1]=-aw[i]
for i in range(2,nx):
    A[i][i-2]=-aww[i]
print("A= ")
print(A)
T = np.dot(np.linalg.inv(A),Su)


#exact solution
x=np.linspace(0,length,nx+1)+dx/2
x=x[:-1]
Texact = Ta+(Tb-Ta)* (np.exp(rho*velocity*x/k) -1) / (np.exp(rho*velocity*length/k) -1)

print("T = ", T)
print("Texact = ",Texact)

#plot
plt.plot(x,T,linestyle='--' ,marker='o',color='b',label='Numerical Solution [QUICK]')
plt.plot(x,Texact,'-gD',label='Exact Solution')

plt.title("Comparison of distribution of property")
plt.xlabel("Position")
plt.ylabel("Property")
plt.legend()
plt.grid()
figname="Example_5.4"
plt.savefig(figname+".png")
plt.show()
