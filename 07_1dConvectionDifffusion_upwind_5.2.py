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

#Case
case="case1"

#geometric
length = 1 #m

#fluid properties
rho=1
k=0.1

if(case=="case1"):
    nx=5
    velocity=-0.1
elif(case=="case2"):
    nx=5
    velocity=2.5
else:#case3 or junk case :)
    nx=20
    velocity=2.5

dx = length/nx

#bc and initial conditions

Ta=1
Tb=0

F = rho*velocity
D = k/dx

#declaring and initialising arrays
u = np.ones(nx)*velocity
T=np.ones(nx)
aw=np.ones(nx)*(D+F)
ae=np.ones(nx)*D
Sp=np.zeros(nx)
Su=np.zeros(nx)

#leftmost node
aw[0]=0
Sp[0]=-(2*D+F)
Su[0]=(2*D+F)*Ta
#rightmost node
ae[nx-1]=0
Sp[nx-1]=-2*D
Su[nx-1]=2*D*Tb
ap=aw+ae-Sp

#solve
tdma(ap,-ae,-aw,Su,T)

#exact solution
x=np.linspace(0,length,nx+1)+dx/2
x=x[:-1]
Texact = Ta+(Tb-Ta)* (np.exp(rho*velocity*x/k) -1) / (np.exp(rho*velocity*length/k) -1)

print(T)
print(Texact)

#plot
plt.plot(x,T,linestyle='--' ,marker='o',color='b',label='Numerical Solution [Upwind]')
plt.plot(x,Texact,'-gD',label='Exact Solution')

plt.title("Comparison of distribution of property")
plt.xlabel("Position")
plt.ylabel("Property")
plt.legend()
plt.grid()
figname="comparison_5.2_"+case
plt.savefig(figname+".png")
plt.show()
