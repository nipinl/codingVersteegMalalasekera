#QUICK is unstable at large Peclet numbers. Equate k =0, ap will be zero at some places and gauss siedel will crash
#QUICK for convection term. CD for diffusion term
#velocity values are known

import numpy as np
import matplotlib.pyplot as plt
#gauss_seidel fn taken from: https://johnfoster.pge.utexas.edu/numerical-methods-book/LinearAlgebra_IterativeSolvers.html
def gauss_seidel(A, b, tolerance=1e-10, max_iterations=10000):
    
    x = np.zeros_like(b, dtype=np.double)
    
    #Iterate
    for k in range(max_iterations):
        
        x_old  = x.copy()
        
        #Loop over rows
        for i in range(A.shape[0]):
            x[i] = (b[i] - np.dot(A[i,:i], x[:i]) - np.dot(A[i,(i+1):], x_old[(i+1):])) / A[i ,i]
            
        #Stop condition 
        if np.linalg.norm(x - x_old, ord=np.inf) / np.linalg.norm(x, ord=np.inf) < tolerance:
            print("converged at iteration no: ",i)
            break

    return x

a = lambda vel:max(np.sign(vel),0) #return 1 if vel>0 else 0. for alpha_e, etc..

#geometric
length = 0.3 #m
width = 0.3 #m

#fluid properties
rho=1
k=1.0

nx=10

ny=nx

dx = length/nx
dy = width/ny

#bc and initial conditions

#initialisation
u = 2 #everywhere in the domain
v = 2


T = np.empty(shape=(nx,ny))
T.fill(0)
for i in range(nx):
    for j in range(ny):
        if(i<=j):
            T[i,j]=100

T[0,0]=50
T[nx-1,ny-1]=50


Fx = rho*u*dy
Dx = k/dx*dy
Fy = rho*v*dx
Dy = k/dy*dx
""" aww=np.ones(nx)*(-F/8)
aw=np.ones(nx)*(D+7*F/8)
ae=np.ones(nx)*(D-3*F/8)
Sp=np.zeros(nx)
Su=np.zeros(nx) """

a_e = np.empty(shape=(nx,ny))
a_e.fill(Dx-(3/8)*a(u)*Fx-(6/8)*(1-a(u))*Fx-(1/8)*(1-a(u))*Fx )
a_w = np.empty(shape=(nx,ny))
a_w.fill(Dx+(6/8)*a(u)*Fx+(1/8)*a(u)*Fx+(3/8)*(1-a(u))*Fx)
a_n = np.empty(shape=(nx,ny))
a_n.fill(Dy-(3/8)*a(v)*Fy-(6/8)*(1-a(v))*Fy-(1/8)*(1-a(v))*Fy )
a_s = np.empty(shape=(nx,ny))
a_s.fill(Dy+(6/8)*a(v)*Fy+(1/8)*a(v)*Fy+(3/8)*(1-a(v))*Fy)

a_ee = np.empty(shape=(nx,ny))
a_ee.fill((1/8)*(1-a(u))*Fx)

a_ww = np.empty(shape=(nx,ny))
a_ww.fill(-(1/8)*a(u)*Fx)

a_nn = np.empty(shape=(nx,ny))
a_nn.fill((1/8)*(1-a(v))*Fy)

a_ss = np.empty(shape=(nx,ny))
a_ss.fill(-(1/8)*a(v)*Fy)

Sp = np.empty(shape=(nx,ny))
Sp.fill(0)
Su = np.empty(shape=(nx,ny))
Su.fill(0)



#left bc
Tleft=100
a_ww[0,:] = 0
a_ww[1,:] = 0
a_w[0,:] = 0
Sp[0,:] += -((8/3)*Dx+(2/8)*a(u)*Fx+Fx) 
Su[0,:] += ((8/3)*Dx+(2/8)*a(u)*Fx+Fx) * Tleft
Sp[1,:] += a(u)*Fx/4
Su[1,:] += -a(u)*Fx* Tleft/4

#right bc
Tright=0
#insulated-> Sp=Su=0
a_e[nx-1,:] = 0
a_ee[nx-1,:] = 0
a_ee[nx-2,:] = 0
Sp[nx-1,:] += -((8/3)*Dx+(2/8)*(1-a(u))*Fx - Fx) 
Su[nx-1,:] += ((8/3)*Dx+(2/8)*(1-a(u))*Fx - Fx)   * Tright
Sp[nx-2,:] += (1-a(u))*Fx/4
Su[nx-2,:] += -(1-a(u))*Fx* Tleft/4

#bottom bc
Tbottom = 0
a_ss[:,0] = 0
a_ss[:,1] = 0
a_s[:,0] = 0
Sp[:,0] += -((8/3)*Dy+(2/8)*a(v)*Fy+Fy) 
Su[:,0] += ((8/3)*Dy+(2/8)*a(v)*Fy+Fy) * Tbottom
Sp[:,1] += a(v)*Fy/4
Su[:,1] += -a(v)*Fy* Tbottom/4

#top bc
Ttop = 100
a_n[:,nx-1] = 0
a_nn[:,nx-1] = 0
a_nn[:,nx-2] = 0
Sp[:,nx-1] += -((8/3)*Dy+(2/8)*(1-a(v))*Fy - Fy) 
Su[:,nx-1] += ((8/3)*Dy+(2/8)*(1-a(v))*Fy - Fy)   * Ttop
Sp[:,nx-2] += (1-a(v))*Fy/4
Su[:,nx-2] += -(1-a(v))*Fy* Ttop/4


a_p = a_e+a_w+a_n+a_s+a_ee+a_ww+a_nn+a_ss-Sp
""" print("aw = ", a_w)
print("ae = ", a_e)
print("as = ", a_s)
print("an = ", a_n)
print("aww = ", a_ww)
print("aee = ", a_ee)
print("ass = ", a_ss)
print("ann = ", a_nn)
print("Sp = ", Sp)

print("ap = ", a_p) """

A = np.empty(shape=(nx*ny,nx*ny))
A.fill(0)
ii=0
b=np.zeros(nx*ny)
for i in range(nx):
    for j in range(ny):
        A[ii,ii] = a_p[j,i]
        if(ii<nx*ny-1):
            A[ii,ii+1] = -a_e[j,i]
        if(ii<nx*ny-2):
            A[ii,ii+2] = -a_ee[j,i]
        if(ii<nx*ny-nx):
            A[ii,ii+nx] = -a_n[j,i]
        if(ii<nx*ny-2*nx):
            A[ii,ii+2*nx] = -a_nn[j,i]
        if(ii>0):
            A[ii,ii-1] = -a_w[j,i]
        if(ii>1):
            A[ii,ii-2] = -a_ww[j,i]
        if(ii>=nx):
            A[ii,ii-nx] = -a_s[j,i]
        if(ii>=2*nx):
            A[ii,ii-2*nx] = -a_ss[j,i]
        b[ii]=Su[j][i]
        ii+=1

#print(A)
#print(b)
for i in range(nx*ny):
    print("A[i][i] = ",A[i,i])

print(len(A))
print(len(b))
Tsim=gauss_seidel(A,b)
#Tsim1= np.dot(np.linalg.inv(A),b)

#print("Tsim = ", Tsim)


# for plotting graph
Tdiag = np.zeros(nx)
for i in range(nx):
    Tdiag[i]= Tsim[(nx-(i+1))*nx+i] #to get value along minor diagonal, bit complex though
diag=np.sqrt(length**2+width**2)

xSim = np.linspace(0,diag,nx)

plt.plot(xSim,Tdiag,linestyle='--' ,marker='o',color='b',label='Numerical Solution [upwind]')


plt.title("QUICK-2D: diagonal flow with diffusion")
plt.xlabel("Position")
plt.ylabel("Property")
plt.legend()
plt.grid()
figname="QUICK-2D"
plt.savefig(figname+".png")
plt.show()
