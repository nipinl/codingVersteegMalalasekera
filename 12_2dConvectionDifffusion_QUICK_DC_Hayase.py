#Upwind for convection term. CD for diffusion term
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
k=0.0

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

a_e = np.empty(shape=(nx,ny))
a_e.fill(Dx-(1-a(u))*Fx)
a_w = np.empty(shape=(nx,ny))
a_w.fill(Dx+a(u)*Fx)
a_n = np.empty(shape=(nx,ny))
a_n.fill(Dy-(1-a(v))*Fy)
a_s = np.empty(shape=(nx,ny))
a_s.fill(Dy+a(v)*Fy)

Sp = np.empty(shape=(nx,ny))
Sp.fill(0)
Su = np.empty(shape=(nx,ny))
Su.fill(0)

for i in range(2,nx-3):
    for j in range(2,ny-3):
        #x direction
        Su[i,j] += (3*T[i,j] - 2*T[i-1,j] - T[i-2,j])*a(u)*Fx/8
        Su[i,j] -= (3*T[i+1,j] - 2*T[i,j] - T[i-1,j])*a(u)*Fx/8
        Su[i,j] += (3*T[i-1,j] - 2*T[i,j] - T[i+1,j])*(1-a(u))*Fx/8
        Su[i,j] -= (3*T[i,j] - 2*T[i+1,j] - T[i+2,j])*(1-a(u))*Fx/8
        #y direction
        Su[i,j] += (3*T[i,j] - 2*T[i,j-1] - T[i,j-2])*a(v)*Fy/8
        Su[i,j] -= (3*T[i,j+1] - 2*T[i,j] - T[i,j-1])*a(v)*Fy/8
        Su[i,j] += (3*T[i,j-1] - 2*T[i,j] - T[i,j+1])*(1-a(v))*Fy/8
        Su[i,j] -= (3*T[i,j] - 2*T[i,j+1] - T[i,j+2])*(1-a(v))*Fy/8
#next to left boundary
for j in range(ny):
    Su[1,j] +=((3*T[1,j] - T[0,j]) - (3*T[2,j] - 2*T[1,j] - T[0,j]))*Fx/8
#next to bottom boundary
for i in range(nx):
    Su[i,1] += ((3*T[i,1] - T[i,0]) - (3*T[i,2] - 2*T[i,1] - T[i,0]))*Fy/8



#left bc
Tleft=100
a_w[0,:] = 0
a_e[0,:] += Dx/3
Sp[0,:] += -((8/3)*Dx+max(0,Fx)) 
Su[0,:] += ((8/3)*Dx+max(0,Fx)) * Tleft + (T[0,:] - 3*T[1,:])*Fx/8

#right bc
Tright=0
a_e[nx-1,:] = 0
Su[nx-1,:] += (3*T[nx-1,:] - 2*T[nx-2,j] - T[nx-3,j])*Fx/8


#bottom bc
Tbottom = 0
a_s[:,0] = 0
a_n[:,0] += Dy/3
Sp[:,0] += -((8/3)*Dy+max(0,Fy)) 
Su[:,0] += ((8/3)*Dy+max(0,Fy)) * Tbottom+ (T[:,0] - 3*T[:,1])*Fy/8

#top bc
Ttop = 100
a_n[:,ny-1] = 0
Su[:,ny-1] += (3*T[:,ny-1] - 2*T[j,ny-2] - T[j,ny-3])*Fy/8

a_p = a_e+a_w+a_n+a_s-Sp
#print("ap = ", a_p)
A = np.empty(shape=(nx*ny,nx*ny))
A.fill(0)
ii=0
b=np.zeros(nx*ny)
for i in range(nx):
    for j in range(ny):
        A[ii,ii] = a_p[j,i]
        if(ii<nx*ny-1):
            A[ii,ii+1] = -a_e[j,i]
        if(ii<nx*ny-nx):
            A[ii,ii+nx] = -a_n[j,i]
        if(ii>0):
            A[ii,ii-1] = -a_w[j,i]
        if(ii>=nx):
            A[ii,ii-nx] = -a_s[j,i]
        b[ii]=Su[j][i]
        ii+=1

print(A)
print(b)
Tsim=gauss_seidel(A,b)
Tsim1= np.dot(np.linalg.inv(A),b)

print("Tsim = ", Tsim)
print("Tsim1 = ", Tsim1)

# for plotting graph
Tdiag = np.zeros(nx)
for i in range(nx):
    Tdiag[i]= Tsim[(nx-(i+1))*nx+i] #to get value along minor diagonal, bit complex though
diag=np.sqrt(length**2+width**2)
Texact = np.ones(nx)*Tbottom
Texact[:int(nx/2)+1]=Tleft
x=np.linspace(0,diag,nx)
x[int(nx/2)]= diag/2-0.001*diag
x[int(nx/2)+1]= diag/2+0.001*diag
xSim = np.linspace(0,diag,nx)

plt.plot(xSim,Tdiag,linestyle='--' ,marker='o',color='b',label='Numerical Solution [upwind]')
plt.plot(x,Texact,'-g',label='Exact Solution')

plt.title("Comparison of transported property")
plt.xlabel("Position")
plt.ylabel("Property")
plt.legend()
plt.grid()
figname="Section_5.6.1"
plt.savefig(figname+".png")
plt.show()
