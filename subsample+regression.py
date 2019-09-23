
snr = 30
mu_0 = 1/1.2566e-6
epsilon_0=1/8.85e-12

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from PDE_FIND import *
import scipy.io as sio
import pylab
import random

def wgn(x, snr):
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)


pylab.rcParams['figure.figsize'] = (12, 8)
# Load data
data = sio.loadmat('C:/Users/15307/Desktop/热传导方程解集构建/initialdata.mat')
u1 = np.transpose(pylab.real(data['Ex_plasma']))
u2 = np.transpose(pylab.real(data['Hy_plasma']))
jx = np.transpose(pylab.real(data['Jx_plasma']))
#jy = pylab.real(data['Jy_plasma'])
#jz = pylab.real(data['Jz_plasma'])
x = data['x'][0]
t = data['t'][0]
for i in range(800):
    a1 = u1[i]
    a2 = u2[i]
    a3 = jx[i]
    n1 = wgn(a1,snr)
    n2 = wgn(a2,snr)
    n3 = wgn(a3,snr)
    
    u1[i] = a1 + n1
    u2[i] = a2 + n2
    jx[i] = a3 + n3
    
    
    
    
n = len(x)
steps = len(t)
dx = x[2]-x[1]
dt = t[2]-t[1]

##降采样
#
num_x = 100
num_t = 600
np.random.seed(0) # so that numbers in paper are reproducible
n = len(x)
num_points = num_x * num_t
boundary = 40
points = {}
count = 0
#U1 = np.ones((num_x,num_t))*1
#U2 = np.ones((num_x,num_t))*1
#JX = np.ones((num_x,num_t))*1
#JY = np.ones((num_x,num_t))*1
#JZ = np.ones((num_x,num_t))*1
#x = 200
x = 205
for p in range(num_x):
#    if x < 601:
        x = np.random.choice(np.arange(200,600),1)[0]
        
#        x = x+5
        for t in range(num_t):
            points[count] = [x,5*t+750]
            t = t+1
            count = count + 1
            
        

Ex = np.zeros((num_points,1))
Hy = np.zeros((num_points,1))
Ext = np.zeros((num_points,1))
Hyt = np.zeros((num_points,1))
Jx = np.zeros((num_points,1))

Ex_z = np.zeros((num_points,1))
Hy_z = np.zeros((num_points,1))
Ex_zz = np.zeros((num_points,1))
Hy_zz = np.zeros((num_points,1))
#
N = 9  # number of points to use in fitting
Nt = N
deg = 5 # degree of polynomial to use
#
#for i in range(u1.size):
#    u1[i] +=random.gauss(mu,sigma)
for p in points.keys():
     
    [x,t] = points[p]
    Ex[p] = u1[x,t]
    Hy[p] = u2[x,t]
    Jx[p] = jx[x,t]
#    Jy[p] = jy[x,t]
#    Jz[p] = jz[x,t]
  
    Ext[p] = PolyDiffPoint(u1[x,t-(Nt-1)//2:t+(Nt+1)//2], np.arange(Nt)*dt, deg, 1)[0]
    Hyt[p] = PolyDiffPoint(u2[x,t-(Nt-1)//2:t+(Nt+1)//2], np.arange(Nt)*dt, deg, 1)[0]
#    Jxt[p] = PolyDiffPoint(jx[x,t-(Nt-1)//2:t+(Nt+1)//2], np.arange(Nt-1)*dt, deg, 1)[0]
#    Jyt[p] = PolyDiffPoint(jy[x,t-(Nt-1)//2:t+(Nt+1)//2], np.arange(Nt-1)*dt, deg, 1)[0]
#    Jzt[p] = PolyDiffPoint(jz[x,t-(Nt-1)//2:t+(Nt+1)//2], np.arange(Nt-1)*dt, deg, 1)[0]
#    if x == 799:
#        break
#    else:

        
    Ex_z_diff = PolyDiffPoint(u1[x-(Nt-1)//2:x+(Nt+1)//2,t], np.arange(Nt)*dx, deg, 2)
    Hy_z_diff = PolyDiffPoint(u2[x-(Nt-1)//2:x+(Nt+1)//2,t], np.arange(Nt)*dx, deg, 2)
    
    Ex_z[p] = Ex_z_diff[0]
    Hy_z[p] = Hy_z_diff[0]
    Ex_zz[p] = Ex_z_diff[1]
    Hy_zz[p] = Hy_z_diff[1]
    
#Ext-Hy_z    
X_data = np.hstack([Jx])
X_ders = np.hstack([np.ones((num_points,1)),  Hy_z])
X_ders_descr = ['', 'Hy_{z}']
X, description = build_Theta(X_data, X_ders, X_ders_descr, 1, data_description = ['Jx'])
['1'] + description[1:]
c1 = TrainSTRidge(X,Ext,10**-5,1)
print_pde(c1, description)
    
epsilon=1/c1[1]
error1 = (abs(c1[0]/epsilon_0) + abs((abs(c1[1])-epsilon_0)/epsilon_0))*50
print('epsilon=',epsilon)
print('error1=',error1)

    
    
    
    
#Hyt-Ex_z
X_data = np.hstack([Ex])
X_ders = np.hstack([np.ones((num_points,1)),  Ex_z])
X_ders_descr = ['', 'Ex_{z}']
X, description = build_Theta(X_data, X_ders, X_ders_descr, 2, data_description = ['Ex'])
['1'] + description[1:]
c2 = TrainSTRidge(X,Hyt,10,10)
print_pde(c2, description)
mu=1/c2[1]
error = (abs(c2[0]/mu_0) + abs((c2[1]+mu_0)/mu_0))*50
print('mu=',mu)
print('error2=',error)
print('error=',(error1+error)/2)
###绘图
#X, T = np.meshgrid(x, t)
#print(data)
#fig = pylab.figure()
#ax = fig.gca(projection='3d')
#surf = ax.plot_surface(X, T, U.T, rstride=1, cstride=1, cmap=pylab.cm.coolwarm,
#    linewidth=0, antialiased=False)
#pylab.title('Two soliton Solution to KdV', fontsize = 20)
#pylab.xlabel('x', fontsize = 16)
#pylab.ylabel('t', fontsize = 16)

#建立微分系统
#Ut, R, rhs_des = build_linear_system(u1, dt, dx, D=3, P=2, time_diff = 'FD', space_diff = 'FD')
#['1']+rhs_des[1:]
#
## Solve with STRidge using 2-norm normalization
#w = TrainSTRidge(R,Ut,10**-5,5)
#print("PDE derived using STRidge")
#print_pde(w, rhs_des)

#err = abs(np.array([(6 - 5.956504)*100/6, (1 - 0.988106)*100]))
#print(mean(err))
#print(std(err))
#
#np.random.seed(0)
#Un = U + 0.01*std(U)*np.random.randn(n,m)
#
#Utn, Rn, rhs_des = build_linear_system(Un, dt, dx, D=3, P=2,
#                                 time_diff = 'poly', space_diff = 'poly',
#                                 width_x = 20, width_t = 10, deg_x = 5)
#
## Solve with STRidge using 2-norm normalization
#wn = TrainSTRidge(Rn,Utn,10**-5,5)
#print("PDE derived using STRidge")
#print_pde(wn, rhs_des)
#
#err = abs(np.array([(6 - 6.152522)*100/6, (1 - 1.124033)*100]))
#print(mean(err))
#print(std(err))
















