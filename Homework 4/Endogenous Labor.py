# PS4 FROM MARIO SERRANO GARCÍA

from scipy import optimize
import numpy as np
import math
import matplotlib.pyplot as plt
import timeit
import quantecon as qe

"""QUESTION 1: VALUE FUNCTION ITERATION

2. Redo item 1 adding a labor choice that is continuous. For this, set kapa = 5:24 and v = 2:0.

a) Solve with brute force iterations of the value function. Plot your value function."""

# The recursive proble can be written as 
# v(k) = max u(c) + Bv(k')
# v(k) = max u(f(k) + (1-d)k - k') + Bv(k')

# We set the value of the parameters

theta = 0.679
beta = 0.988
delta = 0.013
kapa = 5.24
v = 2
penalty = -100000 # A penalty in case consumption being negative

# STEP 1: We create our grid by discretizing k & h

# From the maximization problem, differenciationg wrt to k, c & h we obtain
# dL/dc = B^t*(1/c_t) + lamda_t = 0
# dL/dh = -B^t*kapa*h_t^(1/v) + lamda_t*k_t^(1-theta)*theta*h^(theta-1) = 0
# dL/dk = lamda_t*(h_t^theta*(1-theta)*k_t^(-theta) + (1-delta)) - lamda_t+1 = 0

# Imposing the SS  and combining them we obtain two expressions as a function on k,h
# 1) 1/B - (1-theta)*k^(-theta)*h^theta - (1-delta)
# 2) theta*k^(-theta)*h^(theta-1-1/v) - kapa*(k^(-theta)*h^theta - delta)

def SS(x):
    k = x[0]
    h = x[1]
    e1 = (1-theta)*h**theta*k**(-theta) + 1 - delta - 1/beta
    e2 = theta*k**(-theta)*h**(theta-1-1/v) - kapa*(k**(-theta)*h**theta - delta)
    return e1,e2

k_SS, h_SS = optimize.fsolve(SS,[1,1])

k_min = 10
k_max = 200 # We set 200 cause seems like function does not converge 
h_min = 0.15
h_max = h_SS*1.5
print('k_SS = ' + str(k_SS))
print('h_SS = ' + str(h_SS))
# We create our grid
dim = 200
k = np.linspace(k_min,k_max,num=dim,endpoint=True)
h = np.linspace(h_min,h_max,num=dim,endpoint=True)

# STEP 2: We make an intial guess about the solution of the value function. We will set V = 0

V0 = np.zeros(dim)

# STEP 3 & 4: We define a tarix that cointains the utility associated to every possible combination of k & k'. 
# The method defined deliver the utility associated to the introduced pair as input. Besides, we make sure that
# we do not get any solution for wich consumption is zero by adding a constrain

def f(k,h):
    return k**(1-theta)*h**theta

def u(c,h):
    return math.log(c) - kapa*h**(1+1/v)/(1+1/v)

def return_M(k1,k2,h):
    c = f(k1,h) + (1-delta)*k1 - k2
    if c>0:
        return u(c,h)
    else:
        return penalty

M = np.empty([dim,dim,dim])
i=0
while i<=dim-1:
    j=0
    while j<=dim-1:
        m=0
        while m<=dim-1:
            M[i,j] = return_M(k[i],h[j],k[m]) #We deffine matrix M
            m = m+1
        j = j+1
    i = i+1

# STEP 5.1: From matrix M and vector V, we deffine a matrix X that collect all possible values of V
# Step 5.2: From this matrix, we compute the updated V by choosing the maximum element of each row
# STEP 6: We create a loop to iterate the Bellman equation until it report that the distance between two consecutives
# values of V is small enough, meanning that we have reached the SS.

start = timeit.default_timer()

def Bellman_Labor(V0): #Deffine a new funtion with lees loop cause take too much time
    V1 = np.zeros(dim)
    X = np.empty([dim,dim,dim])
    for i in range(dim):
        for j in range(dim):
             X[i,j,:] =  M[i,j,:] +beta*V0
                        
        V1[i] = np.nanmax(X[i,:,:])
    return V1

V = qe.compute_fixed_point(Bellman_Labor, V0, error_tol=0.05, max_iter=500,print_skip=50) #We use a predetermined method to find the fix point
V_labor = Bellman_Labor(V)

stop = timeit.default_timer()
time_labor = stop - start
print('Time - Endogenous Labor: '+str(time_labor))

# We plot our results for the value & policy functions

plt.plot(k,V_labor,label='value function')
plt.legend()
plt.title('Endogenous Labor',size=15)
plt.xlabel('k')
plt.ylabel('v(k)')
plt.show()


