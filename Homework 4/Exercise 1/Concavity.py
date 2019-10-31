# PS4 FROM MARIO SERRANO GARCÍA

from scipy import optimize
import numpy as np
import math
import matplotlib.pyplot as plt
import timeit

"""QUESTION 1: VALUE FUNCTION ITERATION

1. Pose the recursive formulation of the sequential problem without productivity shocks. Discretize
the state space and the value function and solve for it under the computational variants
listed below. In all these variants use the same initial guess for your value function.

c) Iterations of the value function taking into account concavity of the value function."""

# The recursive proble can be written as 
# v(k) = max u(c) + Bv(k')
# v(k) = max u(f(k) + (1-d)k - k') + Bv(k')

# We set the value of the parameters

theta = 0.679
beta = 0.988
delta = 0.013
h = 1
kapa = 0
v = 2
penalty = -100000 # A penalty in case consumption being negative

# STEP 1: We create our grid by discretizing k. We chose the k_min to be close to 0 but not equal, to avoid violate
# non-zero constrain of consumption and capital. WE will set k_max sligthly above to the SS, for which reason we are 
# going to compute ir as well.

# We now that Euler equation for Neoclassical growth model is
# 1/beta = (1-theta)*(h*z)^theta*k^(-theta) + 1 - d
def SS(k):
    SS = (1-theta)*h**theta*k**(-theta) + 1 - delta - 1/beta
    return SS

k_min = 1
k_SS = optimize.fsolve(SS,1)
k_max = k_SS[0] + 1
print('k_SS = ' + str(k_SS[0]))
# We create our grid
dim = 200
k = np.linspace(k_min,k_max,num=dim)

# STEP 2: We make an intial guess about the solution of the value function. We will set V = 0

V0 = np.zeros(dim)

# STEP 3 & 4: We define a tarix that cointains the utility associated to every possible combination of k & k'. 
# The method defined deliver the utility associated to the introduced pair as input. Besides, we make sure that
# we do not get any solution for wich consumption is zero by adding a constrain

def f(k,h):
    return k**(1-theta)*h**theta

def u(c,h):
    return math.log(c) - kapa*h**(1+1/v)/(1+1/v)

def return_M(k1,k2):
    c = f(k1,h) + (1-delta)*k1 - k2
    if c>0:
        return u(c,h)
    else:
        return penalty

M = np.empty([dim,dim])
i=0
while i<=dim-1:
    j=0
    while j<=dim-1:
        M[i,j] = return_M(k[i],k[j]) #We deffine matrix M
        j = j+1
    i = i+1

# STEP 5.1: From matrix M and vector V, we deffine a matrix X that collect all possible values of V
# Step 5.2: From this matrix, we compute the updated V by choosing the maximum element of each row
# STEP 6: We create a loop to iterate the Bellman equation until it report that the distance between two consecutives
# values of V is small enough, meanning that we have reached the SS.

start = timeit.default_timer()
 
def Bellman_Con(M,V0):
   X = np.empty([dim,dim])
   X[0,0] = M[0,0] + beta*V0[0]
   X[0,1] = M[0,1] + beta*V0[1]
   i=0
   while i<dim:
        j=0
        while X[i,j] <= X[i,j+1] and j<dim-2:
            X[i,j+2] = M[i,j+2] + beta*V0[j+2] #We deffine matrix X
            j += 1
        i = i+1 
   argmax_i = np.argmax(X,axis=1) #We deffine a vector that deliver the maximun of each row
    
   V1 = np.empty(dim)
   i=0
   while i<dim:
       V1[i] = max(X[i,:]) #We take into account only those values of kj for which kj>=g(ki)
       i = i+1
   return V1, argmax_i

def solution(B):
    e = 0.05
    V_s1, argmax_i = B(M,V0)
    diff = abs(V_s1 - V0)
    while max(diff) > e: #We deffine the loop that will deliver the solution V
        V_s = V_s1
        V_s1, argmax_i = B(M,V_s)
        diff = abs(V_s1 - V_s)
    g_s = np.empty(dim)
    for i in range(dim):
        g_s[i] = k[int(argmax_i[i])] #We deffine the policy function vector
    return V_s1, g_s

V_con, g_con = solution(Bellman_Con)

stop = timeit.default_timer()
time_con = stop - start
print('Time - Concavity: '+ str(time_con))

# We plot our results for the value & policy functions

plt.plot(k,V_con)
plt.title('Value Function',size=15)
plt.xlabel('k')
plt.ylabel('v(k)')
plt.show()

plt.plot(k,g_con,label='g(k)')
plt.plot(k,k,color='red',label='45º line')
plt.legend()
plt.title('Policy Function')
plt.xlabel('k_t')
plt.ylabel('k_t+1')
plt.show()

