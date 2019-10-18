import sympy as sym
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt

"""QUESTION 1: COMPUTING TRANSITIONS IN A REPRESENTATIVE AGENT ECONOMY

a) Compute the steady-state. Choose z to match an annual capital-output ratio of 4, and an
investment-output ratio of .25."""

# Choosing starting values for the parameters. We normalize y = 4 and set  k = 4 according to the output-capital ratio
# We set the depreciation equal to 1/4 according to the capital-investment ratio.

theta = 0.67
y = 1
k_SS1 = 4
h = 0.31
i = 0.25

# Defining a function depending on zeta from the technology production function

def zeta(z):    
    f = k_SS1**(1-theta)*(h*z)**theta - y
    return f

# Computing the root of the function with the fsolve method to n¡solve nonlinear equations with a initial guess of z=1

z0 = optimize.fsolve(zeta,1)
print('z1 = ' + str(z0[0]))

#Solving the system, we get the Euler equation. Setting t = t+1 in all variables we get to an expression of the SS:
# SS: 1/beta = (1-theta)*(h*z)^theta*k^(-theta) + 1 - d

# We compute delta, appliying condition of SS(k_t = k_t+1) in the equation of investment. We get 
# i = delta*k so

delta = i/k_SS1
print('Depreciation: ' + str(delta))

# With this expression we compute beta

beta = 1/((1-theta)*(h*z0)**theta*k_SS1**(-theta) + 1 - delta)
print('Beta: ' + str(beta[0]))

"""b) Double permanently the productiviy parameter z and solve for the new steady state."""

# Defining the new value for the productivity parameter z
    
z = z0[0]*2
print('Productivity parameter z: ' + str(z))

# We define the Steady State equation

def SS(k):
    SS = (1-theta)*(z*h)**theta*k**(-theta) + 1 - delta - 1/beta
    return SS

# We solve this non-linear system for k0 =1 as first guess, and return the result and the number of iterations

k_SS_2 = optimize.fsolve(SS,4)
k_SS2 = k_SS_2[0]
print('Steady State value of k: ' + str(k_SS2))

"""c) Compute the transition from the first to the second steady state and report the time-path
for savings, consumption, labor and output."""

# From the maximization problem, we define the Euler equation, which  has the following form
# u'(c_t) = u'(c_t+1)*f'(k_t+1) --> u'(f(k)-k_t+1) = u'(f(k_t+1) - k_t+2)*f'(k_t+1)
# u'(f(k) - k_t+1) = B^t/(k^(1-theta)*(z*h)^theta + (1-d)k - k_t+1)
# u'(f(k_t+1) - k_t+2) = B^t+1/(k_t+1^(1-theta)*(z*h)^theta + (1-d)k_t+1 - k_t+2)
#  
# And taking all of it to one side, we can obtain an expresion for k_t+1 depending on k_t

def Euler(k,k_t1,k_t2):
    return k_t1**(1-theta)*(z*h)**theta + (1-delta)*k_t1 - k_t2 - beta*((1-theta)*(z*h)**theta*k_t1**(-theta) + 1 - delta)*(k**(1-theta)*(z*h)**(theta) + (1 - delta)*k - k_t1)

# We define a function to compute the entire path of capital 

def transition(u): 
    T = np.zeros(200)
    T[0] = Euler(k_SS1,u[1],u[2])
    u[199] = k_SS2
    T[198] = Euler(u[197], u[198], u[199])
    for i in range(1,198):
        T[i] = Euler(u[i],u[i+1],u[i+2])
    u[0] = k_SS1
    return T

u = np.ones(200)*4
k_path = optimize.fsolve(transition, u)

# We deffine domains of our variables of interest

t = np.linspace(0,100,100)
k_t = k_path[0:100]
k_t1 = k_path[1:101]

# We deffine our functions for savings, output and consumption

y_t = k_t**(1-theta)*(z*h)**theta
s_t = k_t1 - (1-delta)*k_t
c_t = y_t - s_t
 
# We plot a graphic for the transition

plt.plot(t,k_t)
plt.title('Transition path from k = 4 to k = 8')
plt.xlabel('t')
plt.ylabel('k(t)')
plt.show()

# We plot graphics for paths of savings, output and consumption

plt.plot(t,y_t,label='output path')
plt.plot(t,s_t,label='savings path')
plt.plot(t,c_t,label='consumption path')
plt.title('Transition Paths')
plt.xlabel('t')
plt.legend()
plt.show()

"""d) Unexpected shocks. Let the agents believe productivity zt doubles once and for all periods.
However, after 10 periods, surprise the economy by cutting the productivity zt back to its
original value. Compute the transition for savings, consumption, labor and output."""

# We know when z=z0 the economy reach the SS for k=4, and when z=2*z0 this occurs for k=8. In this 
# situation, the economy will start its path from SS1 to SS2, but suddenly, in period 10, the shock occurs
# and the situation is reversed and the economy will return to its initial point in SS1

# The break point will occur in period 11, therefore the last period in wich the economy is in the first path
# will be t = 10, so we get the value of the economy at this point and use it to compute the new path as its
# starting point

k_SS2 = 4 #Starting pont for the first path
k_SS1 = k_path[9] #Starting point for the second path
print('Break point: ' + str(k_SS1))
z = z0[0] #New value of z for the second path

k_path2 = optimize.fsolve(transition, u)

# We deffine the new domain and plot the new transition path

k_newPath = np.append(k_path[0:10],k_path2[0:90])

plt.plot(t,k_newPath)
plt.title('Transition path with an Unexpected shock')
plt.xlabel('t')
plt.ylabel('k(t)')
plt.show()

# Plotting new transition paths for output, savings and consumption

y_t = k_newPath**(1-theta)*(z*h)**theta
s_t = k_newPath - (1-delta)*k_newPath
c_t = y_t - s_t

plt.plot(t,y_t,label='Output path')
plt.plot(t,s_t,label='Savings path')
plt.plot(t,c_t,label='Consumption path')
plt.title('New Transition Paths')
plt.legend()
plt.xlabel('t')
plt.show()


"""QUESTION 2: MULTICOUNTRY MODEL WITH MOVILITY OF CAPITAL AND PROGRESSIVE LABOR INCOME TAX

2.1 Consider the case that each of these two countries are closed economies. Write: 
(a) the equilibirum of a closed economy, (b) an algorithm to solve it and (c) solve the economy."""

# We set the value of the parameters
# Common parameters
kapa = 5
v = 1
sigma =0.8
Z = 1
theta = 0.6
k_max = 2
phi = 0.2
# Country A parameters
eta_lowA = 0.5
eta_highA = 5.5
lamdaA = 0.95
#_ Country B parameters
eta_lowB = 2.5
eta_highB = 3.5
lamdaB = 0.84

# We set the maximization problem for the firm and solve prices as functions of K,H

K = sym.symbols('K')
H = sym.symbols('H')
production = Z*K**(1-theta)*H**(theta)

# But since K,H are only aggregation of their respectives ki and hi. We assume without loss of generality K1 = K2 = 1
ki = 1
k_low = sym.symbols('k_low')
k_high = sym.symbols('k_high')
h_low = sym.symbols('h_low')
h_high = sym.symbols('h_high')

def prices(f,k1,k2,h1,h2):
    d_K = sym.diff(f,K)
    d_H = sym.diff(f,H)
    r = d_K.subs([(K,(k1+k2)),(H,(h1+h2))])
    w = d_H.subs([(K,(k1+k2)),(H,(h1+h2))])
    return r, w

r_aut, w_aut = prices(production,ki,ki,h_low,h_high)

# We set the maximization problem of the HH
# We set the Lagrangian and compute its derivatives wrt the decision variables
c = sym.symbols('c')
k = sym.symbols('k')
h = sym.symbols('h')
mu = sym.symbols('mu')
lamda = sym.symbols('lamda')
w = sym.symbols('w')
eta = sym.symbols('eta')
r1 = sym.symbols('r1')
r2 = sym.symbols('r2')

L = c**(1-sigma)/(1-sigma) - kapa*h**(1+1/v)/(1+1/v) + mu*(lamda*(w*h*eta)**(1-theta) + r1*k + r2*(k_max-k) - c)

dL_c = sym.diff(L,c)
dL_h = sym.diff(L,h)
dL_k = sym.diff(L,k)
print('dL/dc = ' + str(dL_c) + ' = 0')
print('dL/dh = ' + str(dL_h) + ' = 0')
print('dL/dk = ' + str(dL_k) + ' = 0')
# Combining these two we obtain the Euler equation
def gen_Euler(c1,h1,eta1,lamda1):
    EE = dL_h.subs(mu,c**(-sigma))
    EE1 = EE.subs([(c,c1),(h,h1),(eta,eta1),(lamda,lamda1)])
    return EE1

# Now, with the Euler and the budget constrain we construct a system of two equations. 
    
budget_const = lamda*(w*h*eta)**(1-theta) + r1*k + r2*(k_max-k) - c

def gen_BC(c1,h1,k1,eta1,lamda1):
    BC1 = budget_const.subs([(c,c1),(h,h1),(eta,eta1),(lamda,lamda1)])
    return BC1

# We compute equilibrium for every situation. We only have to take into account that, in this 
# case, there is not foreign investment, so r2*(k_max-k) = 0
c_low = sym.symbols('c_low')
c_high = sym.symbols('c_high')
#Country A
Euler_lowA = gen_Euler(c_low,h_low,eta_lowA,lamdaA).subs(w,w_aut)
Euler_highA =gen_Euler(c_high,h_high,eta_highA,lamdaA).subs(w,w_aut)
BC_lowA = gen_BC(c_low,h_low,1,eta_lowA,lamdaA).subs([(w,w_aut),(r2,0),(k,ki),(r1,r_aut)])
BC_highA = gen_BC(c_high,h_high,1,eta_highA,lamdaA).subs([(w,w_aut),(r2,0),(k,ki),(r1,r_aut)])
#Country B
Euler_lowB = gen_Euler(c_low,h_low,eta_lowB,lamdaB).subs(w,w_aut)
Euler_highB = gen_Euler(c_high,h_high,eta_highB,lamdaB).subs(w,w_aut)
BC_lowB = gen_BC(c_low,h_low,1,eta_lowB,lamdaB).subs([(w,w_aut),(r2,0),(k,ki),(r1,r_aut)])
BC_highB = gen_BC(c_high,h_high,1,eta_highB,lamdaB).subs([(w,w_aut),(r2,0),(k,ki),(r1,r_aut)])
# x[0]=c_low, x[1]=c_high, x[2]=h_low, x[3]=h_high
def eq_A(x): 
    EE_lowA = Euler_lowA.subs([(c_low,x[0]),(h_low,x[2]),(h_high,x[3])])
    EE_highA = Euler_highA.subs([(c_high,x[1]),(h_low,x[2]),(h_high,x[3])])
    Budget_lowA = BC_lowA.subs([(c_low,x[0]),(h_low,x[2]),(h_high,x[3])])
    Budget_highA = BC_highA.subs([(c_high,x[1]),(h_low,x[2]),(h_high,x[3])])
    return EE_lowA, EE_highA, Budget_lowA, Budget_highA

x0 = [1,1,1,1]
eqA = optimize.fsolve(eq_A,x0)

def eq_B(x): 
    EE_lowB = Euler_lowB.subs([(c_low,x[0]),(h_low,x[2]),(h_high,x[3])])
    EE_highB = Euler_highB.subs([(c_high,x[1]),(h_low,x[2]),(h_high,x[3])])
    Budget_lowB = BC_lowB.subs([(c_low,x[0]),(h_low,x[2]),(h_high,x[3])])
    Budget_highB = BC_highB.subs([(c_high,x[1]),(h_low,x[2]),(h_high,x[3])])
    return EE_lowB, EE_highB, Budget_lowB, Budget_highB

eqB = optimize.fsolve(eq_B,x0)
# Prices
w_autA = w_aut.subs([(h_low,eqA[2]),(h_high,eqA[3])])
w_autB = w_aut.subs([(h_low,eqB[2]),(h_high,eqB[3])])
r_autA = r_aut.subs([(h_low,eqA[2]),(h_high,eqA[3])])
r_autB = r_aut.subs([(h_low,eqB[2]),(h_high,eqB[3])])

# We print the results

print('---------------------------------------------')
print('AUTARKY EQUILIBRIA OF COUNTRY A:')
print('w = ' + str(w_autA))
print('r = ' + str(r_autA))
print('Consumption of Low Type: ' + str(eqA[0]))
print('Consumption of High Type: ' + str(eqA[1]))
print('Labor Supply of Low Type: ' + str(eqA[2]))
print('Labor Supply of High Type: ' + str(eqA[3]))
print('---------------------------------------------')
print('AUTARKY EQUILIBRIA OF COUNTRY B:')
print('w = ' + str(w_autB))
print('r = ' + str(r_autB))
print('Consumption of Low Type: ' + str(eqB[0]))
print('Consumption of High Type: ' + str(eqB[1]))
print('Labor Supply of Low Type: ' + str(eqB[2]))
print('Labor Supply of High Type: ' + str(eqB[3]))

""" 2.2. (a) Write the equilibirum of the union economy, (b) the algorithm to solve it and (3) solve
the economy for a given set of parameters. Notice that in the union economy market clearing is
given by union capital markets and country-speci
c labor markets:"""

# Now we have to take into account also capital movements between countries. We define a one single function 
# collecting Euler equations, budget constrains & market clearing conditions for borh economies.

def eq_multicountry(x):
    # x[0]=h_lowA, x[1]=h_highA, x[2]=c_lowA, x[3]=c_highA, x[4]=k_lowA, x[5]=k_highA , x[6]=w_A, x[7]=r_A
    e1 = -kapa*x[0]**(1/v)+x[2]**(-sigma)*lamdaA*(1-phi)*(x[6]*eta_lowA)*(x[6]*x[0]*eta_lowA)**(-phi)
    e2 = -kapa*x[1]**(1/v)+x[3]**(-sigma)*lamdaA*(1-phi)*(x[6]*eta_highA)*(x[6]*x[1]*eta_highA)**(-phi)
    e3 = -x[7]+Z*(1-theta)*(x[4]+x[5]+(ki-x[12])+(ki-x[13]))**(-theta)*(x[0]*eta_lowA + x[1]*eta_highA)**theta
    e4 = -x[6]+Z*theta*(x[4]+x[5]+(ki-x[12])+(ki-x[13]))**(1-theta)*(x[0]*eta_lowA + x[1]*eta_highA)**(theta-1)
    e5 = -x[15]+x[7]*eta_lowA*x[4]**(eta_lowA-1)
    e6 = -x[15]+x[7]*eta_highA*x[5]**(eta_highA-1)
    e7 = -x[2]+lamdaA*(x[6]*x[0]*eta_lowA)**(1-phi)+x[7]*x[4]**eta_lowA+x[15]*(ki-x[4])
    e8 = -x[3]+lamdaA*(x[6]*x[1]*eta_highA)**(1-phi)+x[7]*x[5]**eta_highA+x[15]*(ki-x[5])
    # x[8]=h_lowB, x[9]=h_highB, x[10]=c_lowB, x[11]=c_highB, x[12]=k_lowB, x[13]=k_highB , x[14]=w_B, x[15]=r_B
    e9 = -kapa*x[8]**(1/v)+x[10]**(-sigma)*lamdaB*(1-phi)*(x[14]*eta_lowB)*(x[14]*x[8]*eta_lowB)**(-phi)
    e10 = -kapa*x[9]**(1/v)+x[11]**(-sigma)*lamdaB*(1-phi)*(x[14]*eta_highB)*(x[14]*x[9]*eta_highB)**(-phi)
    e11 = -x[15]+Z*(1-theta)*(x[12]+x[13]+(ki-x[4])+(ki-x[5]))**(-theta)*(x[8]*eta_lowB + x[9]*eta_highB)**theta
    e12 = -x[14]+Z*theta*(x[12]+x[13]+(ki-x[4])+(ki-x[5]))**(1-theta)*(x[8]*eta_lowB + x[9]*eta_highB)**(theta-1)
    e13 = -x[7]+x[15]*eta_lowB*x[12]**(eta_lowB-1)
    e14 = -x[7]+x[15]*eta_highB*x[13]**(eta_highB-1)
    e15 = -x[10]+lamdaB*(x[14]*x[8]*eta_lowB)**(1-phi)+x[15]*x[12]**eta_lowB+x[7]*(ki-x[12])
    e16 = -x[11]+lamdaB*(x[14]*x[9]*eta_highB)**(1-phi)+x[15]*x[13]**eta_highB+x[7]*(ki-x[13])
    
    return e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12,e13,e14,e15,e16

eq_union = optimize.fsolve(eq_multicountry, [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])

# We print the results

print('---------------------------------------------')
print('UNION ECONOMY EQUILIBRIA')
print('---------------------------------------------')
print('EQUILIBRIA IN COUNTRY A:')
print('w = ' + str(eq_union[6]))
print('r = ' + str(eq_union[7]))
print('Consumption of Low Type: ' + str(eq_union[2]))
print('Consumption of High Type: ' + str(eq_union[3]))
print('Labor Supply of Low Type: ' + str(eq_union[0]))
print('Labor Supply of High Type: ' + str(eq_union[1]))
print('Capital Supply of Low Type: ' + str(eq_union[4]))
print('Capital Supply of High Type: ' + str(eq_union[5]))
print('---------------------------------------------')
print('EQUILIBRIA IN COUNTRY B:')
print('w = ' + str(eq_union[14]))
print('r = ' + str(eq_union[15]))
print('Consumption of Low Type: ' + str(eq_union[10]))
print('Consumption of High Type: ' + str(eq_union[11]))
print('Labor Supply of Low Type: ' + str(eq_union[8]))
print('Labor Supply of High Type: ' + str(eq_union[9]))
print('Capital Supply of Low Type: ' + str(eq_union[12]))
print('Capital Supply of High Type: ' + str(eq_union[13]))




