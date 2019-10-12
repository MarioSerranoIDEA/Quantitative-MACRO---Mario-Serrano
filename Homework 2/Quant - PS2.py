# PS2 FROM MARIO SERRANO GARCÍA

"""QUESTION 1: FUNCTION APPROXIMATION, UNIVARIATE

EXERCISE 1: Approximate f(x) = x^321 with a Taylor series around x = 1. Compare your approximation
over the domain (0,4). Compare when you use up to 1, 2, 5 and 20 order approximations.Discuss your results."""

#We import the modules that we will need to use some methods

import math
import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative
from mpl_toolkits.mplot3d import Axes3D

#We define our objective function f(x) with one argument

x = sym.symbols('x')
obj_function = x**(.321)

# We define a method that makes the Taylor expansion of n order to any input funtion f, around the point x0

def Taylor_expan(f,n,x0):
    i=0
    t=0
    while i<=n:
        t += (f.diff(x,i).subs(x,x0)*(x-x0)**i)/math.factorial(i)
        i += 1
    return t

# We calculate the Taylor expansion in orders 1,2,5 and 20 
    
t1 = Taylor_expan(obj_function,1,1)
t2 = Taylor_expan(obj_function,2,1)
t5 = Taylor_expan(obj_function,5,1)
t20 = Taylor_expan(obj_function,20,1)
print(t1)
print(t2)
print(t5)
print(t20)

# We define the domain for our function

x = np.linspace(0,4,50)
plt.xlim(0,4)
plt.ylim(-2,4)

# We set the values for the approximations

t1 = 0.321*x + 0.679
t2 = 0.321*x - 0.1089795*(x - 1)**2 + 0.679
t5 = 0.321*x + 0.0300570779907967*(x - 1)**5 - 0.040849521596625*(x - 1)**4 + 0.0609921935*(x - 1)**3 - 0.1089795*(x - 1)**2 + 0.679
t20 = 0.321*x - 0.00465389246518441*(x - 1)**20 + 0.00498302100239243*(x - 1)**19 - 0.00535535941204005*(x - 1)**18 + 0.00577951132662155*(x - 1)**17 - 0.00626645146709397*(x - 1)**16 + 0.00683038514023459*(x - 1)**15 - 0.00749000490558658*(x - 1)**14 + 0.0082703737422677*(x - 1)**13 - 0.00920582743809231*(x - 1)**12 + 0.0103445949299661*(x - 1)**11 - 0.0117564360191783*(x - 1)**10 + 0.0135458417089277*(x - 1)**9 - 0.0158761004532294*(x - 1)**8 + 0.0190161406836106*(x - 1)**7 - 0.0234395113198229*(x - 1)**6 + 0.0300570779907967*(x - 1)**5 - 0.040849521596625*(x - 1)**4 + 0.0609921935*(x - 1)**3 - 0.1089795*(x - 1)**2 + 0.679
obj_function = x**(.321)

# We perform a graphic that sows all the approximations and the real function in order to see how accurate they are

plt.plot(x,t1,label='t1')
plt.plot(x,t2,label='t2')
plt.plot(x,t5,label='t3')
plt.plot(x,t20,label='t4')
plt.plot(x,obj_function,label='x^3')
plt.title('Taylor expansion of x^3 at x=1')
plt.xlabel('x')
plt.ylabel('t')
plt.legend()
plt.show()

"""EXERCISE 2: Approximate the ramp function f(x) = (x+|x|)/2 with a Taylor series around x = 2. Compare your 
approximation over the domain (-2,6). Compare when you use up to 1, 2, 5 and 20 order approximations. Discuss 
your results."""

# We define our objective function and our variable. We define an intermediate variable to avoid changes in its value

x = sym.symbols('x')
obj_function = (x + abs(x))/2
# From -inf to 0 obj_function = 0
# From 0 to inf obj_function = x --> we aproximate the function with this shape
obj_f = x

# We calculate the Taylor expansion in orders 1,2,5 and 20

t1_1 = Taylor_expan(obj_f,1,2)
t2_1 = Taylor_expan(obj_f,2,2)
t5_1 = Taylor_expan(obj_f,5,2)
t20_1 = Taylor_expan(obj_f,20,2)

print(t1_1)
print(t2_1)
print(t5_1)
print(t20_1)

# We define the domain for our function

x = np.linspace(-2,6,100)
plt.xlim(-2,6)
plt.ylim(-4,4)

# We set the values for the approximations and for the function

t1_1 = x
t2_1 = x
t5_1 = x
t20_1 = x
obj_function = (x + abs(x))/2

# We perform our graphic

plt.plot(x,t1_1,label='t1,t2,t5,t20')
plt.plot(x,obj_function,label='x^3')
plt.title('Taylor expansion of (x+|x|)/2 at x=2')
plt.xlabel('x')
plt.ylabel('t')
plt.legend()
plt.show()

"""EXERCISE 3: Approximate these three functions: e^(1/x) , the runge function 1/(1+25x^2) , and the ramp function
f(x) = (x+|x|)/2 for the domain x ∈ [−1, 1] with

3.1 Evenly spaced interpolation nodes and a cubic polynomial. Redo with monomials of
order 5 and 10. Plot the exact function and the three approximations in the same
graph. Provide an additional plot that reports the errors as the distance between the
exact function and the approximand."""

#%%EXPONENTIAL FUNCTION
# We define the function on the interval [-1,1], We set the number of nodes at 20.

x = np.linspace(-1,1,20)
def exp_fnc(x):
    return np.exp(1/x)

# We copute the parameters for the monomial form by minimizing the error: f(x)=01+02*x+03*x^2+... for orders 3,5 and 10

par_mon3 = np.polyfit(x,exp_fnc(x),3)
par_mon5 = np.polyfit(x,exp_fnc(x),5)
par_mon10 = np.polyfit(x,exp_fnc(x),10)

# We compute the value of the polinomials at the solved value of the parameters

mon3 = np.polyval(par_mon3,x)
mon5 = np.polyval(par_mon5,x)
mon10 = np.polyval(par_mon10,x)

# We perform our graphic with the real function and the approximantions

plt.xlim(-1,1)

plt.plot(x,exp_fnc(x),label='Exponential function')
plt.plot(x,mon3,label='Mon_approximation, order 3')
plt.plot(x,mon5,label='Mon_approximation, order 5')
plt.plot(x,mon10,label='Mon_approximation, order 10')
plt.title('Exponential function - Monomial Interpolation with Evenly Separated Nodes')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(loc ='upper left',fontsize=8)
plt.show()

# We compute errors as the difference with the original function

error3 = abs(exp_fnc(x) - mon3)
error5 = abs(exp_fnc(x) - mon5)
error10 = abs(exp_fnc(x) - mon10)

# We perform the errors graphic

plt.xlim(-1,1)

plt.plot(x,error3,label='Error of order 3')
plt.plot(x,error5,label='Error of order 5')
plt.plot(x,error10,label='Error of order 10')
plt.title('Exponential Error Function - Monomial Interpolation with Evenly Separated Nodes')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(loc ='upper left',fontsize=8)
plt.show()

#%% RUNGE FUNCTION
# We define the function 

def runge_fnc(x):
    return 1/(1+25*x**2)

# We copute the parameters for the monomial form by minimizing the error: f(x)=01+02*x+03*x^2+... for orders 3,5 and 10

par_mon3_1 = np.polyfit(x,runge_fnc(x),3)
par_mon5_1 = np.polyfit(x,runge_fnc(x),5)
par_mon10_1 = np.polyfit(x,runge_fnc(x),10)

# We compute the value of the polinomials at the solved value of the parameters

mon3_1 = np.polyval(par_mon3_1,x)
mon5_1 = np.polyval(par_mon5_1,x)
mon10_1 = np.polyval(par_mon10_1,x)

# We perform our graphic with the real function and the approximantions

plt.xlim(-1,1)

plt.plot(x,runge_fnc(x),label='Runge function')
plt.plot(x,mon3_1,label='Mon_approximation, order 3')
plt.plot(x,mon5_1,label='Mon_approximation, order 5')
plt.plot(x,mon10_1,label='Mon_approximation, order 10')
plt.title('Runge Function - Monomial Interpolation with Evenly Spaced Nodes')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(loc ='upper left',fontsize=8)
plt.show()

# We compute errors as the difference with the original function

error3_1 = abs(runge_fnc(x) - mon3_1)
error5_1 = abs(runge_fnc(x) - mon5_1)
error10_1 = abs(runge_fnc(x) - mon10_1)

# We perform the errors graphic

plt.xlim(-1,1)

plt.plot(x,error3_1,label='Error of order 3')
plt.plot(x,error5_1,label='Error of order 5')
plt.plot(x,error10_1,label='Error of order 10')
plt.title('Runge Error Function - Monomial Interpolation with Evenly Separated Nodes')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(loc ='upper left',fontsize=8)
plt.show()

#%% RAMP FUNCTION
# We define the function 

def ramp_fnc(x):
    return (x+abs(x))/2

# We copute the parameters for the monomial form by minimizing the error: f(x)=01+02*x+03*x^2+... for orders 3,5 and 10

par_mon3_2 = np.polyfit(x,ramp_fnc(x),3)
par_mon5_2 = np.polyfit(x,ramp_fnc(x),5)
par_mon10_2 = np.polyfit(x,ramp_fnc(x),10)

# We compute the value of the polinomials at the solved value of the parameters

mon3_2 = np.polyval(par_mon3_2,x)
mon5_2 = np.polyval(par_mon5_2,x)
mon10_2 = np.polyval(par_mon10_2,x)

# We perform our graphic with the real function and the approximantions

plt.xlim(-1,1)

plt.plot(x,ramp_fnc(x),label='Ramp function')
plt.plot(x,mon3_2,label='Mon_approximation, order 3')
plt.plot(x,mon5_2,label='Mon_approximation, order 5')
plt.plot(x,mon10_2,label='Mon_approximation, order 10')
plt.title('Ramp Function - Monomial Interpolation with Evenly Spaced Nodes')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(loc ='upper left',fontsize=8)
plt.show()

# We compute errors as the difference with the original function

error3_2 = abs(ramp_fnc(x) - mon3_2)
error5_2 = abs(ramp_fnc(x) - mon5_2)
error10_2 = abs(ramp_fnc(x) - mon10_2)

# We perform the errors graphic

plt.xlim(-1,1)

plt.plot(x,error3_2,label='Error of order 3')
plt.plot(x,error5_2,label='Error of order 5')
plt.plot(x,error10_2,label='Error of order 10')
plt.title('Ramp Error Function - Monomial Interpolation with Evenly Separated Nodes')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(loc ='upper left',fontsize=8)
plt.show()

"""3.2 Chebyshev interpolation nodes and a cubic polynomial. Redo with monomials of order
5 and 10. Plot the exact function and the three approximations in the same graph.
Provide an additional plot that reports the errors as the distance between the exact
function and the approximand."""

# We define the Chebyshev nodes. We set the number of nodes at 30.

def ch_nodes(n):
    j = 0
    ch_grid = []
    while j<=n+1:
        ch_root = math.cos(((2*j-1)/(2*n))*math.pi)*(-1)
        ch_grid.append(ch_root)
        j += 1
    return ch_grid

cheb_nodes = ch_nodes(28)
cheb_nodes_np = np.asarray(cheb_nodes)
print(cheb_nodes)
#%%EXPONENTIAL FUNCTION
# We call the function at these points

exp_cheb = exp_fnc(cheb_nodes_np)

# We copute the parameters for the monomial form by minimizing the error: f(x)=01+02*x+03*x^2+... for orders 3,5 and 10

par_cheb3 = np.polyfit(cheb_nodes,exp_cheb,3)
par_cheb5 = np.polyfit(cheb_nodes,exp_cheb,5)
par_cheb10 = np.polyfit(cheb_nodes,exp_cheb,10) 

# We compute the value of the polinomials at the solved value of the parameters

cheb3 = np.polyval(par_cheb3,cheb_nodes)
cheb5 = np.polyval(par_cheb5,cheb_nodes)
cheb10 = np.polyval(par_cheb10,cheb_nodes)

# We perform our graphic with the real function and the approximantions

plt.plot(cheb_nodes,exp_cheb,label='Exponential function')
plt.plot(cheb_nodes,cheb3,label='Cheb_approximation, order 3')
plt.plot(cheb_nodes,cheb5,label='Cheb_approximation, order 5')
plt.plot(cheb_nodes,cheb10,label='Cheb_approximation, order 10')
plt.title('Exponential function - Monomial Interpolation with Chebishev Nodes')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(loc ='upper left',fontsize=8)
plt.show()

# We compute errors as the difference with the original function

error3_3 = abs(exp_fnc(cheb_nodes_np) - cheb3)
error5_3 = abs(exp_fnc(cheb_nodes_np) - cheb5)
error10_3 = abs(exp_fnc(cheb_nodes_np) - cheb10)

# We perform the errors graphic

plt.xlim(-1,1)

plt.plot(cheb_nodes,error3_3,label='Error of order 3')
plt.plot(cheb_nodes,error5_3,label='Error of order 5')
plt.plot(cheb_nodes,error10_3,label='Error of order 10')
plt.title('Exponential Error Function - Monomial Interpolation with Chebyshev Nodes')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(loc ='upper left',fontsize=8)
plt.show()

#%% RUNGE FUNCTION
# We call the function at these points

runge_cheb = runge_fnc(cheb_nodes_np)

# We copute the parameters for the monomial form by minimizing the error: f(x)=01+02*x+03*x^2+... for orders 3,5 and 10

par_cheb3_1 = np.polyfit(cheb_nodes,runge_cheb,3)
par_cheb5_1 = np.polyfit(cheb_nodes,runge_cheb,5)
par_cheb10_1 = np.polyfit(cheb_nodes,runge_cheb,10) 

# We compute the value of the polinomials at the solved value of the parameters

cheb3_1 = np.polyval(par_cheb3_1,cheb_nodes)
cheb5_1 = np.polyval(par_cheb5_1,cheb_nodes)
cheb10_1 = np.polyval(par_cheb10_1,cheb_nodes)

# We perform our graphic with the real function and the approximantions

plt.plot(cheb_nodes,runge_cheb,label='Ramp function')
plt.plot(cheb_nodes,cheb3_1,label='Cheb_approximation, order 3')
plt.plot(cheb_nodes,cheb5_1,label='Cheb_approximation, order 5')
plt.plot(cheb_nodes,cheb10_1,label='Cheb_approximation, order 10')
plt.title('Runge function - Monomial Interpolation with Chebishev Nodes')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(loc ='upper left',fontsize=8)
plt.show()

# We compute errors as the difference with the original function

error3_4 = abs(runge_fnc(cheb_nodes_np) - cheb3_1)
error5_4 = abs(runge_fnc(cheb_nodes_np) - cheb5_1)
error10_4 = abs(runge_fnc(cheb_nodes_np) - cheb10_1)

# We perform the errors graphic

plt.xlim(-1,1)

plt.plot(cheb_nodes,error3_4,label='Error of order 3')
plt.plot(cheb_nodes,error5_4,label='Error of order 5')
plt.plot(cheb_nodes,error10_4,label='Error of order 10')
plt.title('Runge Error Function - Monomial Interpolation with Chebyshev Nodes')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(loc ='upper left',fontsize=8)
plt.show()

#%% RAMP FUNCTION
# We call the function at these points

ramp_cheb = ramp_fnc(cheb_nodes_np)

# We copute the parameters for the monomial form by minimizing the error: f(x)=01+02*x+03*x^2+... for orders 3,5 and 10

par_cheb3_2 = np.polyfit(cheb_nodes,ramp_cheb,3)
par_cheb5_2 = np.polyfit(cheb_nodes,ramp_cheb,5)
par_cheb10_2 = np.polyfit(cheb_nodes,ramp_cheb,10) 

# We compute the value of the polinomials at the solved value of the parameters

cheb3_2 = np.polyval(par_cheb3_2,cheb_nodes)
cheb5_2 = np.polyval(par_cheb5_2,cheb_nodes)
cheb10_2 = np.polyval(par_cheb10_2,cheb_nodes)

# We perform our graphic with the real function and the approximantions

plt.plot(cheb_nodes,ramp_cheb,label='Ramp function')
plt.plot(cheb_nodes,cheb3_2,label='Cheb_approximation, order 3')
plt.plot(cheb_nodes,cheb5_2,label='Cheb_approximation, order 5')
plt.plot(cheb_nodes,cheb10_2,label='Cheb_approximation, order 10')
plt.title('Ramp function - Monomial Interpolation with Chebishev Nodes')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(loc ='upper left',fontsize=8)
plt.show()

# We compute errors as the difference with the original function

error3_5 = abs(ramp_fnc(cheb_nodes_np) - cheb3_2)
error5_5 = abs(ramp_fnc(cheb_nodes_np) - cheb5_2)
error10_5 = abs(ramp_fnc(cheb_nodes_np) - cheb10_2)

# We perform the errors graphic

plt.xlim(-1,1)

plt.plot(cheb_nodes,error3_5,label='Error of order 3')
plt.plot(cheb_nodes,error5_5,label='Error of order 5')
plt.plot(cheb_nodes,error10_5,label='Error of order 10')
plt.title('Ramp Error Function - Monomial Interpolation with Chebyshev Nodes')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(loc ='upper left',fontsize=8)
plt.show()

"""3.3 Chebyshev interpolation nodes and Chebyshev polynomial of order 3, 5 and 10. How
does it compare to the previous results? Report your approximation and errors"""

#%% EXPONENTIAL FUNCTION

# We copute the parameters for the Chevyshev polinomial form for orders 3,5 and 10

vector = np.linspace(-1,1,31)
ch = np.polynomial.chebyshev.chebroots(vector)

y = exp_fnc(ch)
    
par_pol3 = np.polynomial.chebyshev.chebfit(ch, y, 3)
par_pol5 = np.polynomial.chebyshev.chebfit(ch, y, 5)
par_pol10 = np.polynomial.chebyshev.chebfit(ch, y, 10)

# We compute the value of the polinomials at the solved value of the parameters

pol3 = np.polynomial.chebyshev.chebval(ch, par_pol3)
pol5 = np.polynomial.chebyshev.chebval(ch, par_pol5)
pol10 = np.polynomial.chebyshev.chebval(ch, par_pol10)

# We perform our graphic with the real function and the approximantions

plt.xlim(-1,1)

plt.plot(ch, y, label = 'Exponential function')
plt.plot(ch, pol3, label = 'Cheb_pol_approx, order 3')
plt.plot(ch, pol5, label = 'Cheb_pol_approx, order 5')
plt.plot(ch, pol10, label = 'Cheb_pol_approx, order 10')
plt.legend(loc = 'upper left')
plt.title('Exponential Function-Cheb polynomial approx')
plt.ylabel('f(x)')
plt.xlabel('x')
plt.show()

# We compute errors as the difference with the original function

er3 = abs(y - pol3)
er5 = abs(y - pol5)
er10 = abs(y - pol10)

# We perform the errors graphic

plt.xlim(-1,1)

plt.plot(cheb_nodes,er3,label='Error of order 3')
plt.plot(cheb_nodes,er5,label='Error of order 5')
plt.plot(cheb_nodes,er10,label='Error of order 10')
plt.title('Exponential Error Function - Cheb polynomial approx')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(loc ='upper left',fontsize=8)
plt.show()

#%% RUNGE FUNCTION

# We copute the parameters for the Chevyshev polinomial form for orders 3,5 and 10

y1 = runge_fnc(ch)
    
par_pol3_1 = np.polynomial.chebyshev.chebfit(ch, y1, 3)
par_pol5_1 = np.polynomial.chebyshev.chebfit(ch, y1, 5)
par_pol10_1 = np.polynomial.chebyshev.chebfit(ch, y1, 10)

# We compute the value of the polinomials at the solved value of the parameters

pol3_1 = np.polynomial.chebyshev.chebval(ch, par_pol3_1)
pol5_1 = np.polynomial.chebyshev.chebval(ch, par_pol5_1)
pol10_1 = np.polynomial.chebyshev.chebval(ch, par_pol10_1)

# We perform our graphic with the real function and the approximantions

plt.xlim(-1,1)

plt.plot(ch, y1, label = 'Exponential function')
plt.plot(ch, pol3_1, label = 'Cheb_pol_approx, order 3')
plt.plot(ch, pol5_1, label = 'Cheb_pol_approx, order 5')
plt.plot(ch, pol10_1, label = 'Cheb_pol_approx, order 10')
plt.legend(loc = 'upper left')
plt.title('Runge Function-Cheb polynomial approx')
plt.ylabel('f(x)')
plt.xlabel('x')
plt.show()

# We compute errors as the difference with the original function

er3_1 = abs(y1 - pol3_1)
er5_1 = abs(y1 - pol5_1)
er10_1 = abs(y1 - pol10_1)

# We perform the errors graphic

plt.xlim(-1,1)

plt.plot(cheb_nodes,er3_1,label='Error of order 3')
plt.plot(cheb_nodes,er5_1,label='Error of order 5')
plt.plot(cheb_nodes,er10_1,label='Error of order 10')
plt.title('Runge Error Function - Cheb polynomial approx')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(loc ='upper left',fontsize=8)
plt.show()


#%% RAMP FUNCTION

# We copute the parameters for the Chevyshev polinomial form for orders 3,5 and 10

y2 = ramp_fnc(ch)
    
par_pol3_2 = np.polynomial.chebyshev.chebfit(ch, y2, 3)
par_pol5_2 = np.polynomial.chebyshev.chebfit(ch, y2, 5)
par_pol10_2 = np.polynomial.chebyshev.chebfit(ch, y2, 10)

# We compute the value of the polinomials at the solved value of the parameters

pol3_2 = np.polynomial.chebyshev.chebval(ch, par_pol3_2)
pol5_2 = np.polynomial.chebyshev.chebval(ch, par_pol5_2)
pol10_2 = np.polynomial.chebyshev.chebval(ch, par_pol10_2)

# We perform our graphic with the real function and the approximantions

plt.xlim(-1,1)

plt.plot(ch, y2, label = 'Ramp function')
plt.plot(ch, pol3_2, label = 'Cheb_pol_approx, order 3')
plt.plot(ch, pol5_2, label = 'Cheb_pol_approx, order 5')
plt.plot(ch, pol10_2, label = 'Cheb_pol_approx, order 10')
plt.legend(loc = 'upper left')
plt.title('Ramp Function-Cheb polynomial approx')
plt.ylabel('f(x)')
plt.xlabel('x')
plt.show()

# We compute errors as the difference with the original function

er3_2 = abs(y2 - pol3_2)
er5_2 = abs(y2 - pol5_2)
er10_2 = abs(y2 - pol10_2)

# We perform the errors graphic

plt.xlim(-1,1)

plt.plot(cheb_nodes,er3_2,label='Error of order 3')
plt.plot(cheb_nodes,er5_2,label='Error of order 5')
plt.plot(cheb_nodes,er10_2,label='Error of order 10')
plt.title('Ramp Error Function - Cheb polynomial approx')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(loc ='upper left',fontsize=8)
plt.show()


"""QUESTION 2: FUNCTION APPROXIMATION, MULTIVARIATE

EXERCISE 1: Consider the following CES function f(k, h) = ((1 − α)*k^(σ−1/σ) + α*h^(σ−1/σ))^(σ/σ−1)
where σ is the elasticity of substitution (ES) between capital and labor and α is a relative input share 
parameter. Set α = 0.5, σ = 0.25, k ∈ [0, 10] and h ∈ [0, 10]. Do the following items:

1.3 Approximate f(k, h) using a 2-dimensional Chebyshev regression algorithm. Fix the number
of nodes to be 20 and try Cheby polynomials that go from degree 3 to 15. For each case,
plot the exact function and the approximation (vertical axis) in the (k, h) space."""

# We start setting the parameters of the CES function:

sigma = 0.25
alpha = 0.5
k = np.linspace(0,10,150)
h = np.linspace(0,10,150)
a = 0 # Lower bound of k and h
b = 10 # Upper bound of k and h
n = 20 # Number of nodes
k_min = 1e-3
k_max = 10
h_min = 1e-3
h_max = 10

# CES function:
 
def y_fun(k, h):
    return ((1 - alpha) * k ** ((sigma-1)/sigma) + alpha * h ** ((sigma - 1)/sigma)) ** (sigma/(sigma -1))

y = y_fun(k, h)
y_r = np.matrix(y)

# Create the chebyshev nodes and adapt then to the interval [0,10]:

def cheb_nodes(n,a,b):
    x = []
    y = []
    z = []
    for j in range(1,n+1):   
        z_k=-np.cos(np.pi*(2*j-1)/(2*n))   
        x_k=(z_k+1)*((b-a)/2)+a  
        y_k=(z_k+1)*((b-a)/2)+a
        z.append(z_k)
        x.append(x_k)
        y.append(y_k)
    return (np.array(z),np.array(x),np.array(y))

z, k_nodes, h_nodes = cheb_nodes(n,a,b)

# Evaluate the function at the approximation nodes:

w = np.matrix(y_fun(k_nodes[:, None],h_nodes[None, :])) 

# In order to compute the Chebyshev coefficients, we need to create the Chebyshev basis functions:

def cheb_poly(d,x):
    psi = []
    psi.append(np.ones(len(x)))
    psi.append(x)
    for i in range(1,d):
        p = 2*x*psi[i]-psi[i-1]
        psi.append(p)
    pol_d = np.matrix(psi[d]) 
    return pol_d

def cheb_coeff(z, w, d):
    thetas = np.empty((d+1) * (d+1))
    thetas.shape = (d+1,d+1)
    for i in range(d+1):
        for j in range(d+1):
            thetas[i,j] = (np.sum(np.array(w)*np.array((np.dot(cheb_poly(i,z).T,cheb_poly(j,z)))))/np.array((cheb_poly(i,z)*cheb_poly(i,z).T)*(cheb_poly(j,z)*cheb_poly(j,z).T)))
    return thetas

def cheb_approx(x, y, thetas, d):
    f = []
    in1 = (2*(x-a)/(b-a)-1)
    in2 = (2*(y-a)/(b-a)-1)
    for u in range(d):
        for v in range(d):
                f.append(np.array(thetas[u,v])*np.array((np.dot(cheb_poly(u,in1).T,cheb_poly(v,in2)))))
    f_sum = sum(f)
    return f_sum

## Degree 3 approximation:

order = 3
thetas = cheb_coeff(z,w,order)
y_approx3 = cheb_approx(k_nodes, h_nodes,thetas,order)

X, Y = np.meshgrid(k_nodes,h_nodes)
## Plotting the real function:

real = y_fun(X,Y)
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot_surface(X, Y, real)
plt.title('Real CES production function')
ax.set_xlabel('k')
ax.set_ylabel('h')
ax.set_zlabel('y_fun');
plt.show()

#Plotting the approximation:


fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot_surface(X, Y, y_approx3)
plt.title('Approximation of degree 3')
ax.set_xlabel('k')
ax.set_ylabel('h')
ax.set_zlabel('y_fun');
plt.show()

#Approx error
error3 = abs(real-y_approx3)
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot_surface(X, Y, error3)
plt.title('Errors of approximation of degree 3')
ax.set_xlabel('k')
ax.set_ylabel('h')
ax.set_zlabel('y_fun');
plt.show()

## Degree 5 approximation:

order = 5
thetas = cheb_coeff(z,w,order)
y_approx5 = cheb_approx(k_nodes, h_nodes,thetas,order)

X, Y = np.meshgrid(k_nodes,h_nodes)


#Plotting the approximation:


fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot_surface(X, Y, y_approx5)
plt.title('Approximation of degree 5')
ax.set_xlabel('k')
ax.set_ylabel('h')
ax.set_zlabel('y_fun');
plt.show()

#Approx error

error5 = abs(real-y_approx5)
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot_surface(X, Y, error5)
plt.title('Errors of approximation of degree 5')
ax.set_xlabel('k')
ax.set_ylabel('h')
ax.set_zlabel('y_fun');
plt.show()

## Degree 10 approximation:

order = 10
thetas = cheb_coeff(z,w,order)
y_approx10 = cheb_approx(k_nodes, h_nodes,thetas,order)

X, Y = np.meshgrid(k_nodes,h_nodes)

#Plotting the approximation:

fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot_surface(X, Y, y_approx10)
plt.title('Approximation of degree 10')
ax.set_xlabel('k')
ax.set_ylabel('h')
ax.set_zlabel('y_fun');
plt.show()

#Approx error

error10 = abs(real-y_approx10)
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot_surface(X, Y, error10)
plt.title('Errors of approximation of degree 10')
ax.set_xlabel('k')
ax.set_ylabel('h')
ax.set_zlabel('y_fun');
plt.show()

## Degree 15 approximation:

order = 15
thetas = cheb_coeff(z,w,order)
y_approx15 = cheb_approx(k_nodes, h_nodes,thetas,order)

X, Y = np.meshgrid(k_nodes,h_nodes)

#Plotting the approximation:

fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot_surface(X, Y, y_approx15)
plt.title('Approximation of degree 15')
ax.set_xlabel('k')
ax.set_ylabel('h')
ax.set_zlabel('y_fun');
plt.show()

#Approx error

error15 = abs(real-y_approx15)
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot_surface(X, Y, error15)
plt.title('Errors of approximation of degree 15')
ax.set_xlabel('k')
ax.set_ylabel('h')
ax.set_zlabel('y_fun');
plt.show()

"""1.4 Plot the exact isoquants associated with the percentiles 5, 10, 25, 50, 75, 90 and 95 of
output. Use your approximation to plot the isoquants of the your approximation. Plot the
associated errors per each of these isoquant.For each case, show the associated approximation errors 
(vertical axis) in the (k, h) space."""

#Real function

percentiles = np.array([5,10,25,50,75,90,95])
j = -1
levels = np.empty(len(percentiles))
for p in percentiles:
    j += 1
    levels[j] = np.percentile(real, p)

plt.contour(X,Y,real,levels)
plt.title('Isoquants for the real CES function')
plt.show()

# Approximation 3:

j = -1
levels3 = np.empty(len(percentiles))
for p in percentiles:
    j += 1
    levels3[j] = np.percentile(y_approx3, p)

plt.contour(X,Y,y_approx3,levels3)
plt.title('Isoquants of order 3 approximation')
plt.show()

errorlevels3 = abs(levels-levels3)
plt.plot(errorlevels3)
plt.title('Error between percentiles-order 3')
plt.show()

# Approximation 5:

j = -1
levels5 = np.empty(len(percentiles))
for p in percentiles:
    j += 1
    levels5[j] = np.percentile(y_approx5, p)

plt.contour(X,Y,y_approx5,levels5)
plt.title('Isoquants of order 5 approximation')
plt.show()

errorlevels5 = abs(levels-levels5)
plt.plot(errorlevels5)
plt.title('Error between percentiles-order 5')
plt.show()

# Approximation 10:

j = -1
levels10 = np.empty(len(percentiles))
for p in percentiles:
    j += 1
    levels10[j] = np.percentile(y_approx10, p)

plt.contour(X,Y,y_approx10,levels10)
plt.title('Isoquants of order 10 approximation')
plt.show()

errorlevels10 = abs(levels-levels10)
plt.plot(errorlevels10)
plt.title('Error between percentiles-order 10')
plt.show()

# Approximation 15:

j = -1
levels15 = np.empty(len(percentiles))
for p in percentiles:
    j += 1
    levels15[j] = np.percentile(y_approx15, p)

plt.contour(X,Y,y_approx15,levels15)
plt.title('Isoquants of order 15 approximation')
plt.show()

errorlevels15 = abs(levels-levels15)
plt.plot(errorlevels15)
plt.title('Error between percentiles-order 15')
plt.show()












