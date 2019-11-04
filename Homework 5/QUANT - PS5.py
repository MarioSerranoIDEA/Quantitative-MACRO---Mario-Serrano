# PS5 FROM MARIO SERRANO GARCÍA

import numpy as np
from numpy import random
import random as rnd
import math
from scipy import optimize
from scipy import integrate
import seaborn
import matplotlib.pyplot as plt

"""QUESTION 1: FACTOR INPUT MISSALOCATION

1.1 Firm-speci c output, capital and productivity are, respectively, yi; ki and zi. Assume that ln zi and ln ki follow a 
joint normal distribution. Assume that the correlation between ln zi and ln ki is zero, the variance of ln zi is equal 
to 1.0, the variance of ln ki is equal to 1.0, and that average s and k is equal to one. Then simulate 10,000,000 
observations and plot the joint density in logs and in levels. We are going to assume that these 10,000,000 observations 
are your complete (or administrative) data that captures the entire population/universe of fi
rms in a given country."""

# We set values of the parameters from the statement

sigma_log = 1
E_k = 0
E_s = 0
gamma = 0.6
# Exercise 2
# gamma = 0.8
dim = 10000000

# First, we need moment condition of first oreder(mean) for the log variables. We know that if one variable X has a normal
# distribution, the e^(X) follows a log-normal distribution. This distribution approaches to:

def log_normal_K(X,mu,sigma):
    f = 1/(X*sigma*math.sqrt(2*math.pi))*math.exp(-(math.log(X)-mu)**2/(2*sigma**2))
    return f

# Taking expectations we get: E(X) = e^(mu + 1/2*sigma^2). This is the expectation of the log-normal variable (in this case k & z)
# Solving for the expectation of the original variable X:

def mu_log(mu,sigma):
    return 0 - sigma**2/2

# We compute the expectation for logK
mu_logK = mu_log(E_k,sigma_log)
print('E(ln k) = ' + str(mu_logK))

# To compute expectation for variable logZ we have to rearrange the distribution function, since s = z^(1-gamma)
def log_normal_Z(s,mu):
    return (1-gamma)/(np.sqrt(2*math.pi))*np.exp(-0.5*((1-gamma)*np.log(s)-mu)**2)

# We integrate
def F(mu):
    F = integrate.quad(log_normal_Z,0,np.inf,args=(mu))[0]
    return 1-F

x = optimize.fsolve(F,-1)
print(x[0])
mu_logZ = -1.25 # Approximately
# Exercise 2
# mu_logZ = -2.5
print('E(ln z) = ' + str(mu_logZ))

# Now, we can construct the variance-covariance matrix & the mean array to create our sample from a pseudo-random 
# multivariate normal distribution

cov = np.identity(2)
# Exercise 1.6, Case I
# cov = np.matrix([[1,0.5],[0.5,1]])
# Exercise 1.6, Case II
# cov = np.matrix([[1,-0.5],[-0.5,1]])
mean = np.array([mu_logK,mu_logZ])

np.random.seed(7)
log_data = np.random.multivariate_normal(mean,cov,size=dim)
level_data = np.exp(log_data)

# Plotting the results for each variable

logk = log_data[:,0]
logz = log_data[:,1]
k = level_data[:,0]
z = level_data[:,1]

# Level samples
seaborn.jointplot(k,z,kind="scatter").set_axis_labels("Capital", "Productivity")
plt.show()

# Log samples
seaborn.jointplot(logk,logz,kind="scatter").set_axis_labels("Log Capital", "Log Productivity")
plt.show()

"""1.2 Compute the firm output yi for each of your observations"""

# We deffine the function

def yi(si,ki):
    return si**(1-gamma)*ki**gamma

# Compute variable si from zi

s = np.empty(dim)
for i in range(dim):
    s[i]= z[i]**(1/(1-gamma))

# Compute output for each observaion

y = np.empty(dim)
for i in range(dim):
    y[i] = yi(s[i],k[i])

"""1.3 Solve the maximization problem."""

# The social planner maximization problem is: max sum(si^(1-gamma)*ki^gamma)
# Substituting the optimal capital into it: max s1^(1-gamma)*(K - sum(ki))^gamma + sum(si^(1-gamma)*ki^gamma)
# FOC: -s1^(1-gamma)*gamma*(K - sum(ki))^(gamma-1) + gamma*si^(1-gamma)*ki^(gamma-1) = 0 --> s1*ki = si*k1
# Therefore if we sum across i: ki = si/S*K

# We define the aggregate capital demand K & aggregate productivity

K = np.sum(k)
S = np.sum(s)

# We compute the optimal k  for each firm

k_opt = np.empty(dim)
for i in range(dim):
    k_opt[i] = s[i]/S*K

"""1.4 Compare the optimal allocations against the data."""

# Plotting capital distributions on front of firm's productivity

seaborn.jointplot(k,s,kind="scatter",color='yellow').set_axis_labels("Capital", "Productivity")
plt.show()

seaborn.jointplot(k_opt,s,kind="scatter",color='green').set_axis_labels("Capital", "Productivity")
plt.show()

"""1.5 Compute the ouptut gains from reallocation"""

# We compute the actual output
Y = np.sum(y)

# We compute the output with the reallocation
y_opt = np.empty(dim)
for i in range(dim):
    y_opt[i] = yi(s[i],k_opt[i])

Y_opt = np.sum(y_opt)

# The gains are

gains_pop = (Y_opt/Y - 1)*100
print('Gains over the total output from reallocation: ' + str(gains_pop) + '%')

"""1.6 Redo items (2)-(5) assuming that the correlation between ln zi and ln ki is 0.50. Redo with
correlastion -0.50."""

# We can compute the covarance from the correlation coefficient of Pearson with: corr(X,Y) = cov(X,Y)/std(X)*std(Y). Therefore:
def Pearson(var_X,var_Y,cov):
    return cov/(math.sqrt(var_X)*math.sqrt(var_Y))

# Case I: corr(X,Y) = 0.50 --> cov(X,Y) = 0.50
# Case II: corr(X,Y) = -0.50 --> cov(X,Y) = -0.50


"""QUESTION 3: FROM COMPLETE DISTRIBUTIONS TO RANDOM SAMPLES

3.1 Please, random sample (without replacement) 10,000 observations. That is, your data sample
implies a sample-to-population ratio of 1/1,000. What is the variance of ln zi and ln ki in
your random sample? How do they compare compare to the complete data? How about the
correlation between ln zi and ln ki?"""

# We generate our sample

# sample_size = 10000
# Exercise 3.5
sample_size = 100000
# sample_size = 1000
# sample_size = 100
logk_list = list(logk)
logz_list = list(logz)
sample_logk = rnd.sample(logk_list,sample_size)
sample_logz = rnd.sample(logz_list,sample_size)

# We compute the variance of the sample & the population

vark_sample = np.var(sample_logk)
varz_sample = np.var(sample_logz)
varCov_sample = np.cov(sample_logk,sample_logz)
cov_sample = varCov_sample[0,1]

vark_pop = np.var(logk)
varz_pop = np.var(logz)
varCov_pop = np.cov(logk,logz)
cov_pop = varCov_pop[0,1]

print('                                   ')
print('Population Variance')
print('----------------------------')
print('var(log_k) = ' + str(vark_pop))
print('var(log_z) = ' + str(varz_pop))
print('Sample Variance')
print('----------------------------')
print('var(log_k) = ' + str(vark_sample))
print('var(log_z) = ' + str(varz_sample))

# We compute the correlation in population & sample as before

corr_sample = Pearson(vark_sample,varz_sample,cov_sample)
corr_pop = Pearson(vark_pop,varz_pop,cov_pop)

# Printing resutls

print('                                   ')
print('Population Correlation')
print('----------------------------')
print('corr(logk,logz) = ' + str(corr_pop))
print('                                   ')
print('Sample Correlation')
print('----------------------------')
print('corr(logk,logz) = ' + str(corr_sample))
print('                                   ')

"""3.2 Redo items (3) to (5) in Question 1 for your random sample of 10,000 firms. Compare your results for 
misallocation using your random sample to the results obtained using the complete distribution."""

# We define the aggregate capital demand K & aggregate productivity

k_sample = np.exp(sample_logk)
K_sample = np.sum(k_sample)

z_sample = np.exp(sample_logz)
s_sample = np.empty(sample_size)
for i in range(sample_size):
    s_sample[i]= z_sample[i]**(1/(1-gamma))

S_sample = np.sum(s_sample)   

# We compute the optimal k  for each firm

k_opt_sample = np.empty(sample_size)
for i in range(sample_size):
    k_opt_sample[i] = s_sample[i]/S_sample*K_sample

# Plotting capital distributions on front of firm's productivity
  
seaborn.jointplot(k_sample,s_sample,kind="scatter",color='yellow').set_axis_labels("Capital", "Productivity")
plt.show()

seaborn.jointplot(k_opt_sample,s_sample,kind="scatter",color='green').set_axis_labels("Capital", "Productivity")
plt.show()

# We compute the actual output

y_sample = np.empty(sample_size)
for i in range(sample_size):
    y_sample[i] = yi(s_sample[i],k_sample[i])
    
Y_sample = np.sum(y_sample)    

# We compute the output with the reallocation

y_opt_sample = np.empty(sample_size)
for i in range(sample_size):
    y_opt_sample[i] = yi(s_sample[i],k_opt_sample[i])

Y_opt_sample = np.sum(y_opt_sample)
    
# The gains are

gains_sample = (Y_opt_sample/Y_sample - 1)*100
print('Gains over the total output in the sample from reallocation: ' + str(gains_sample) + '%')

"""3.3 Do the previous two items 1,000 times. Notice that each random sample is drawn from the
entire population. This implies that you will compute 1,000 measures of misallocation. Show
the histogram of the output gains, and provide some statistics of that distribution of these
output gains, in particular, the median. Discuss your results."""

t = 1000
gains_samplet = np.empty(t)
j=0
while j<t:
    # Generating sample
    samplet_logk = rnd.sample(logk_list,sample_size)
    samplet_logz = rnd.sample(logz_list,sample_size)
    # Computing agreggate capital & productivity
    k_samplet = np.exp(samplet_logk)
    K_samplet = np.sum(k_samplet)
    
    z_samplet = np.exp(samplet_logz)
    s_samplet = np.empty(sample_size)
    for i in range(sample_size):
        s_samplet[i]= z_samplet[i]**(1/(1-gamma))
    S_samplet = np.sum(s_samplet)   
    # Computing optimal k  for each firm
    k_opt_samplet = np.empty(sample_size)
    for i in range(sample_size):
        k_opt_samplet[i] = s_samplet[i]/S_samplet*K_samplet
    # Computing actual output
    y_samplet = np.empty(sample_size)
    for i in range(sample_size):
        y_samplet[i] = yi(s_samplet[i],k_samplet[i])
    
    Y_samplet = np.sum(y_sample)  
    # Computing output after reallocation
    y_opt_samplet = np.empty(sample_size)
    for i in range(sample_size):
        y_opt_samplet[i] = yi(s_samplet[i],k_opt_samplet[i])

    Y_opt_samplet = np.sum(y_opt_samplet)
    # Computing gains
    gains_samplet[j] = (Y_opt_samplet/Y_samplet - 1)*100
    j = j+1

mean_gst = np.mean(gains_samplet)
median_gst = np.median(gains_samplet)

print('Mean: ' + str(mean_gst))
print('Median: ' + str(median_gst))

# Plotting the histogram of gains
    
plt.hist(gains_samplet, bins = 100)
plt.show()

"""3.5 What is the probability that a random sample delivers the misallocation gains within an
interval of 10% with respect to the actual misallocation gains obtained from complete data?"""

# From ours 1000 extractions, we can compute the percentage of them that are inside the specified bonds

tolerance = 0.1
upper = gains_pop*(1+tolerance)
lower = gains_pop*(1-tolerance)

positive_result = 0
for i in range(t):
    if gains_samplet[i]<=upper and gains_samplet[i]>=lower:
        positive_result += 1

inside_per = positive_result/t*100
print('Percentage of gains inside bonds: ' + str(inside_per))




