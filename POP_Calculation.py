import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.integrate import quad
import math


S_current = 96
K=100
T=0.5
rate = 0.007
sigma = 0.54

#S_current: current stock price
#K: strike price
#T: time to maturity (years)
#rate: interest rate
#sigma: volatility of underlying asset

def Black_Sholes(S_current, K, T, rate, sigma, option = 'call'):
    
    d1 = (np.log(S_current / K) + (rate + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S_current / K) + (rate - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    if option == 'call':
        result = (S_current * st.norm.cdf(d1) - K * np.exp(-rate * T) * st.norm.cdf(d2))
    if option == 'put':
        result = (K * np.exp(-rate * T) * st.norm.cdf(-d2) - S_current * st.norm.cdf(-d1))
        
    return result

C = Black_Sholes(S_current,K,T,rate,sigma, option='call')

def integrand(S):
	numerator = math.exp((-math.log(S/K)-((rate-(sigma**2/2))*T))**2/(2*(sigma**2)*T))
	denominator = S*sigma*math.sqrt(2*math.pi*T)
	return(numerator/denominator)

print("Profit Probability :", (quad(integrand,0,K+C)[0]))  #gives a tuple of integral value and the error