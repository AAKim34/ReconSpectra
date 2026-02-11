import pandas as pd
import math
import numpy as np
import scipy.optimize, scipy.integrate
from scipy.stats import norm
import random




def lambdau10(x):
    _lambdau = lambda t, x: np.exp(-t)*np.cos(t*x + 2*t/np.pi*np.log(t))
    res = np.array(scipy.integrate.quad(_lambdau, 0, np.inf, args=(x,)))
    res[0] /= np.pi
    return res
try:
    _landau10_y, _landau10_x
except:
    print("Initializing Landau10)")
    _landau10_x = np.linspace(-30, 30, int(3e3))
    _landau10_y = np.array([lambdau10(i) for i in _landau10_x])[:,0]


def spec(e, a, b, E0):
    return a * Landau10((E0 - e) / b)/b

def distr(df_mono,spec):
    ar_mono = df_mono.to_numpy()
    return np.matmul(ar_mono, spec)


Landau10 = lambda x: np.interp(x, _landau10_x, _landau10_y)



def gauss(x, a, sigma, mu):
  gauss = a*np.exp(-((x-mu)**2)/(2*(sigma)**2))/(sigma*(2*np.pi)**(1/2))

  return gauss



def gauss_mult(x,a,sigma, mu, peaks):
  if peaks ==1:
    ans = gauss(x,a,sigma, mu)
  elif peaks == 2:
    a1 = random.random()
    a2 = 1 - a1
    a0 = [a1, a2]
    a0.sort()

    mu2 = mu*(1 - random.random())
    sigma2 = random.uniform(0.1, 1)
    ans = gauss(x,a0[0],sigma2, mu2) + gauss(x,a0[1],sigma, mu)


  return ans/max(ans)


def mono_land(a,b, e):
  mono_land = []
  for E0 in np.linspace(1, 19.9, 380):
    sp = spec(e, a, b, E0).tolist()
    mono_land.append(sp)

  return np.array(mono_land)
