import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# generate data
N = 100
X1 = np.random.randn(N)
X2 = np.random.randn(N) + .5

plt.scatter(np.arange(N), X1)
plt.scatter(np.arange(N), X2)
plt.show()

# degrees of freedom
df = 2 * (N - 1)

# pooled variance
# Sp = np.sqrt((X1.std()**2 + X2.std()**2) / 2)  # my way, calculated std
Sp = np.sqrt((X1.var(ddof=1) + X2.var(ddof=1)) / 2)  # using variance with N-1

# t-value
t_value = (X1.mean() - X2.mean())/(Sp * np.sqrt(2/N))

# p-value (works regardless of which mean is larger, thus which tail we're in)
p_value = (.5 - np.abs(stats.t.cdf(t_value, df) - .5)) * 2

# now compare with built-in scipy function
scipy_t, scipy_p = stats.ttest_ind(X1, X2)

print('Degrees of freedom:', df)
print('Pooled Standard Deviation:', Sp)

print('my t-value:', t_value)
print('my p-value: %.5f' % p_value)

print('scipy t-value:', scipy_t)
print('scipy p-value: %.5f' % scipy_p)
