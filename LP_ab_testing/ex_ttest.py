import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

data = pd.read_csv('click_data.csv')
ads = data['advertisement_id'].unique()
num_ads = len(ads)  # there are 2, only A and B

A_clicks = data['action'][data['advertisement_id'] == 'A']
B_clicks = data['action'][data['advertisement_id'] == 'B']
print('N (A, B): (%d, %d)' % (len(A_clicks), len(B_clicks)))
N = len(A_clicks)  # same for each group, so no problem

# degrees of freedom
df = 2 * (N - 1)
# pooled variance (using variance with N-1)
Sp = np.sqrt((A_clicks.var(ddof=1) + B_clicks.var(ddof=1)) / 2)
# t-value
t_value = (A_clicks.mean() - B_clicks.mean())/(Sp * np.sqrt(2/N))
# p-value (works regardless of which mean is larger, thus which tail we're in)
p_value = (.5 - np.abs(stats.t.cdf(t_value, df) - .5)) * 2

print('Degrees of freedom:', df)
print('Pooled Standard Deviation:', Sp)

print('my t-value:', t_value)
print('my p-value: %.5f' % p_value)

# now compare with built-in scipy function
# scipy_t, scipy_p = stats.ttest_ind(A_clicks, B_clicks)
# print('scipy t-value:', scipy_t)
# print('scipy p-value: %.5f' % scipy_p)

plt.bar('Advertisement A', A_clicks.mean())
plt.bar('Advertisement B', B_clicks.mean())
plt.ylabel('Mean Clicks')
plt.title('p-value: %.5f' % p_value)
plt.show()
