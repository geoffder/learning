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

# degrees of freedom (# of groups - 1)
df = num_ads - 1
# build contingency table
table = np.zeros((2, 2))
table[0, :] = [A_clicks.sum(), N - A_clicks.sum()]  # ad A, click vs no click
table[1, :] = [B_clicks.sum(), N - B_clicks.sum()]  # ad B, click vs no click

print('Degrees of freedom:', df)
print('Advertisement A (click, no click): (%d, %d)'
      % (table[0, 0], table[0, 0]))
print('Advertisement B (click, no click): (%d, %d)'
      % (table[1, 0], table[1, 1]))

# 2x2 shortcut method of calculating
# chisq = ((ad-bc)**2 * (a+b+c+d)) / ((a+b)+(c+d)+(a+c)+(b+d))
chisq = ((table[0, 0]*table[1, 1] - table[0, 1]*table[1, 0])**2
         * (table[0, 0] + table[1, 1] + table[0, 1] + table[1, 0])) \
        / ((table[0, 0]+table[0, 1])*(table[1, 0]+table[1, 1])
           * (table[0, 0]+table[1, 0])*(table[0, 1]+table[1, 1]))
p_value = 1 - stats.chi2.cdf(chisq, df)

# should also do it using observed vs expected method as well since it is
# intuitive, rather than unreadable, like the shortcut above.

print('my chi-square:', chisq)
print('my p-value: %.5f' % p_value)

# now compare with built-in scipy function
scipy_chisq, scipy_p, dof, ex = stats.chi2_contingency(table, correction=False)
print('scipy chisq:', scipy_chisq)
print('scipy p-value: %.5f' % scipy_p)
print('scipy dof:', dof)
# print('scipy expected:', ex)

plt.bar('Advertisement A', A_clicks.mean())
plt.bar('Advertisement B', B_clicks.mean())
plt.ylabel('Mean Clicks')
plt.title('p-value: %.5f' % p_value)
plt.show()
