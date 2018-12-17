import csv # loading of delimited files (should do myself tbh)
import matplotlib.pyplot as plt # plotting functions
import re #regular expressions
import numpy as np # arrays

keys = ['chip', 'transistors', 'year', 'make', 'size', 'area']
mooreDicts = []

with open('moore.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='\t')
    rowNum = 0
    for row in csv_reader:
        mooreDicts.append({})
        for col in range(len(row)):
            mooreDicts[rowNum][keys[col]] = row[col]
        rowNum += 1

# pull out the years and transistor counts
years = list(range(len(mooreDicts)))
transistors = list(range(len(mooreDicts)))
for i in range(len(mooreDicts)):
    years[i] = mooreDicts[i]['year']
    transistors[i] = mooreDicts[i]['transistors']

# clean the data
for i in range(len(years)):
    years[i] = int(years[i].split('[')[0])
    transistors[i] = transistors[i].split('[')[0] #.replace(',','') instead use regex
    transistors[i] = int(re.sub('[^0-9]','', transistors[i]))
years = np.array(years)
transistors = np.array(transistors)

# linearize the data and calculate a best fit
logTransistors = np.log(transistors)
denom = years.dot(years) - years.mean()*years.sum()
a = (logTransistors.dot(years) - years.mean() *logTransistors.sum()) / denom
b = (logTransistors.mean()*years.dot(years) - years.mean()*logTransistors.dot(years)) / denom
logTransFit = a*years + b

# calculate quality of fit (r-squared)
SSres = ((logTransistors - logTransFit)**2).sum()
SStot = ((logTransistors - logTransistors.mean())**2).sum()
Rsq = 1.0 - SSres/SStot
print("R^2 = " + str(Rsq))

# calculating doubling time (incl notes from course)
# log(trans) = a*year + b, therefore...
# trans = exp(b) * exp(a*year) [remember log changes betw multi and addi]
# therefore, 2*trans = 2 * exp(b) * exp(a*year) = exp(ln(2)) * exp(b) * exp(a*year)
#                    = exp(b) * exp(a * year + ln(2)) [multiplying exponents eg 2^2 * 2^2 = 2^4]
# exp(b)*exp(a*year2) = exp(b) * exp(a * year1 + ln(2)) [divide out exp(b)]
# [log it] a*year2 = a*year1 + ln(2)
# year2 = year1 + ln(2)/a
print("time to double transistor count: " + str(np.log(2)/a) + " years")

plt.scatter(years, logTransistors)
plt.plot(years, logTransFit)
plt.show()
