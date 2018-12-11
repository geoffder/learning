import csv
import matplotlib.pyplot as plt
import re
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

plt.plot(years, np.log(transistors))
plt.show()
