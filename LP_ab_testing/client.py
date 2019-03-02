import requests
import pandas as pd

'''
LazyProgrammers simple client to interact with the simulated server that is
learning which advertisement to serve. The client checks, which ad is being
served, then takes the next sample (1 or 0 / click or no click) for that ad
and sends the result (until one of the advertisements datasets is exhausted.)
'''

# get data
df = pd.read_csv('click_data.csv')
ad_A = df[df['advertisement_id'] == 'A']
ad_B = df[df['advertisement_id'] == 'B']
ad_A = ad_A['action'].values
ad_B = ad_B['action'].values

print("a.mean:", ad_A.mean())
print("b.mean:", ad_B.mean())

idxA = 0
idxB = 0
count = 0
while idxA < len(ad_A) and idxB < len(ad_B):
    # quit when there's no data left for either ad
    r = requests.get('http://localhost:8888/get_ad')
    r = r.json()
    if r['advertisement_id'] == 'A':
        action = ad_A[idxA]
        idxA += 1
    else:
        action = ad_B[idxB]
        idxB += 1

    if action == 1:
        # only click the ad if our dataset determines that we should
        # adds a click to the ad specified in the file 'click_data.csv'
        requests.post(
          'http://localhost:8888/click_ad',
          data={'advertisement_id': r['advertisement_id']}
        )

    # log some stats
    count += 1
    if count % 50 == 0:
        print("Seen %s ads, A: %s, B: %s" % (count, idxA, idxB))
