import pickle
import matplotlib.pyplot as plt
from collections import deque

with open('results/scores-2.pkl', 'rb') as f:
    scores = pickle.load(f)

cumsum = 0
last_100 = deque(maxlen=100)
running_avg = []
running100_avg = []

for i, score in enumerate(scores, 1):
    cumsum += score
    last_100.append(score)
    running_avg.append(cumsum/i)
    # if sum(last_100)/100 > 13:
    #     print(i)
    #     break
    if i < 100:
        running100_avg.append(sum(last_100)/i)
    else:
        running100_avg.append(sum(last_100)/100)

plt.plot(running100_avg)
plt.savefig('results/myplot2.png')


