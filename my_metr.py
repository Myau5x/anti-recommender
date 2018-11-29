import pandas as pd
import numpy as np

look = pd.read_csv('whats_wrong/model2_t.csv')

act = look.stars
base = look.mean_rating
pred = look.prediction
pred[pred>5] = 5
pred[pred<1] = 1


def accu(act, pred, thres):
    return ((act >thres) == (pred >thres)).mean()

thres = np.linspace(1,5,9)
a_base = np.zeros(9)
a_pred = np.zeros(9)
fig, ax = plt.subplots()

for i  in range(9):
    a_base[i] = accu(act, base, thres[i])
    a_pred[i] = accu(act, pred, thres[i])
ax.plot(thres, a_base, color = 'r')
ax.plot(thres, a_pred, color = 'b')


pred[act <3.5].mean(), base[act<3.5].mean()
###(3.1792809133547317, 3.4462038511106057)
