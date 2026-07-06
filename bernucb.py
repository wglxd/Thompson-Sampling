import numpy as np
import matplotlib.pyplot as plt
import math

trials = int(input())

arr = [0.9, 0.8, 0.7, 0.6]
k = len(arr)

fin_ans = [0]*k
fin_regret = [0]*1000

for trial in range(trials):
    alphas = [1]*k
    betas = [1]*k
    hash = [0]*k

    sum = 0
    rng = np.random.default_rng()
    regrets = [0]*1000

    for sample in range(1000):
        n = [alphas[i] + betas[i] - 2 for i in range(k)]   
        ind = -1
        if min(n) == 0:
            ind = n.index(0)                                
        else:
            param_probs = [alphas[i]/(alphas[i]+betas[i]) + math.sqrt(2*math.log(sample+1)/n[i])
                   for i in range(k)]
            ind = param_probs.index(max(param_probs))
        
        ch = rng.choice([1,0], p = [arr[ind], 1 - arr[ind]])
        if(ch == 1):
            alphas[ind]+=1
        else:
            betas[ind]+=1

        best_arm = max(arr)
        sum += (best_arm - arr[ind])
        regrets[sample] = sum

    for i in range(k):
        fin_ans[i] += (alphas[i]/(alphas[i]+betas[i]))
    for i in range(1000):
        fin_regret[i] += regrets[i]


final_a = [fin_ans[i]/(trials) for i in range(k)]
final_r = [(fin_regret[i])/(trials) for i in range(1000)]

plt.plot(final_r)
plt.xlabel("Round")
plt.ylabel("Cumulative Regret")
plt.show()
print(final_a)

        
