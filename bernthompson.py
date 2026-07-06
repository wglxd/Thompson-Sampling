import numpy as np
import matplotlib.pyplot as mp

trials = int(input())

arr = [0.9, 0.8, 0.7, 0.6]
k = len(arr)

fin_ans = [0]*k
fin_regret = [0]*1000

for trial in range(trials):
    alphas = [1]*k
    betas = [1]*k
    rng = np.random.default_rng()

    regrets = [0]*1000
    sum = 0
    
    for sample in range(1000):
        param_probs = ([rng.beta(a=alphas[i], b = betas[i]) for i in range(k)])
        x = max(param_probs)
        ind = param_probs.index(x)
        ch = rng.choice([1,0], p = [arr[ind], 1 - arr[ind]])
        if(ch == 1):
            alphas[ind]+=1
        else:
            betas[ind]+=1

        best_arm = max(arr)
        sum += (best_arm - arr[ind])
        regrets[sample] = sum

    for i in range(k):
        fin_ans[i] += (alphas[i])/(betas[i]+alphas[i])
    for i in range(1000):
        fin_regret[i] += regrets[i]

    
    
final_a = [fin_ans[i]/(trials) for i in range(k)]
final_r = [(fin_regret[i])/(trials) for i in range(1000)]

mp.plot(final_r)
mp.xlabel("Round")
mp.ylabel("Cumulative Regret")
mp.show()
print(final_a)



    





    







