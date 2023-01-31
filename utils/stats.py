import scipy.stats
from scipy.stats import ttest_ind, wilcoxon
import numpy as np
import matplotlib.pyplot as plt

def ttest_ratio(estimations, th=0.05):
    cpt_diff, cpt = 0, 0
    for i in range(len(estimations[0])):
        for j in range(i + 1, len(estimations[0])):

            s, p = ttest_ind(estimations[:,i], estimations[:,j])

            if p < th:
                cpt_diff += 1
            cpt+=1

    ratio = cpt_diff / cpt
    return ratio

def wilcoxon_ratio(estimations, th=0.05):
    cpt_diff, cpt = 0, 0
    for i in range(len(estimations[0])):
        for j in range(i + 1, len(estimations[0])):

            diff = estimations[:,i] - estimations[:,j]
            w, p = wilcoxon(diff)

            if p < th:
                cpt_diff += 1
            cpt+=1

    ratio = cpt_diff / cpt
    return ratio

def evaluate_fitting(all_contents, mean_estimations, std_estimations, N, display_eval=True):
    all_stds = []
    all_estims = []
    sum_rmse = 0
    sum_intra_rmse = 0
    sum_intra_rmse2 = 0
    all_contents_norm = all_contents / all_contents[-1][-1]
    for x in range(len(all_contents)):
        estimated_values = [0] + list(mean_estimations[x*(N-1):(x+1)*(N-1)])
        std_err = np.array([0] + list(std_estimations[x*(N-1):(x+1)*(N-1)]))# * 1.96
        all_stds.append(std_err)
        all_estims.append(estimated_values)
        rmse_val = np.mean((all_contents_norm[x] - estimated_values) ** 2) ** 0.5
        rmse_intra_val = np.mean((all_contents_norm[x]/all_contents_norm[x][-1] - estimated_values/estimated_values[-1]) ** 2) ** 0.5
        sum_rmse += rmse_val
        sum_intra_rmse += rmse_intra_val
        
        rmse_intra_val = np.mean((all_contents_norm[x]/all_contents_norm[x][-1] - estimated_values/estimated_values[-1])[3:] ** 2) ** 0.5
        sum_intra_rmse2 += rmse_intra_val
        
        if display_eval:
            index = [x for x in range(N)]
            plt.subplot(1, 2, 1)
            plt.errorbar(index, estimated_values, yerr=std_err, ls='--', marker='o', capsize=5, capthick=1)
            plt.subplot(1, 2, 2)
            plt.plot(index,all_contents[x])
    
    if display_eval:
        print(all_contents)
        for u in all_estims:
            print(u)
        plt.show()
    print(sum_rmse/len(all_contents), sum_intra_rmse/len(all_contents))
    return sum_rmse/len(all_contents), sum_intra_rmse/len(all_contents)

