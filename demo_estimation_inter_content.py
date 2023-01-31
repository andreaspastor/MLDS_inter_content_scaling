import utils.solvers as solvers
import utils.data_preprocessing as dp
import utils.stats as stats

import numpy as np
import argparse
from time import time

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid", palette="muted")

fontsize = 16
plt.rc('font', size=fontsize)
plt.rc('ytick', labelsize=fontsize)
plt.rc('xtick', labelsize=fontsize)
plt.rc('axes', titlesize=fontsize)


def solving_main(data, source_list, source_qps, nb_distortion_levels, nb_content, solver, n_sim_solving, filename, content_dic):
    print(nb_content, nb_distortion_levels)
    data = data.astype(np.float32)
    print(data, data.shape) # (Nb judgements collected, N levels * nbcontent + 1 answer)

    print(source_list)
    print(content_dic)
    print()

    mean_estimations, std_estimations, estimations = solvers.solve_for_inter_data2(data, nb_distortion_levels, nb_content, solver, n_sim_solving)
    if not std_estimations is None: # A boostrapping has been performed
        print('Discriminability score from ttest', stats.ttest_ratio(estimations))
        print('Discriminability score from wtest', stats.wilcoxon_ratio(estimations))

    print(mean_estimations, len(mean_estimations))
    print(source_list, len(source_list))
    print([content_dic[source_list[x]] for x in range(len(source_list))])
    print([source_list[x] for x in range(len(source_list))])

    for x in range(len(source_list)):
        print(x, sum(nb_distortion_levels[:x+1]), - 2 - x + sum(nb_distortion_levels[:x+1]))

    print([str(round(mean_estimations[- 2 - x + sum(nb_distortion_levels[:x+1])], 4)) for x in range(len(source_list))])
    
    labels = [source_list[x].split('_')[-1] + " - " + str(round(mean_estimations[- 2 - x + sum(nb_distortion_levels[:x+1])], 2)) for x in range(len(source_list))]

    if not std_estimations is None: # A boostrapping has been performed
        solvers.plot_boostrap_inter_estimation2(mean_estimations, std_estimations, nb_content, nb_distortion_levels, None, labels)
    else:
        solvers.plot_single_inter_estimation2(mean_estimations, nb_content, nb_distortion_levels, labels)

    ### Print information
    print("Running estimation of inter content scaling:", filename)
    print("Dataset contains", nb_content, "contents with", nb_distortion_levels,"distortion levels each")
    print("Using solver:", solver)
    if n_sim_solving != 0:
        print("Using boostrapping over", n_sim_solving, "runs")
    runtime = time() - t
    print("Run in", round(runtime, 0), "secondes")
    print("Estimated values:", mean_estimations)

    if not std_estimations is None:
        print("Standard deviation of estimated values:", std_estimations)
        print("Mean std value", np.mean(std_estimations[:-1]))
    else:
        std_estimations = np.array([0 for x in range(len(mean_estimations))])
    
    return mean_estimations, std_estimations, runtime


if __name__ == '__main__':
    t = time()

    # python demo_estimation_inter_content.py --nb_bootstrap 100 --filename ./datasets/triplet_dataset.npz --solver mle
    # python demo_estimation_inter_content.py --nb_bootstrap 0 --filename ./datasets/triplet_dataset.npz --solver mle
    # python demo_estimation_inter_content.py --nb_bootstrap 100 --filename ./datasets/triplet_dataset.npz --solver glm
    # python demo_estimation_inter_content.py --nb_bootstrap 0 --filename ./datasets/triplet_dataset.npz --solver glm

    info = "This script is converting intra and inter annotations over a set of contents" + \
           "through our improved MLDS solving to retrieve scores on a linear scale and inter-content scaling. \n"
    print(info)
    
    parser = argparse.ArgumentParser(description='Processing estimation information.')
    parser.add_argument("--nb_bootstrap", type=int,
                        default=100,
                        help="an integer for the number of boostrapping")
    parser.add_argument("--filename",
                        default="./datasets/triplet_dataset.npz",
                        help="numpy file where the subjective judgments are stored.")
    parser.add_argument("--solver",
                        default="mle",
                        help="numpy file where the subjective judgments are stored.")
    display_eval = True

    args = parser.parse_args()
    filename = args.filename
    n_sim_solving = args.nb_bootstrap
    solver = args.solver

    # Load the subjective judgments of the dataset
    data_in_file = np.load(filename, allow_pickle=True)
    data = data_in_file['all_equations']
    print(data.shape)

    source_list = data_in_file["source_list"]
    content_dic = data_in_file["content_dic"].item()
    source_qps = {j: [i for i in range(6)] for j in source_list}
    print(source_qps, type(source_qps))
    
    nb_distortion_levels = [len(source_qps[k]) for k in source_qps]
    nb_content = len(nb_distortion_levels)

    mean_estimations, std_estimations, runtime = solving_main(data, source_list, source_qps, nb_distortion_levels, nb_content, \
                                                             solver, n_sim_solving, filename, content_dic)


        

    

    
