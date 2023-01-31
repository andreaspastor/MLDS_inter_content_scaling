import utils.solvers as solvers
import utils.data_preprocessing as dp

import numpy as np
import argparse
from time import time


if __name__ == '__main__':

    # python demo_estimation.py --nb_bootstrap 1000 --filename ./datasets/quad_intra.npz --solver mle
    # python demo_estimation.py --nb_bootstrap 1000 --filename ./datasets/triplet_intra.npz --solver mle
    # python demo_estimation.py --nb_bootstrap 1000 --filename ./datasets/pair_intra.npz --solver mle


    info = "This script is converting intra annotations over a content" + \
           "through MLDS solving to retrieve score estimates on a linear scale. \n"
    print(info)
    
    parser = argparse.ArgumentParser(description='Processing estimation information.')
    parser.add_argument("--nb_bootstrap", type=int,
                        default=1000,
                        help="an integer for the number of boostrapping")
    parser.add_argument("--filename",
                        default="./datasets/quad_intra.npz",
                        help="numpy file where the subjective judgments are stored.")
    parser.add_argument("--solver",
                        default="mle",
                        help="numpy file where the subjective judgments are stored.")
    parser.add_argument("--plot_folder",
                        default="./figs/",
                        help="numpy file where the subjective judgments are stored.")
    display_eval = True

    args = parser.parse_args()
    filename = args.filename
    n_sim_solving = args.nb_bootstrap
    solver = args.solver
    plot_folder = args.plot_folder

    #Number of pool (multiprocess) to run in parrallel for the solver
    nb_pool = 6
    
    # Load the subjective judgments of the dataset
    datas = np.load(filename, allow_pickle=True)
    equations = datas['equations'].item()

    # Display the name of the different tube-contents in the dataset
    cpt = 0
    for k in equations:
        print(cpt, k)
        cpt += 1
    
    # Solve for each of the tube-content in the dataset
    cpt = 0
    for k in equations:
        t = time()

        cpt += 1
        if cpt < 0:
            continue

        data = np.array(equations[k]).astype(np.float32)
        print(k, data, data.shape)

        savefig = True
        # solving from the dataset of annotations return mean and std
        mean_estimations, std_estimations = solvers.solve_for_intra_data(data, solver, n_sim_solving, nb_pool, plot_folder + k + f'_{len(data)}_annotations', savefig)

        ### Print information
        print("Running estimation of intra content scaling:", filename)
        print("Using solver:", solver)
        if n_sim_solving != 0:
            print("Using boostrapping over", n_sim_solving, "runs")
        print("Run in", round(time() - t, 1), "secondes")
        print("Estimated values:", mean_estimations)

        if not std_estimations is None:
            print("Standard deviation of estimated values:", std_estimations)
        print('\n')

    

    

    
