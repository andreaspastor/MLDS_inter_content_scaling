import numpy as np
from tqdm import tqdm

def generate_valid_inter_quadruplets(N, inter_quad_mode=2):
    if inter_quad_mode == 1:
        quad = [[x,y] for x in range(N) for y in range(x+1,N)]
    elif inter_quad_mode == 9:
        quad = [[0,y] for y in range(1,N)]
        quad += [[y-1,y] for y in range(2,N)] #0-1 already in
    elif inter_quad_mode in [2, 4]:
        quad = [[0,y] for y in range(1,N)]
    elif inter_quad_mode in [5]:
        quad = [[0,N-1]]
    elif inter_quad_mode in [6]:
        quad = [[0,1]]
    elif inter_quad_mode in [7]:
        quad = [[0,N//2], [0,N-1]]
    elif inter_quad_mode in [8]:
        quad = [[0,N-2], [0,N-1]]
    elif inter_quad_mode == 3:
        quad = [[y-1,y] for y in range(1,N)]
    else:
        print("inter quad mode", inter_quad_mode, "not supported")
        quit()
    return quad

def params_to_estimates(nbC, Ns, mean_estimations, std_estimations):
    all_estims = []
    all_estims_std = []
    for x in range(nbC):
        posN = sum([u - 1 for u in Ns[:x]])
        posN1 = sum([u - 1 for u in Ns[:x+1]])

        estimated_values = [0] + list(mean_estimations[posN:posN1])
        estimated_std_values = [0] + list(std_estimations[posN:posN1])
        all_estims.append(estimated_values)
        all_estims_std.append(estimated_std_values)
    all_estims = np.array(all_estims)
    all_estims_std = np.array(all_estims_std)
    return all_estims, all_estims_std

def tqdm2(x):
    return x

def compute_AFAD_R_in_quadruplets(quad_inters, all_estims):
    all_dists = []
    tqdm = tqdm2
    for va in tqdm(range(len(all_estims))):
        quad_interA = quad_inters[va]
        for vb in tqdm(range(va + 1, len(all_estims))):
            quad_interB = quad_inters[vb]
            for comp1 in quad_interA:
                for comp2 in quad_interB:
                    dist_1 = abs(all_estims[va][comp1[0]] - all_estims[va][comp1[1]])
                    dist_2 = abs(all_estims[vb][comp2[0]] - all_estims[vb][comp2[1]])
                    dist = abs(dist_1 - dist_2)
                    dist /= (dist_1 + dist_2 + 1e-6) # AFAD_R     
                    all_dists.append([dist, comp1, comp2, va, vb])
    return all_dists
