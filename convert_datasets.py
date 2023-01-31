import utils.solvers as solvers
import utils.data_preprocessing as dp
import utils.stats as stats

import numpy as np
import pandas as pd
from time import time


def extract_df_equations_quad(df_all_obs, nb_level_dist, nbContent):
    equations = {}
    all_equations = []
    source_list = []
    source_qps = {}
    for i in range(len(df_all_obs.index)):
        ligne = df_all_obs.iloc[i]
        svote = ligne["source_name_vote"]
        rvote = ligne["reversed"]
        prrint = False

        if ligne['src_name_bis'] != None: # case of the data from inter content
            splitted = ligne["source_a"].split("_")
            qpA, qpB, qpC, qpD = splitted[6], splitted[8], splitted[10], splitted[12]
            lvla, lvlb, lvlc = int(splitted[7][3:]) - 1, int(splitted[9][3:]) - 1, int(splitted[11][3:]) - 1
            if "reverse" in ligne["source_a"]:
                lvld = int(splitted[13][3:]) - 1
            else:
                lvld = int(splitted[13][3:-4]) - 1
        elif not "source_b" in df_all_obs.columns or type(ligne["source_b"]) == float:
            splitted = ligne["source_a"].split("_")
            qpA, qpB, qpC, qpD = splitted[3], splitted[5], splitted[7], splitted[9]
            lvla, lvlb, lvlc = int(splitted[4][3:]) - 1, int(splitted[6][3:]) - 1, int(splitted[8][3:]) - 1

            if "reverse" in ligne["source_a"]:
                lvld = int(splitted[10][3:]) - 1
            else:
                lvld = int(splitted[10][3:-4]) - 1
        else:# no sync case missing qp info
            if rvote:
                lvlc = int(ligne["source_a"].split('_')[-1][3:-4]) - 1
                lvld = int(ligne["source_b"].split('_')[-1][3:-4]) - 1
                lvla = int(ligne["source_c"].split('_')[-1][3:-4]) - 1
                lvlb = int(ligne["source_d"].split('_')[-1][3:-4]) - 1
            else:
                lvla = int(ligne["source_a"].split('_')[-1][3:-4]) - 1
                lvlb = int(ligne["source_b"].split('_')[-1][3:-4]) - 1
                lvlc = int(ligne["source_c"].split('_')[-1][3:-4]) - 1
                lvld = int(ligne["source_d"].split('_')[-1][3:-4]) - 1


        if ligne['src_name_bis'] == None: # avoid the inter content equations
            coeff = [0 for x in range(nb_level_dist)]
            coeff[lvla] = 1
            coeff[lvlb] = -1
            coeff[lvlc] = -1
            coeff[lvld] = 1
            eq = [svote] + coeff
            if not ligne.src_name in equations:
                equations[ligne.src_name] = [eq]
            else:
                equations[ligne.src_name].append(eq)
                
            if not ligne.src_name in source_qps:
                source_qps[ligne.src_name] = [qpA, qpB, qpC, qpD]
            else:
                qps = [qpA, qpB, qpC, qpD]
                for qp in qps:
                    if not qp in source_qps[ligne.src_name]:
                        source_qps[ligne.src_name].append(qp)


        if not ligne.src_name in source_list:
            source_list.append(ligne.src_name)
        a = source_list.index(ligne.src_name)
        b = a
        if ligne['src_name_bis'] != None:
            if not ligne.src_name_bis in source_list:
                source_list.append(ligne.src_name_bis)
            b = source_list.index(ligne.src_name_bis)

        coeffg = [0 for x in range(nb_level_dist * nbContent)]
        coeffg[a*nb_level_dist + lvla] = 1
        coeffg[a*nb_level_dist + lvlb] = -1
        coeffg[b*nb_level_dist + lvlc] = -1
        coeffg[b*nb_level_dist + lvld] = 1

        eqg = [svote] + coeffg
        all_equations.append(eqg)
    return all_equations, equations, source_list, source_qps

def extract_df_equations_pair(df_all_obs, nb_level_dist, nbContent):
    equations = {}
    all_equations = []
    source_list = []
    source_qps = {}
    for i in range(len(df_all_obs.index)):
        ligne = df_all_obs.iloc[i]
        svote = ligne["source_name_vote"]
        rvote = ligne["reversed"]
        prrint = False

        if ligne['src_name_bis'] != None: # case of the data from inter content
            splitted = ligne["source_a"].split("_")
            qpA, qpB, qpC, qpD = splitted[6], splitted[8], splitted[10], splitted[12]
            lvla, lvlb, lvlc = int(splitted[7][3:]) - 1, int(splitted[9][3:]) - 1, int(splitted[11][3:]) - 1
            if "reverse" in ligne["source_a"]:
                lvld = int(splitted[13][3:]) - 1
            else:
                lvld = int(splitted[13][3:-4]) - 1
        elif not "source_b" in df_all_obs.columns or type(ligne["source_b"]) == float:
            splitted = ligne["source_a"].split("_")
            qpA, qpB = splitted[3], splitted[5]
            lvla = int(splitted[4][3:]) - 1

            if "reverse" in ligne["source_a"]:
                lvlb = int(splitted[6][3:]) - 1
            else:
                lvlb = int(splitted[6][3:-4]) - 1

        if ligne['src_name_bis'] == None: # avoid the inter content equations
            coeff = [0 for x in range(nb_level_dist)]
            coeff[lvla] = 1
            coeff[lvlb] = -1
            eq = [svote] + coeff
            if not ligne.src_name in equations:
                equations[ligne.src_name] = [eq]
            else:
                equations[ligne.src_name].append(eq)
                
            if not ligne.src_name in source_qps:
                source_qps[ligne.src_name] = [qpA, qpB]
            else:
                qps = [qpA, qpB]
                for qp in qps:
                    if not qp in source_qps[ligne.src_name]:
                        source_qps[ligne.src_name].append(qp)


        if not ligne.src_name in source_list:
            source_list.append(ligne.src_name)
        a = source_list.index(ligne.src_name)
        b = a
        if ligne['src_name_bis'] != None:
            if not ligne.src_name_bis in source_list:
                source_list.append(ligne.src_name_bis)
            b = source_list.index(ligne.src_name_bis)

        coeffg = [0 for x in range(nb_level_dist * nbContent)]
        
        if ligne['src_name_bis'] == None:
            coeffg[a*nb_level_dist + lvla] = 1
            coeffg[a*nb_level_dist + lvlb] = -1
        else:
            coeffg[a*nb_level_dist + lvla] = 1
            coeffg[a*nb_level_dist + lvlb] = -1
            coeffg[b*nb_level_dist + lvlc] = -1
            coeffg[b*nb_level_dist + lvld] = 1

        eqg = [svote] + coeffg
        all_equations.append(eqg)
    return all_equations, equations, source_list, source_qps

def extract_df_equations(df_all_obs, nb_level_dist, nbContent):
    equations = {}
    all_equations = []
    source_list = []
    source_qps = {}
    for i in range(len(df_all_obs.index)):
        ligne = df_all_obs.iloc[i]
        svote = ligne["source_name_vote"]
        
        if ligne['src_name_bis'] == None:
            if svote == 1:
                svote = 0
            else:
                svote = 1
        
        rvote = ligne["reversed"]
        prrint = False

        if ligne['src_name_bis'] != None: # case of the data from inter content
            splitted = ligne["source_a"].split("_")
            qpA, qpB, qpC, qpD = splitted[6], splitted[8], splitted[10], splitted[12]
            lvla, lvlb, lvlc = int(splitted[7][3:]) - 1, int(splitted[9][3:]) - 1, int(splitted[11][3:]) - 1
            if "reverse" in ligne["source_a"]:
                lvld = int(splitted[13][3:]) - 1
            else:
                lvld = int(splitted[13][3:-4]) - 1
        elif not "source_b" in df_all_obs.columns or type(ligne["source_b"]) == float:
            splitted = ligne["source_a"].split("_")
            qpA, qpB, qpC = splitted[3], splitted[5], splitted[7]
            lvla, lvlb = int(splitted[4][3:]) - 1, int(splitted[6][3:]) - 1

            if "reverse" in ligne["source_a"]:
                lvlc = int(splitted[8][3:]) - 1
            else:
                lvlc = int(splitted[8][3:-4]) - 1

        if ligne['src_name_bis'] == None: # avoid the inter content equations
            coeff = [0 for x in range(nb_level_dist)]
            coeff[lvla] = 1
            coeff[lvlb] = -2
            coeff[lvlc] = 1
            eq = [svote] + coeff
            if not ligne.src_name in equations:
                equations[ligne.src_name] = [eq]
            else:
                equations[ligne.src_name].append(eq)
                
            if not ligne.src_name in source_qps:
                source_qps[ligne.src_name] = [qpA, qpB]
            else:
                qps = [qpA, qpB, qpC]
                for qp in qps:
                    if not qp in source_qps[ligne.src_name]:
                        source_qps[ligne.src_name].append(qp)


        if not ligne.src_name in source_list:
            source_list.append(ligne.src_name)
        a = source_list.index(ligne.src_name)
        b = a
        if ligne['src_name_bis'] != None:
            if not ligne.src_name_bis in source_list:
                source_list.append(ligne.src_name_bis)
            b = source_list.index(ligne.src_name_bis)

        coeffg = [0 for x in range(nb_level_dist * nbContent)]
        
        if ligne['src_name_bis'] == None:
            coeffg[a*nb_level_dist + lvla] = 1
            coeffg[a*nb_level_dist + lvlb] = -2
            coeffg[a*nb_level_dist + lvlc] = 1
        else:
            coeffg[a*nb_level_dist + lvla] = 1
            coeffg[a*nb_level_dist + lvlb] = -1
            coeffg[b*nb_level_dist + lvlc] = -1
            coeffg[b*nb_level_dist + lvld] = 1

        eqg = [svote] + coeffg
        all_equations.append(eqg)
    return all_equations, equations, source_list, source_qps


if __name__ == '__main__':
    # Open the triplet intra-content dataset
    df_all_obs_triplet = pd.read_csv("./csv_datasets/df_all_obs_triplet.csv")
    df_all_obs_triplet["src_name_bis"] = None
    df_all_obs_triplet = df_all_obs_triplet.sort_values("src_name")

    # Open the quadruplet intra-content dataset
    df_all_obs_quadruplet = pd.read_csv("./csv_datasets/df_all_obs_quadruplet.csv")
    df_all_obs_quadruplet["src_name_bis"] = None
    df_all_obs_quadruplet = df_all_obs_quadruplet.sort_values("src_name")

    # Open the pairwise intra-content dataset
    df_all_obs_pair = pd.read_csv("./csv_datasets/df_all_obs_pair.csv")
    df_all_obs_pair["src_name_bis"] = None
    df_all_obs_pair = df_all_obs_pair.sort_values("src_name")


    # Print the name of the tube-contents (SRCs)
    nameofcontent = df_all_obs_triplet.src_name.unique()
    print(nameofcontent)
    print(df_all_obs_pair.src_name.unique())

    '''
    # Extract the same
    print(len(df_all_obs_pair))
    df_all_obs_pair = df_all_obs_pair[df_all_obs_pair.src_name.isin(nameofcontent)]
    print(len(df_all_obs_pair))

    # Extract the same
    print(len(df_all_obs_quadruplet))
    df_all_obs_quadruplet = df_all_obs_quadruplet[df_all_obs_quadruplet.src_name.isin(nameofcontent)]
    print(len(df_all_obs_quadruplet))
    input()
    '''

    df_all_obs_inter8c = pd.read_csv("./csv_datasets/df_all_obs_inter8c.csv")


    df_pair = pd.concat([df_all_obs_pair, df_all_obs_inter8c])
    df_triplet = pd.concat([df_all_obs_triplet, df_all_obs_inter8c])
    df_quadruplet = pd.concat([df_all_obs_quadruplet, df_all_obs_inter8c])

    nb_level_dist = 6
    nbContent = 8

    all_equations_pair, equations_pairs, source_list, source_qps = extract_df_equations_pair(df_pair, nb_level_dist, nbContent)
    all_equations_triplet, equations, source_list, source_qps = extract_df_equations(df_triplet, nb_level_dist, nbContent)
    all_equations_quad, equations_quad, source_list, source_qps = extract_df_equations_quad(df_quadruplet, nb_level_dist, nbContent)

    content_dic = {i: i for i in source_list}

    print("pairwise dataset")
    total = 0
    for k in equations_pairs:
        print("number of annotations for content", k, len(equations_pairs[k]))
        total += len(equations_pairs[k])
    print(total)
    np.savez("./datasets/pairwise_dataset.npz", equations=equations_pairs, \
            all_equations=all_equations_pair, source_list=source_list, content_dic=content_dic)

    print("triplet dataset")
    total = 0
    for k in equations:
        print("number of annotations for content", k, len(equations[k]))
        total += len(equations[k])
    print(total)
    np.savez("./datasets/triplet_dataset.npz", equations=equations, \
            all_equations=all_equations_triplet, source_list=source_list, content_dic=content_dic)
    
    print("quadruplet dataset")
    total = 0
    for k in equations_quad:
        print("number of annotations for content", k, len(equations_quad[k]))
        total += len(equations_quad[k])
    print(total)
    np.savez("./datasets/quad_dataset.npz", equations=equations_quad, \
            all_equations=all_equations_quad, source_list=source_list, content_dic=content_dic)


