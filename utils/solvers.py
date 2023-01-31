import utils.data_preprocessing as dp

from multiprocessing import Pool
from scipy.special import logit, expit
from scipy.optimize import minimize
from scipy.stats import norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf

from time import time
from scipy.special import ndtr
from tqdm import tqdm

## FUNCTION FOR INTRA AND INTER CONTENT SOLVING WITH GLM
def fit_glm_once(df, N, nbC=1, link_g=sm.genmod.families.links.probit):
    """
        Function to fit a GLM on data
    """
    formula = ''
    for x in range(nbC):
        formula += ' + '.join(['x' + str(x+1) + "_" + str(i) for i in range(2,N+1)])
        if x < nbC - 1:
            formula += ' + '
    formula = 'y ~ ' + formula + ' - 1' # -1 to remove intercept

    model = smf.glm(formula=formula, data=df, family=sm.families.Binomial(link_g))
    fitted = model.fit()

    new_esti = fitted.params.values
    new_esti_bounded = new_esti / new_esti[-1]
    return new_esti, new_esti_bounded, fitted

def fit_glm_once2(df, Ns, nbC=1, link_g=sm.genmod.families.links.probit):
    """
        Function to fit a GLM on data
    """
    formula = ''
    for x in range(nbC):
        formula += ' + '.join(['x' + str(x+1) + "_" + str(i) for i in range(2,Ns[x]+1)])
        if x < nbC - 1:
            formula += ' + '
    formula = 'y ~ ' + formula + ' - 1' # -1 to remove intercept

    model = smf.glm(formula=formula, data=df, family=sm.families.Binomial(link_g))
    fitted = model.fit()

    new_esti = fitted.params.values
    new_esti_bounded = new_esti / new_esti[-1]
    return new_esti, new_esti_bounded, fitted

def multi_glm_sim(pred, data, col, N, n_sim, nbC=1, link_g=sm.genmod.families.links.probit):
    """
        Function without multi processing for GLM fitting and bootstrapping
    """
    sim_data = np.random.binomial(1, pred.values, (n_sim, len(pred.values)))
    data = np.array(data)
    estimations = []
    
    for sim in sim_data:
        data[:,0] = sim # change the answer of the trials
        df = pd.DataFrame(data=data, columns=col)
        _, new_esti_bounded, _ = fit_glm_once(df, N, nbC, link_g)
        estimations.append(new_esti_bounded)
    estimations = np.array(estimations)
    return estimations

def run_inter_glm_solving_(arg):
    """
        Function to run for multi processing for GLM fitting and bootstrapping
    """
    nbC, N, link_g, sim, data, col = arg[0], arg[1], arg[2], arg[3], arg[4], arg[5]
    data[:,0] = sim # change the answer of the trials
    df = pd.DataFrame(data=data, columns=col)

    _, new_esti_bounded, _ = fit_glm_once(df, N, nbC, link_g)
    return new_esti_bounded

def run_inter_glm_solving2_(arg):
    """
        Function to run for multi processing for GLM fitting and bootstrapping
    """
    nbC, Ns, link_g, sim, data, col = arg[0], arg[1], arg[2], arg[3], arg[4], arg[5]
    data[:,0] = sim # change the answer of the trials
    df = pd.DataFrame(data=data, columns=col)

    _, new_esti_bounded, _ = fit_glm_once2(df, Ns, nbC, link_g)
    return new_esti_bounded

def multi_glm_sim_pooled(pred, data, col, N, n_sim, nbC=1, nb_pool=5, link_g=sm.genmod.families.links.probit):
    """
        Function to call for multi processing for GLM fitting and bootstrapping
    """
    sim_data = np.random.binomial(1, pred.values, (n_sim, len(pred.values)))
    data = np.array(data)
    estimations = []
    
    with Pool(nb_pool) as p:
        estimations = p.map(run_inter_glm_solving_, [(nbC, N, link_g, sim, data, col) for sim in sim_data])

    estimations = np.array(estimations)
    return estimations

def multi_glm_sim_pooled2(pred, data, col, Ns, n_sim, nbC=1, nb_pool=5, link_g=sm.genmod.families.links.probit):
    """
        Function to call for multi processing for GLM fitting and bootstrapping
    """
    sim_data = np.random.binomial(1, pred.values, (n_sim, len(pred.values)))
    data = np.array(data)
    estimations = []
    
    with Pool(nb_pool) as p:
        estimations = p.map(run_inter_glm_solving2_, [(nbC, Ns, link_g, sim, data, col) for sim in sim_data])

    estimations = np.array(estimations)
    return estimations

## FUNCTION FOR INTER CONTENT SOLVING WITH MLE
def rosen_inter(x, d, N, nbC):
    n = [0.5 for _ in range(N*nbC)]
    n[0::N] = [0 for _ in range(nbC)]
    n[-1] = 1

    for i in range(nbC - 1):
        n[i*N+1:(i+1)*N] = x[i*(N-1):(i+1)*(N-1)] ** 2
    i = nbC - 1
    n[i*N+1:(i+1)*N - 1] = x[i*(N-1):(i+1)*(N-1) - 1] ** 2

    s = x[-1] ** 2

    del_ = np.sum(np.multiply(d[:,1:], n), 1)
    z = (del_ / s)
    p = norm.cdf(z)
    p[p < 1e-10] = 1e-10
    p[p > (1 - 1e-10)] = (1 - 1e-10)

    total = - np.sum(np.log(p[d[:,0] == 1])) - np.sum(np.log(1 - p[d[:,0] == 0]))
    return total

## FUNCTION FOR INTER CONTENT SOLVING WITH MLE
def rosen_inter2(x, d, Ns, nbC):
    nb_params = sum(Ns)
    
    n = [0 for _ in range(nb_params)]
    n[-1] = 1

    for i in range(nbC - 1):
        n[sum(Ns[:i])+1:sum(Ns[:i+1])] = x[sum([N - 1 for N in Ns[:i]]):sum([N - 1 for N in Ns[:i+1]])] ** 2
    i = nbC - 1
    
    n[sum(Ns[:i])+1:sum(Ns[:i+1]) - 1] = x[sum([N - 1 for N in Ns[:i]]):sum([N - 1 for N in Ns[:i+1]]) - 1] ** 2

    s = x[-1] ** 2
    d_1 = d[:,1:]
    del_2 = np.einsum('ij,j', d_1, n)
    z = (del_2 / s)

    p = ndtr(z)
    p[p < 1e-10] = 1e-10
    p[p > (1 - 1e-10)] = (1 - 1e-10)
    
    d_0 = d[:,0]
    p1 = p[d_0 == 1]
    p0 = 1 - p[d_0 == 0]
    total = - np.sum(np.log(p1)) - np.sum(np.log(p0))
    return total

def fit_mle_once_inter(df, N, nbC):
    n = [0.5 for _ in range((N - 1) * nbC - 1)]
    s = 0.1
    x = np.array(list(np.sqrt(n)) + [np.sqrt(s)])
    print(N, nbC, df.values.shape)
    dfv = df.values.astype(np.float32)
    res = minimize(rosen_inter, x, method='BFGS', args=(dfv, N, nbC),
                    options={'disp': True})

    params = res.x[:-1] ** 2
    sigma = res.x[-1] ** 2
    return params, sigma

def fit_mle_once_inter2(df, Ns, nbC, n=None):
    nb_params = sum([N - 1 for N in Ns])
    if n is None:
        n = [0.5 for _ in range(nb_params - 1)]
    
    assert len(n) == nb_params - 1, ("len check", len(n), nb_params - 1)

    s = 0.1
    x = np.array(list(np.sqrt(n)) + [np.sqrt(s)])
    print(Ns, nbC, df.values.shape)
    dfv = df.values.astype(np.float32)
    
    res = minimize(rosen_inter2, x, method='BFGS', args=(dfv, Ns, nbC),
                    options={'disp': True, 'maxiter': None})

    params = res.x[:-1] ** 2
    sigma = res.x[-1] ** 2
    return params, sigma


def run_inter_mle_solving_(arg):
    N, nbC, sim, data, col = arg[0], arg[1], arg[2], arg[3], arg[4]
    data[:,0] = sim # change the answer of the trials
    df = pd.DataFrame(data=data, columns=col)

    params, sigma = fit_mle_once_inter(df, N, nbC)
    return params

def run_inter_mle_solving2_(arg):
    Ns, nbC, sim, data, col = arg[0], arg[1], arg[2], arg[3], arg[4]
    data[:,0] = sim # change the answer of the trials
    df = pd.DataFrame(data=data, columns=col)
    params, sigma = fit_mle_once_inter2(df, Ns, nbC)
    return params

def multi_optim_sim_inter_pooled(pred, data, col, N, n_sim, nbC, nb_pool=5):
    sim_data = np.random.binomial(1, pred, (n_sim, len(pred)))
    data = np.array(data).astype(np.float32)

    with Pool(nb_pool) as p:
        estimations = p.map(run_inter_mle_solving_, [(N, nbC, sim, data, col) for sim in sim_data])
    estimations = np.array(estimations)
    return estimations, None

def multi_optim_sim_inter_pooled2(pred, data, col, Ns, n_sim, nbC, nb_pool=5):
    sim_data = np.random.binomial(1, pred, (n_sim, len(pred)))
    data = np.array(data).astype(np.float32)

    with Pool(nb_pool) as p:
        estimations = p.map(run_inter_mle_solving2_, [(Ns, nbC, sim, data, col) for sim in sim_data])
    estimations = np.array(estimations)
    return estimations, None

def multi_optim_sim_inter(pred, data, col, N, n_sim, nbC):
    sim_data = np.random.binomial(1, pred, (n_sim, len(pred)))
    data = np.array(data)
    estimations = []
    esti_sigma = []
    for sim in tqdm(sim_data):
        data[:,0] = sim # change the answer of the trials
        df = pd.DataFrame(data=data, columns=col)
        
        n = [0.5 for _ in range((N - 1) * nbC)]
        s = 0.1
        x = np.array(list(np.sqrt(n)) + [np.sqrt(s)])
        res = minimize(rosen_inter, x, method='BFGS', args=(df.values, N, nbC),
                       options={'disp': False})
        
        new_esti = res.x[:-1] ** 2
        estimations.append(new_esti)
        esti_sigma.append(res.x[-1] ** 2)
    estimations = np.array(estimations)
    esti_sigma = np.array(esti_sigma)
    return estimations, esti_sigma

def mle_params_inter_to_prediction(N, nbContent, params, sigma, df):
    n = [0.5 for _ in range(N*nbContent)]
    n[0::N] = [0 for _ in range(nbContent)]
    n[-1] = 1
    for i in range(nbContent - 1):
        n[i*N+1:(i+1)*N] = params[i*(N-1):(i+1)*(N-1)]
    i = nbContent - 1
    n[i*N+1:(i+1)*N - 1] = params[i*(N-1):(i+1)*(N-1) - 1]

    z = np.sum(np.multiply(df.values[:,1:], n), 1) / sigma
    pred = norm.cdf(z)
    return pred

def mle_params_inter_to_prediction2(Ns, nbContent, params, sigma, df):
    nb_params = sum(Ns)
    n = [0.5 for _ in range(nb_params)]
    for v in [sum(Ns[:i]) for i in range(len(Ns))]:
        n[v] = 0

    n[-1] = 1
    for i in range(nbContent - 1):
        n[sum(Ns[:i])+1:sum(Ns[:i+1])] = params[sum([N - 1 for N in Ns[:i]]):sum([N - 1 for N in Ns[:i+1]])]
    i = nbContent - 1
    n[sum(Ns[:i])+1:sum(Ns[:i+1]) - 1] = params[sum([N - 1 for N in Ns[:i]]):sum([N - 1 for N in Ns[:i+1]]) - 1]

    z = np.sum(np.multiply(df.values[:,1:], n), 1) / sigma
    pred = norm.cdf(z)
    return pred

## FUNCTION FOR INTRA CONTENT SOLVING WITH MLE
def rosen(x, d):
    """
        Function to estimate the numerical values of the stimuli in intra estimation
    """
    nlen = len(x)
    n = [0 for _ in range(nlen + 1)]
    n[0] = 0
    n[nlen] = 1

    n[1:-1] = expit(x[[i for i in range(0,nlen-1)]])
    s = np.exp(x[-1])
    del_ = np.sum(np.multiply(d[:,1:], n), 1)
    z = del_ / s
    p = norm.cdf(z)
    p[p < 1e-10] = 1e-10
    p[p > (1 - 1e-10)] = (1 - 1e-10)

    total = - np.sum(np.log(p[d[:,0] == 1])) - np.sum(np.log(1 - p[d[:,0] == 0]))
    return total

def fit_mle_once_intra(df, N):
    n = [0.5 for x in range(N - 2)]
    s = 0.1
    x = np.array(list(logit(n)) + [np.log(s)])
    res = minimize(rosen, x, method='BFGS', args=(df.values,),
                   options={'disp': False})
    params = expit(res.x[:-1])
    sigma = np.exp(res.x[-1])
    return params, sigma
    
def run_intra_mle_solving_(arg):
    """
        Function to run the mle soving when using multiprocessing
        prepare the data to then call the estimation function
    """
    N, sim, data, col = arg[0], arg[1], arg[2], arg[3]
    data[:,0] = sim # change the answer of the trials
    df = pd.DataFrame(data=data, columns=col)

    params, sigma = fit_mle_once_intra(df, N)
    new_esti = np.array(list(params) + [1])
    return new_esti

def multi_optim_sim_intra_pooled(pred, data, col, N, n_sim, nb_pool=5):
    """
        Function to run WITH multiprocessing the estimation of an intra curve
        with boostrapping
    """
    sim_data = np.random.binomial(1, pred, (n_sim, len(pred)))
    data = np.array(data)

    with Pool(nb_pool) as p:
        estimations = p.map(run_intra_mle_solving_, [(N, sim, data, col) for sim in sim_data])
    
    estimations = np.array(estimations)
    return estimations, None

def multi_optim_sim_intra(pred, data, col, N, n_sim):
    """
        Function to run WITHOUT multi processing the estimation of an intra curve
        with boostrapping
    """
    sim_data = np.random.binomial(1, pred, (n_sim, len(pred)))
    data = np.array(data)
    estimations = []
    esti_sigma = []
    
    for sim in sim_data:
        data[:,0] = sim # change the answer of the trials
        df = pd.DataFrame(data=data, columns=col)
        
        n = [0.5 for _ in range(N - 2)]
        s = 0.1
        x = np.array(list(logit(n)) + [np.log(s)])
        res = minimize(rosen, x, method='BFGS', args=(df.values,),
                       options={'disp': False})
        
        new_esti = np.array(list(expit(res.x[:-1])) + [1])
        estimations.append(new_esti)
        esti_sigma.append(np.exp(res.x[-1]))
    estimations = np.array(estimations)
    esti_sigma = np.array(esti_sigma)
    return estimations, esti_sigma

def mle_params_intra_to_prediction(params, sigma, df):
    n = np.array([0] + list(params) + [1])
    z = np.sum(np.multiply(df.values[:,1:], n), 1) / sigma
    pred = norm.cdf(z)
    return pred

### Solving on real data in intra
def plot_single_intra_estimation(new_esti, label=None):
    plt.figure(figsize=(9,6))
    xticks = [i for i in range(len(new_esti))]
    plt.errorbar(xticks, new_esti, yerr=np.zeros(new_esti.shape), label=label, capsize=5, ls='--')
    plt.xlabel("Distortion levels")
    plt.ylabel("Estimation of perception")
    if label:
        plt.legend()
    plt.show()
    return None

def plot_boostrap_intra_estimation(mean_estimations, std_estimations, label=None, savefig=False):
    plt.figure(figsize=(9,6))
    xticks = [i for i in range(len(mean_estimations))]
    plt.errorbar(xticks, mean_estimations, yerr=std_estimations, label=label, capsize=5, ls='--')
    plt.xlabel("Distortion levels")
    plt.ylabel("Estimation of perception")
    if label:
        plt.legend()
        if savefig:
            plt.savefig(label + '.png')
            return None
    plt.show()
    return None

def select_solver(solver):
    if solver == "glm":
        GLM = True
    elif solver == "mle":
        GLM = False
    else:
        quit("solver not recognized: " + solver)
    return GLM

def solve_for_intra_data(data, solver, n_sim_solving, nb_pool=5, label=None, savefig=False):
    N = data.shape[1] - 1 # number of distortion levels
    df, col = dp.data_to_dataframe(data, N)

    if label is None:
        label = solver + " estimation"

    GLM = select_solver(solver)
    # First estimation
    if GLM:
        new_esti, new_esti_bounded, fitted = fit_glm_once(df, N)
        new_esti = np.array([0] + list(new_esti_bounded))
    else:
        params, sigma = fit_mle_once_intra(df, N)
        new_esti = np.array([0] + list(params) + [1])

    if n_sim_solving == 0:
        plot_single_intra_estimation(new_esti, label)
        return new_esti, None

    if GLM:
        pred = fitted.predict(df)
        estimations = multi_glm_sim_pooled(pred, data, col, N, n_sim_solving)
    else:
        pred = mle_params_intra_to_prediction(params, sigma, df)
        estimations, esti_sigma = multi_optim_sim_intra_pooled(pred, data, col, N, n_sim_solving, nb_pool)
    mean_estimations = np.mean(estimations, axis=0)
    mean_estimations = np.array([0] + list(mean_estimations))
    std_estimations = np.std(estimations, axis=0)
    std_estimations = np.array([0] + list(std_estimations))
    plot_boostrap_intra_estimation(mean_estimations, std_estimations, label, savefig)
    return mean_estimations, std_estimations

### Solving on real data in inter
def plot_single_inter_estimation(new_esti, nbContent, N, label):
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    markers = ["o", "v", "s", "^", "D", "d", "x", "<", ">", "1", "2", "3", "4", "p", "h", "X", 4, 5, 6, 7, 8, 9, 10, 11]

    plt.figure(figsize=(9,6))
    for x in range(nbContent):
        estimated_values = [0] + list(new_esti[x*(N-1):(x+1)*(N-1)])
        std_err = [0 for _ in range(N)]
        index = [x for x in range(N)]
        if label:
            plt.errorbar(index, estimated_values, yerr=std_err, ls='--', marker=markers[(x//len(colors))%len(markers)], capsize=5, capthick=1, label=label[x], ecolor=colors[x%len(colors)])
        else:
            plt.errorbar(index, estimated_values, yerr=std_err, ls='--', marker=markers[(x//len(colors))%len(markers)], capsize=5, capthick=1, ecolor=colors[x%len(colors)])
    plt.xlabel("Distortion levels")
    plt.ylabel("Estimation of perception")
    if label:
        plt.legend(loc=2, ncol=len(label)//40 + 1, fontsize="small")
    plt.show()
    return None

def plot_single_inter_estimation2(new_esti, nbContent, Ns, label):
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    markers = ["o", "v", "s", "^", "D", "d", "x", "<", ">", "1", "2", "3", "4", "p", "h", "X", 4, 5, 6, 7, 8, 9, 10, 11]

    plt.figure(figsize=(9,6))
    for x in range(nbContent):
        posN = sum([u - 1 for u in Ns[:x]])
        posN1 = sum([u - 1 for u in Ns[:x+1]])
        #print(posN, posN1)
        estimated_values = [0] + list(new_esti[posN:posN1])
        std_err = [0 for _ in range(Ns[x])]
        index = [u for u in range(Ns[x])]
        if label:
            plt.errorbar(index, estimated_values, yerr=std_err, ls='--', marker=markers[(x//len(colors))%len(markers)], capsize=5, capthick=1, label=label[x], ecolor=colors[x%len(colors)])
        else:
            plt.errorbar(index, estimated_values, yerr=std_err, ls='--', marker=markers[(x//len(colors))%len(markers)], capsize=5, capthick=1, ecolor=colors[x%len(colors)])
    plt.xlabel("Distortion levels")
    plt.ylabel("Estimation of perception")
    if label:
        plt.legend(loc=2, ncol=len(label)//40 + 1, fontsize="small")
    plt.show()
    return None

def plot_boostrap_inter_estimation(mean_estimations, std_estimations, nbContent, N, max_level, label=None):
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    markers = ["o", "v", "s", "^", "D", "d", "x", "<", ">", "1", "2", "3", "4", "p", "h", "X", 4, 5, 6, 7, 8, 9, 10, 11]

    plt.figure(figsize=(9,6))
    for x in range(nbContent):
        estimated_values = [0] + list(mean_estimations[x*(N-1):(x+1)*(N-1)])
        std_err = np.array([0] + list(std_estimations[x*(N-1):(x+1)*(N-1)]))
        index = [x for x in range(N)]
        if label:
            plt.errorbar(index[:max_level], estimated_values[:max_level], yerr=std_err[:max_level], ls='--', marker=markers[(x//len(colors))%len(markers)], capsize=5, capthick=1, label=label[x], ecolor=colors[x%len(colors)])
        else:
            plt.errorbar(index[:max_level], estimated_values[:max_level], yerr=std_err[:max_level], ls='--', marker=markers[(x//len(colors))%len(markers)], capsize=5, capthick=1, ecolor=colors[x%len(colors)])#, ecolor='black')
    plt.xlabel("Distortion levels")
    plt.ylabel("Estimation of perception")
    if label:
        plt.legend(loc=2, ncol=len(label)//40 + 1, fontsize="small")
    plt.show()
    return None

def plot_boostrap_inter_estimation2(mean_estimations, std_estimations, nbContent, Ns, max_level, label=None):
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    markers = ["o", "v", "s", "^", "D", "d", "x", "<", ">", "1", "2", "3", "4", "p", "h", "X", 4, 5, 6, 7, 8, 9, 10, 11]
    fontsize = 16

    plt.figure(figsize=(9,6))
    for x in range(nbContent):
        posN = sum([u - 1 for u in Ns[:x]])
        posN1 = sum([u - 1 for u in Ns[:x+1]])
        #print(posN, posN1)
        estimated_values = [0] + list(mean_estimations[posN:posN1])
        std_err = np.array([0] + list(std_estimations[posN:posN1]))
        index = [u for u in range(Ns[x])]
        if label:
            plt.errorbar(index[:max_level], estimated_values[:max_level], yerr=std_err[:max_level], ls='--', marker=markers[(x//len(colors))%len(markers)], capsize=5, capthick=1, label=label[x], ecolor=colors[x%len(colors)])
        else:
            plt.errorbar(index[:max_level], estimated_values[:max_level], yerr=std_err[:max_level], ls='--', marker=markers[(x//len(colors))%len(markers)], capsize=5, capthick=1, ecolor=colors[x%len(colors)])#, ecolor='black')
    plt.xlabel("Distortion levels", fontsize=fontsize+4)
    plt.ylabel("Estimation of perception", fontsize=fontsize+4)
    if label:
        plt.legend(loc=2, ncol=len(label)//40 + 1, fontsize=fontsize)

    plt.show()
    return None

def solve_for_inter_data(data, N, nbContent, solver, n_sim_solving):
    df, col = dp.data_to_dataframe(data, N, nbC=nbContent)

    label = None#solver + " estimation"

    GLM = select_solver(solver)
    # First estimation
    if GLM:
        new_esti, new_esti_bounded, fitted = fit_glm_once(df, N, nbContent)
    else:
        params, sigma = fit_mle_once_inter(df, N, nbContent)
        new_esti = np.array(list(params) + [1])

    print(new_esti)
    if n_sim_solving == 0:
        #plot_single_inter_estimation(new_esti, nbContent, N, label)
        return new_esti, None

    if GLM:
        pred = fitted.predict(df)
        estimations = multi_glm_sim_pooled(pred, data, col, N, n_sim_solving, nbContent)
        mean_estimations = np.mean(estimations, axis=0)
        std_estimations = np.std(estimations, axis=0)
    else:
        pred = mle_params_inter_to_prediction(N, nbContent, params, sigma, df)
        estimations, esti_sigma = multi_optim_sim_inter_pooled(pred, data, col, N, n_sim_solving, nbContent)
        mean_estimations = np.array(list(np.mean(estimations, axis=0)) + [1])
        std_estimations = np.array(list(np.std(estimations, axis=0)) + [0])
    
    #plot_boostrap_inter_estimation(mean_estimations, std_estimations, nbContent, N, label)
    return mean_estimations, std_estimations


current_params = None
def solve_for_inter_data2(data, Ns, nbContent, solver, n_sim_solving, plot=False):
    df, col = dp.data_to_dataframe2(data, Ns, nbC=nbContent)

    label = None#solver + " estimation"

    GLM = select_solver(solver)
    # First estimation
    if GLM:
        new_esti, new_esti_bounded, fitted = fit_glm_once2(df, Ns, nbContent)
        #new_esti = np.array([0] + list(new_esti_bounded))
    else:
        t = time()
        params, sigma = fit_mle_once_inter2(df, Ns, nbContent, current_params)
        #current_params = params
        new_esti = np.array(list(params) + [1])

    if n_sim_solving == 0:
        if plot:
            plot_single_inter_estimation2(new_esti, nbContent, Ns, label)
        return new_esti, None, None

    if GLM:
        print("start multi glm solving")
        pred = fitted.predict(df)
        estimations = multi_glm_sim_pooled2(pred, data, col, Ns, n_sim_solving, nbContent)
        mean_estimations = np.mean(estimations, axis=0)
        std_estimations = np.std(estimations, axis=0)
    else:
        pred = mle_params_inter_to_prediction2(Ns, nbContent, params, sigma, df)
        estimations, esti_sigma = multi_optim_sim_inter_pooled2(pred, data, col, Ns, n_sim_solving, nbContent)
        mean_estimations = np.array(list(np.mean(estimations, axis=0)) + [1])
        std_estimations = np.array(list(np.std(estimations, axis=0)) + [0])
    
    #plot_boostrap_inter_estimation2(mean_estimations, std_estimations, nbContent, Ns, None, label)
    return mean_estimations, std_estimations, estimations


