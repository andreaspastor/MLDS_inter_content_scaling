import pandas as pd


def data_to_dataframe(data, N, nbC=1):
    """ 
        Function to store in a dataframe the equations and judgements of observers
        Needed for GLM solving
    """
    col = ["y"]

    for x in range(nbC):
        col += ['x' + str(x+1) + "_" + str(i) for i in range(1,N+1)]
    df = pd.DataFrame(data=data, columns=col)
    return df, col

def data_to_dataframe2(data, Ns, nbC=1):
    """ 
        Function to store in a dataframe the equations and judgements of observers
        Needed for GLM solving
    """
    col = ["y"]

    for x in range(nbC):
        col += ['x' + str(x+1) + "_" + str(i) for i in range(1,Ns[x]+1)]
    df = pd.DataFrame(data=data, columns=col)
    return df, col