import pandas as pd
import numpy as np
import scipy.spatial.distance as distance

def filter_elements_std(elems, x, std=3):
    df_ = pd.DataFrame(elems)
    df_['Dist'] = distance.cdist(elems, [x])
    mean = df_.Dist.mean()
    dev = df_.Dist.std()
    closest = df_[df_['Dist'] < mean+std*dev]
    return closest.drop(labels=['Dist'], axis=1)

def extract_points_furthest_distance(df, num_points):

    df_copy = df.copy()
    
    if num_points > len(df):
        raise ValueError(f"num_points ({num_points}) > len(df) ({len(df)})")
    
    features = df.drop('Target', axis=1)
    dist_mat = np.tril(distance.squareform(distance.pdist(features.to_numpy())))
    # get the indices of the most distant points
    mdp_ind = [x[0] for x in [i for i in np.where(dist_mat == dist_mat.max())]]
    # extract the most distant elements and remove them from the df
    extracted = df_copy.iloc[mdp_ind]
    df_copy = df_copy.drop(labels=mdp_ind)

    while len(extracted) < num_points:
        md = df_copy.drop(labels=['Target'], axis=1).apply(lambda row: np.max(np.array([distance.euclidean(row,ex) for ex in extracted.drop(labels='Target', axis=1)])), axis=1).idxmax()
        extracted = pd.concat([extracted, df_copy.iloc[[md]]])
        df_copy = df_copy.drop(labels=[md], axis=0)
    
    return extracted