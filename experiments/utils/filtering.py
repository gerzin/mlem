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