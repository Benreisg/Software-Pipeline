from pm4py.objects.conversion.log import converter as log_converter
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

""" not working - change in traces needed"""

def calculate_helper_data_structures(df, trace_id, event_id, resource,
                                     activity, timestamp):
    """
    Calculates data structures to create a weak order matrix which will be used to create behaviour profiles
    afterward.

    Parameters
    ----------
    df:         pandas DataFrame
    trace_id:   column name of the trace id
    event_id:   column name of the event id
    resource:   column name of the resources
    activity:   column name of the activities
    timestamp:  column name of the timestamps

    Returns
    -------
    Variables used to create a weak order matrix:
    df, trace_id, event_id, resource, activity, timestamp, group_by_case, aToi, iToa, uToi, iTou

    """

    group_by_case = df.sort_values([trace_id, event_id], ascending=True).groupby(trace_id)
    aToi = {j: i for (i, j) in enumerate(df[activity].drop_duplicates())}
    iToa = {j: i for i, j in aToi.items()}
    uToi = {j: i for (i, j) in enumerate(df[resource].drop_duplicates())}
    iTou = {j: i for i, j in uToi.items()}
    # activity_value_counts=log[activity].value_counts().to_dict()
    # resource_value_counts=log[resource].value_counts().to_dict()

    return trace_id, event_id, resource, activity, timestamp, group_by_case, \
        aToi, iToa, uToi, iTou

def _behaviour_matrix_heatmap_plot(df):
    """
    Creates behaviour matrix heatmap

    Parameters
    ----------
    df: pandas DataFrame

    Returns
    -------
    Behaviour matrix in heatmap format

    """

    f, ax = plt.subplots(figsize=(14, 12))
    #    mask = np.zeros_like(df, dtype=np.bool)
    #    mask[np.triu_indices_from(mask)] = True
    return sns.heatmap(df,
                       linewidths=.5,
                       fmt="",
                       # mask=mask,
                       cbar=False,
                       annot=df.replace([0, 1, 2, 3, 4], ['-->', "<--", "||", "+", ""]),
                       cmap=sns.cubehelix_palette(8, rot=0, start=2.8, light=0.982, as_cmap=True)
                       )

def _matrix_heatmap_plot(df):
    """
    Creates matrix heatmap

    Parameters
    ----------
    df: pandas DataFrame

    Returns
    -------
    Matrix in heatmap format

    """

    # f, ax = plt.subplots(figsize=(14, 12))
    return sns.heatmap(df + 1,
                       linewidths=.5,
                       vmin=1,
                       annot=df, annot_kws={"size": 8},
                       norm=LogNorm(vmin=1, vmax=df.max().max()),
                       cbar=False,
                       cmap=sns.cubehelix_palette(8, rot=0, start=2.8, light=0.982, as_cmap=True))

def weak_order_matrix(activity, aToi, iToa, resource, uToi, iTou, group_by_case, grouped_by_activity=True,
                      as_df=False, as_plot=True):
    """

    Calculates the weak order matrix for a pandas groupby (normally grouped by case)

    Parameters
    ----------
    The following arguments were created in calculate_helper_data_structures and are used in order to create a weak
    order matrix.

    activity:               column name of the resources
    aToi:                   dict with activities
    iToa:                   same dict with activities, but keys and values are reversed
    resource:               column name of the resources
    uToi:                   dict with resources
    iTou:                   same dict with resources, but keys and values are reversed
    group_by_case:          DataFrame grouped by case
    grouped_by_activity:    if True (default): activities will be used to create weak order matrix
                            if False: resources will be used to create weak order matrix
    as_df:                  if as_df: returns DataFrame (default value = False)
    as_plot:                if as_plot: returns matrix in heatmap format (default value = True)

    Returns
    -------
    Pandas DataFrame or matrix in heatmap format (depends on values of as_df and as_plot)

    if both values (as_df and as_plot) are False, the created array F will be returned

    """
    if grouped_by_activity:
        c = activity
        d = aToi
        cols = [iToa[i] for i in range(len(iToa))]
    else:
        c = resource
        d = uToi
        cols = [iTou[i] for i in range(len(iTou))]

    F = np.zeros(shape=(len(cols), len(cols)))
    for name, group in group_by_case:
        activities = group[c].values
        for i in range(0, len(activities) - 1):
            for j in range(i + 1, len(activities)):
                F[d[activities[i]], d[activities[j]]] += 1

    if as_df:
        return pd.DataFrame(F, columns=cols, index=cols)
    if as_plot:
        return _matrix_heatmap_plot(pd.DataFrame(F, columns=cols, index=cols))

    return F

def behavioural_profiles(log, trace_id='case:concept:name', resource='org:resource',
                         activity='concept:name', timestamp='time:timestamp', grouped_by_activity=True,
                         as_df=False, as_plot=True):
    """
    Creates Behavioural Profiles of an Event Log or DataFrame
    Based on https://doi.org/10.1007/978-3-642-13094-6_38.

    Parameters
    ----------
    log:                    EventLog or Dataframe
    trace_id:               str
                            trace id of Event Log/Dataframe ("case:concept:name" as default value)
    resource:               str
                            column name of the resources ("org:resource" as default value)
    activity:               str
                            column name of the activities ("concept:name" as default value)
    timestamp:              str
                            column name of the timestamps ("time:timestamp" as default value)
    grouped_by_activity:    bool
                            if True (default): activities will be used to create behaviour profiles
                            if False: resources will be used to create behaviour profiles
    as_df:                  bool
                            if as_df: returns DataFrame (default value = False)
    as_plot:                bool
                            if as_plot: returns matrix in heatmap format (default value = True)



    Returns
    -------
    DataFrame or matrix in heatmap format (depends on values as_df and as_plot)

    if both values (as_df and as_plot) are False, the calculated sets strict_order, reverse_strict_order,
    exclusive, interleaving will be returned

    """

    """
    checking if log has type DataFrame (if not: applying log_converter)
    """

    df = log

    if not isinstance(log, pd.DataFrame):
        df = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)

    if df.empty:
        raise ValueError("Dataframe is empty")

    """
    using calculate_helper_data_structures with default values to create weak order matrix in the next step
    """

    df['event:id'] = df.groupby(trace_id).cumcount()

    trace_id, event_id, resource, activity, timestamp, group_by_case, aToi, iToa, uToi, iTou \
        = calculate_helper_data_structures(df, trace_id=trace_id, event_id="event:id", resource=resource,
                                           activity=activity, timestamp=timestamp)

    wom = weak_order_matrix(activity, aToi, iToa, resource, uToi, iTou, group_by_case,
                            grouped_by_activity=grouped_by_activity, as_df=True)

    """
    creating behaviour profiles
    """

    cols = wom.columns
    wom = wom.values
    l = len(wom)
    res = np.empty((l, l), dtype=float)
    strict_order, reverse_strict_order, exclusive, interleaving = set(), set(), set(), set()
    for i in range(l):
        for j in range(l):
            if i == j:  res[i, j] = 4; continue
            if wom[i, j] == 0 and wom[j, i] != 0: res[j, i] = 0; strict_order.add((cols[i], cols[j])); continue
            if wom[j, i] == 0 and wom[i, j] != 0: res[j, i] = 1; reverse_strict_order.add((cols[i], cols[j])); continue
            if wom[i, j] != 0 and wom[j, i] != 0: res[j, i] = 2; interleaving.add((cols[i], cols[j])); continue
            if wom[i, j] == 0 and wom[j, i] == 0: res[j, i] = 3; exclusive.add((cols[i], cols[j])); continue

    if as_df:
        return pd.DataFrame(res, columns=cols, index=cols)
    if as_plot:
        return _behaviour_matrix_heatmap_plot(pd.DataFrame(res, columns=cols, index=cols))



    return strict_order, reverse_strict_order, exclusive, interleaving
