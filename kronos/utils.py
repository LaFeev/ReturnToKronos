"""
Utility functions for the 'Return to Kronos' analysis, VAST 2016 MC2
Author: Aaron LaFevers
Date: 18-AUG-2023
"""
import pandas as pd
import numpy as np
from datetime import datetime, time, date, timedelta
import re
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import cluster
from sklearn.preprocessing import StandardScaler

def coordToZone(floor, x, y, zone_map):
    # retuns the zone (according to zone_map) corresponding to the x,y coordinates on the given floor
    fringe = 0
    while fringe < 3:
        for i, row in zone_map[zone_map.floor == floor].iterrows():
            if (x > row.left - fringe) & (x <= row.right):
                if (y > row.bottom - fringe) & (y <= row.top):
                    try:
                        return int(row.zone)
                    except ValueError:
                        return row.zone
        # in case the boundaries of the zones do not meet perfectly, we will expand each by a foot left and down
        fringe += 1


def getCentroid(left, top, right, bottom):
    # returns the x,y integer coordinates of the centroid of a rectangle, given the x-values left/right, and y-values top/bottom
    return round(np.mean([left, right])), round(np.mean([top, bottom]))


def officeToZone(office, floor_map, zone_map):
    row = floor_map[floor_map.office == str(office)]
    x, y = getCentroid(row.left, row.top, row.right, row.bottom)
    return coordToZone(int(row.floor), x, y, zone_map)


def floorToZone(row, zone_map):
    x, y = getCentroid(row.left, row.top, row.right, row.bottom)
    return coordToZone(int(row.floor), x, y, zone_map)


def interpolateProx(prox_full, proxIds):
    # create a blank DF with a datetime index and proxIds for columns.  The values will be a floor-zone categorical reference
    window = 14  # days
    timestamps = [datetime(2016, 5, 31, 0, 0, 0) + timedelta(minutes=n) for n in range(1440 * window)]
    loc = pd.DataFrame(-1, index=timestamps, columns=proxIds.index)

    # first need to sort the prox data DF by timestamp
    prox_full.sort_values('timestamp', inplace=True)
    # group the DF by proxId
    grouped = prox_full.groupby('proxId', axis=0)

    # loop thru each proxId
    for pid, group in grouped:
        # now group the proxId-grouped DF further, by day (requires timestamp to be the index)
        day_grouped = group.set_index('timestamp').groupby(lambda x: x.date, axis=0)
        first_entry = True
        # loop thru each day now
        for day, subgroup in day_grouped:
            # now loop thru every time entry for this employee on this day
            last_dt = 0
            prev_loc = -1
            for dt, row in subgroup.iterrows():
                cur_dt = dt.replace(second=0)
                # hash the location specified by current entry
                cur_loc = str(row.floor) + "_" + str(row.zone)

                if last_dt == 0:
                    if not first_entry:
                        # first entry of the day, but not the first day in the DF
                        ##### weekend logic!
                        # find the last prox entry for this employee, could have been days ago
                        tmp = group.timestamp[group.timestamp < dt].max()
                        last_dt = tmp.replace(second=0)
                        prev_loc = loc.loc[last_dt, pid]
                    else:
                        # first entry ever for this proxId
                        first_entry = False
                        # reset last_dt to the first day in the query window
                        last_dt = timestamps[0]

                if (cur_loc == '1_1') & (prev_loc == '1_1'):
                    # likely reentering the building
                    if cur_dt - last_dt > timedelta(minutes=1):
                        # only bother if there is more than 1 minute since last_time
                        # fill intervening time with the out-of-office location: '0_0'
                        last_dt_plus_1 = last_dt + timedelta(minutes=1)
                        loc.loc[last_dt_plus_1:cur_dt, pid] = '0_0'
                else:
                    # fill in time since last entry with prev_loc
                    loc.loc[last_dt:cur_dt, pid] = prev_loc

                # now set the current time entry to the cur_loc
                loc.loc[cur_dt, pid] = cur_loc

                # set passforward variables
                prev_loc = cur_loc
                last_dt = cur_dt

    return loc


def parseBuildingCol(col):
    # this function is designed to parse the column names of the building data DF to
    #  pull out floor, zone, and node/sensor names for use in a multi-index

    reg1 = re.compile(r'F_(\d)_(\w+):?\s?(.+?)$')
    reg2 = re.compile(r'F_(\d):?\s?(.+?)$')
    regZone = re.compile(r'Z_(\w+)')

    res1 = reg1.findall(col)
    if len(res1) == 0:
        # did not match the F_#_Z_#: format
        # Look for the F_#: format next
        res2 = reg2.findall(col)
        if len(res2) == 0:
            # did not match the F_#: format
            # assume it is a general building sensor
            floor = 0
            zone = '0'
            node = col
        else:
            floor = int(res2[0][0])
            zone = '0'
            node = res2[0][1]
    else:
        floor = int(res1[0][0])
        node = res1[0][2]
        resZone = regZone.findall(res1[0][1])
        if len(resZone) == 0:
            # zone is not of format Z_#
            zone = res1[0][1]
        else:
            zone = resZone[0]

    return (floor, zone, node)


def proxTimeSeriesViz(df_stack, title, cat_orders=None, height=None, cmap=None,
                      facet_by=None, horiz_rect_df=None, horiz_rect_by=None,
                      vert_rect_df=None, vert_rect_by=None, x_range=["2016-05-29", "2016-06-14 5:00:00"]):
    """
    must input a tall DF with timeseries index, a "proxId" and "location" column, and optionally
    a column whose name matches <facet_by>.
    """
    df_stack = df_stack.reset_index(names='datetime')

    colors = px.colors.qualitative.Plotly
    dept_cmap = {'Facilities': 9, 'Executive': 1, 'Engineering': 2, 'Administration': 3, 'IT': 4, 'HR': 6,
                 'Security': 7}

    if not cat_orders:
        cat_orders = {'proxId': df_stack.proxId.sort_values().unique(),
                      'location': ['office', 'building', 'away', 'undefined']}
    else:
        if (facet_by is not None) & (facet_by not in cat_orders):
            cat_orders[facet_by] = list(range(len(df_stack[facet_by].unique())))

    if not cmap:
        cmap = {'undefined': colors[3], 'away': colors[2], 'building': colors[0], 'office': colors[5]}

    if len(df_stack.proxId.unique()) <= 20:
        marker_size = 10
    elif len(df_stack.proxId.unique()) <= 200:
        marker_size = 5
    else:
        marker_size = 2

    fig = px.scatter(df_stack, x='datetime', y='proxId', color='location',
                     category_orders=cat_orders,
                     color_discrete_map=cmap,
                     title=title,
                     height=height,
                     facet_row=facet_by,
                     facet_row_spacing=0.02
                     )

    if facet_by:
        fig.update_yaxes(matches=None, tickfont=dict(size=10))

    fig.update_traces(
        marker=dict(size=marker_size, symbol="line-ns-open"),
        selector=dict(mode="markers"),
    )

    # horizontal rectangles
    if horiz_rect_df is not None:
        if facet_by:
            # Plotly seems to put the first facet group in row 0, but then the last facet group is in row 1, next to
            #  last in row 2, etc etc.  Assuming this behavior holds and the facet_by variable is simply sorted,
            #  this row assignment logic should work...
            facets = df_stack[facet_by].sort_values().unique()
            num_facets = len(facets)
            facet_row_map = {facets[0]: 0}
            facet_row_map.update({facets[n]: num_facets - n for n in range(1, num_facets)})
            for i, facet in enumerate(facets):
                facet_group = horiz_rect_df[horiz_rect_df[facet_by] == facet]
                for d, group in facet_group.groupby(horiz_rect_by):
                    p = group.index.sort_values()
                    if horiz_rect_by == 'dept':
                        c = colors[dept_cmap[d]]
                    else:
                        c = colors[list(horiz_rect_df[horiz_rect_by].unique()).index(d)]
                    fig.add_hrect(
                        y0=p[0],
                        y1=p[-1],
                        row=facet_row_map[facet],
                        label=dict(
                            text=f"{d}",
                            textposition="middle left",
                            font=dict(size=12),
                        ),
                        fillcolor=c,
                        opacity=0.2,
                        line_width=0,
                    )

        else:
            for d, group in horiz_rect_df.groupby(horiz_rect_by):
                p = group.index.sort_values()
                if horiz_rect_by == 'dept':
                    c = colors[dept_cmap[d]]
                else:
                    c = colors[list(horiz_rect_df[horiz_rect_by].unique()).index(d)]
                fig.add_hrect(
                    y0=p[0],
                    y1=p[-1],
                    label=dict(
                        text=f"{d}",
                        textposition="middle left",
                        font=dict(size=12),
                    ),
                    fillcolor=c,
                    opacity=0.2,
                    line_width=0,
                )

    # vertical rectangles
    if type(vert_rect_df) != type(None):
        # add vertical rectangles over the supplied activity windows
        for i, row in vert_rect_df.iterrows():
            if vert_rect_by is None:
                c = colors[6]
                l = None
                t = f"{vert_rect_df.loc[i, 'cluster']}"
            else:
                c = colors[list(vert_rect_df[vert_rect_by].unique()).index(row[vert_rect_by])]
                l = row[vert_rect_by]
                t = f"{vert_rect_df.loc[i, 'cluster']} by {vert_rect_by}"
            fig.add_vrect(
                x0=row.t_start,
                x1=row.t_end,
                label=dict(
                    text=l,
                    textposition="top center",
                    font=dict(size=10),
                ),
                fillcolor=c,
                opacity=0.2,
                line_width=0,
            )
        i = vert_rect_df.index[0]
        fig.add_annotation(x=vert_rect_df.loc[i, 't_start'],
                           y=1.01,
                           yref="paper",
                           text=t,
                           showarrow=False,
                           )
    fig.for_each_trace(
        lambda trace: trace.update(visible="legendonly") if trace.name in ['away', 'undefined'] else (),
    )
    fig.update_xaxes(range=x_range)
    # fig.show()
    return fig


def bldgTimeSeriesViz(df_stack, title, cat_orders=None, height=None, cmap=None,
                      facet_by=None, horiz_rect_df=None, horiz_rect_by=None,
                      vert_rect_df=None, vert_rect_by=None, x_range=None):
    """
    must input a tall DF with timeseries index, a "sensor" and "signal" column, and optionally
    a column whose name matches <facet_by>.
    """
    df_stack = df_stack.reset_index(names='timestamp')

    colors = px.colors.qualitative.Plotly
    dept_cmap = {'Facilities': 9, 'Executive': 1, 'Engineering': 2, 'Administration': 3, 'IT': 4, 'HR': 6,
                 'Security': 7}

    if not cat_orders:
        cat_orders = {'sensor': df_stack.sensor.sort_values().unique(),
                      'signal': ['high', 'medium', 'low', 'flat']}
    else:
        if (facet_by is not None) & (facet_by not in cat_orders):
            cat_orders[facet_by] = list(range(len(df_stack[facet_by].unique())))

    if not cmap:
        cmap = {'high': colors[1], 'medium': colors[4], 'low': colors[2], 'flat': colors[3]}

    if len(df_stack.sensor.unique()) <= 20:
        marker_size = 10
    elif len(df_stack.sensor.unique()) <= 200:
        marker_size = 5
    else:
        marker_size = 2

    fig = px.scatter(df_stack, x='timestamp', y='sensor', color='signal',
                     category_orders=cat_orders,
                     color_discrete_map=cmap,
                     title=title,
                     height=height,
                     facet_row=facet_by,
                     facet_row_spacing=0.02
                     )

    if facet_by:
        fig.update_yaxes(matches=None, tickfont=dict(size=10))

    fig.update_traces(
        marker=dict(size=marker_size, symbol="line-ns-open"),
        selector=dict(mode="markers"),
    )

    # horizontal rectangles
    if horiz_rect_df is not None:
        if facet_by:
            # Plotly seems to put the first facet group in row 0, but then the last facet group is in row 1, next to
            #  last in row 2, etc etc.  Assuming this behavior holds and the facet_by variable is simply sorted,
            #  this row assignment logic should work...
            facets = df_stack[facet_by].sort_values().unique()
            num_facets = len(facets)
            facet_row_map = {facets[0]:0}
            facet_row_map.update({facets[n]:num_facets-n for n in range(1,num_facets)})
            for i, facet in enumerate(facets):
                facet_group = horiz_rect_df[horiz_rect_df[facet_by] == facet]
                for d, group in facet_group.groupby(horiz_rect_by):
                    p = group.index.sort_values()
                    if horiz_rect_by == 'dept':
                        c = colors[dept_cmap[d]]
                    else:
                        c = colors[list(horiz_rect_df[horiz_rect_by].unique()).index(d)]
                    fig.add_hrect(
                        y0=p[0],
                        y1=p[-1],
                        row=facet_row_map[facet],
                        label=dict(
                            text=f"{d}",
                            textposition="middle left",
                            font=dict(size=12),
                        ),
                        fillcolor=c,
                        opacity=0.2,
                        line_width=0,
                    )
        else:
            for d, group in horiz_rect_df.groupby(horiz_rect_by):
                p = group.index.sort_values()
                if horiz_rect_by == 'dept':
                    c = colors[dept_cmap[d]]
                else:
                    c = colors[list(horiz_rect_df[horiz_rect_by].unique()).index(d)]
                fig.add_hrect(
                    y0=p[0],
                    y1=p[-1],
                    label=dict(
                        text=f"{d}",
                        textposition="middle left",
                        font=dict(size=12),
                    ),
                    fillcolor=c,
                    opacity=0.2,
                    line_width=0,
                )

    # vertical rectangles
    if type(vert_rect_df) != type(None):
        # add vertical rectangles over the supplied activity windows
        for i, row in vert_rect_df.iterrows():
            if vert_rect_by is None:
                c = colors[6]
                l = None
                t = f"{vert_rect_df.loc[i, 'cluster']}"
            else:
                c = colors[list(vert_rect_df[vert_rect_by].unique()).index(row[vert_rect_by])]
                l = row[vert_rect_by]
                t = f"{vert_rect_df.loc[i, 'cluster']} by {vert_rect_by}"
            fig.add_vrect(
                x0=row.t_start,
                x1=row.t_end,
                label=dict(
                    text=l,
                    textposition="top center",
                    font=dict(size=10),
                ),
                fillcolor=c,
                opacity=0.2,
                line_width=0,
            )
        i = vert_rect_df.index[0]
        fig.add_annotation(x=vert_rect_df.loc[i, 't_start'],
                           y=1.01,
                           yref="paper",
                           text=t,
                           showarrow=False,
                           )
    fig.for_each_trace(
        lambda trace: trace.update(visible="legendonly") if trace.name in ['flat'] else (),
    )
    fig.update_xaxes(range=x_range)
    return fig


def elbowPlot(df, max_clusters=20, optimal_cluster=1):
    """
    Run a second time with the <optimal_cluster> var set to update the plot
    """
    n_clusters = list(range(1, max_clusters))
    within_cluster_var = []
    for i in n_clusters:
        clu = cluster.KMeans(n_clusters=i, n_init='auto', random_state=42)
        clu = clu.fit(df)
        within_cluster_var.append(clu.inertia_)

    sns.lineplot(x=[i for i in range(1, max_clusters)], y=within_cluster_var)
    plt.axvline(optimal_cluster, c='r')
    plt.xticks(np.arange(0, max_clusters, 1))
    plt.title("Elbow Plot")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Within-Cluster Variance")
    plt.show()


def bldgUnstack(df_stack, by_node=False):
    if by_node:
        to_drop = set(df_stack.columns).intersection({'cluster', 'sensor'})
        df_unstack = df_stack.set_index(['floor', 'node', 'zone'], append=True).drop(to_drop, axis=1)
        df_unstack = df_unstack.unstack(['floor', 'node', 'zone'])
        df_unstack.columns = df_unstack.columns.droplevel(0)
    else:
        to_drop = set(df_stack.columns).intersection({'floor', 'zone', 'node'})
        df_unstack = df_stack.set_index(['cluster', 'sensor'], append=True).drop(to_drop, axis=1)
        df_unstack = df_unstack.unstack(['cluster', 'sensor'])
        df_unstack.columns = df_unstack.columns.droplevel(0)
    return df_unstack


def stackToWindows(df_stack, threshold=0.5, window_width=2, by_node=False, by_floor=False):
    # by default this will average the values per time-instance by cluster, unless <by_node> is set
    #  to True, in which case it will average per time-instance by node (all zones all floors).  If <by_floor>
    #  is set to True, it will average per-time instance by node on each floor separately.
    # If by_node is False, by_floor is ignored.
    # The threshold needs to change with respect to the input values, whether binerized or just standardized.
    # The function averages each time instance by the specified grouping, which will provide an activity score
    #  for each group for every instance, and can provide windows of time in which activity was above a certain threshold.
    # we will use these windows to overlay on other activity visuzlizations

    # create an un-stacked DF, which will make aggregation easier
    df_unstack = bldgUnstack(df_stack, by_node=by_node)

    # create a DF to hold the aggregation of the df_unstack
    if by_node:
        if by_floor:
            # create 1 column per node per floor
            df_activity = pd.DataFrame(np.nan, columns=df_unstack.columns.droplevel(2).unique(), index=df_unstack.index)
            # compute the average of every time instance in df_stack, by node per floor.
            for col in df_activity.columns:
                df_activity[col] = df_unstack.loc[:, (col[0], col[1], slice(None))].abs().mean(axis=1)
        else:
            # create 1 column per node
            df_activity = pd.DataFrame(np.nan, columns=df_unstack.columns.levels[1], index=df_unstack.index)
            # compute the average of every time instance in df_stack, by node.
            for node in df_activity.columns:
                df_activity[node] = df_unstack.loc[:, (slice(None), node, slice(None))].abs().mean(axis=1)
    else:
        # using the default by-cluster approach, create 1 column per cluster
        df_activity = pd.DataFrame(np.nan, columns=df_unstack.columns.levels[0], index=df_unstack.index)
        # compute the average of every time instance in df_stack, by cluster.
        for clus in df_activity.columns:
            df_activity[clus] = df_unstack.loc[:, (clus, slice(None))].abs().mean(axis=1)

    # Now we need to comb thru the 'activity' DF to isolate the edges of events, based on a given threshold.
    #  The idea is to capture the major events while weeding out the noise.
    windows = []

    for col in df_activity.columns:
        prev = 0
        count = 0
        tmp_start = 0
        for i in df_activity.index:
            score = df_activity[col][i]
            if score >= threshold:
                count += 1
                if prev < threshold:
                    # found leading edge
                    tmp_start = i
                prev = score
            else:
                if prev >= threshold:
                    # found trailing edge
                    if count >= window_width:
                        # minimum window achieved, record the event as (cluster, start time, end time)
                        windows.append((col, tmp_start, i - timedelta(minutes=5)))
                    count = 0
                    tmp_start = 0
                prev = score

    # now we'll go through the windows and combine adjacent windows that are separated by 30 minutes or less
    windows_combined = []
    prev_row = (np.nan, np.nan, np.nan)
    for i, row in enumerate(windows):
        if i != 0:
            # check cluster
            if row[0] == prev_row[0]:
                if row[1] - prev_row[2] <= timedelta(minutes=30):
                    # combine
                    prev_row = (row[0], prev_row[1], row[2])
                else:
                    # don't combine
                    windows_combined.append(prev_row)
                    if i == len(windows) - 1:
                        # if the last row didn't get combined, add it in now
                        windows_combined.append(row)
            else:
                # different clusters, don't combine
                windows_combined.append(prev_row)
                if i == len(windows) - 1:
                    # if the last row happens to be a different cluster, make sure to add it
                    windows_combined.append(row)
        prev_row = row

    if by_node & by_floor:
        windows_ = [[row[0][0], row[0][1], row[1], row[2]] for row in windows_combined]
        df_windows = pd.DataFrame(windows_, columns=['floor', 'cluster', 't_start', 't_end'])
    else:
        df_windows = pd.DataFrame(windows_combined, columns=['cluster', 't_start', 't_end'])
    return df_windows


def plotLocations(mobile_prox_df, maps, jitter=True, jitter_scale=1, savefig=None):
    """
    plots mobile prox locations only, since it relies on coordinates
    """
    dfc = mobile_prox_df.copy()  # to avoid any alterations to input DF

    # create a combined (x,y) column to groupby so we can use marker size as an indicator for how often that location is visited
    dfc['xy'] = dfc.apply(lambda row: (row.x, row.y), axis=1)
    df2 = dfc.loc[:, ['x', 'floor', 'prox-id', 'xy']] \
        .groupby(['prox-id', 'floor', 'xy']) \
        .count().reset_index().rename({'x': 'count'}, axis=1)
    # now explode x and y back out
    if jitter:
        # jitter all the x,y values a bit to see overlapping points
        # np.random.seed(42)
        jit = lambda: np.random.uniform(-jitter_scale, jitter_scale)
        df2['x'] = df2.apply(lambda row: row.xy[0] + jit(), axis=1)
        df2['y'] = df2.apply(lambda row: row.xy[1] + jit(), axis=1)
    else:
        df2['x'] = df2.apply(lambda row: row.xy[0], axis=1)
        df2['y'] = df2.apply(lambda row: row.xy[1], axis=1)

    # read in the images
    fims = [plt.imread(x) for x in maps]

    # plot the location data onto a facet grid, faceted by floor
    g = sns.relplot(df2, x='x', y='y', row='floor', hue='prox-id',
                    size='count', sizes=(75, 250), alpha=0.7, kind='scatter')
    # then add the floor map and other axis specific options
    for i, f in enumerate(g.axes_dict.keys()):
        ax = g.facet_axis(i, 0)
        ax.imshow(fims[f-1], extent=[0, 189, 0, 111])
        ax.set_xlabel(None)
        ax.set_ylabel(None)

    g.figure.set(figwidth=8, figheight=4 * len(g.axes))
    g.tight_layout()

    if savefig is None:
        plt.show()
    else:
        plt.savefig(savefig, dpi=300, format='png')

