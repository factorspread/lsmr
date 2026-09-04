import pandas as pd
import numpy as np
from hftbacktest.data.validation import validate_event_order, correct_event_order
from hftbacktest.types import (DEPTH_SNAPSHOT_EVENT, EXCH_EVENT, LOCAL_EVENT, DEPTH_EVENT, BUY_EVENT, ADD_ORDER_EVENT, DEPTH_CLEAR_EVENT, SELL_EVENT, TRADE_EVENT)
from numba import njit
from hftbacktest import BacktestAsset, HashMapMarketDepthBacktest
from json import loads
from copy import deepcopy

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def l2_convert(l2:list, fix_interval_nanos:int=None, clear_offset:int=None, start_dtime:str=None):

    #l2 can be list of dataframe or list of parquet files

    if isinstance(l2[0], str):
        src = [pd.read_parquet(rf'{p}') for p in l2]
    else:
        src = l2
    

    src_cols = ['symbol_id', 'time_exchange', 'time_coinapi']
    evt = [[]]

    #UNNEST
    j = 0
    for depth in src:
        n = depth.shape[0]
        if fix_interval_nanos:
            depth.loc[0, 'time_exchange'] = pd.to_datetime(start_dtime)
            fixed_range = pd.Series(pd.date_range(start=depth['time_exchange'].iloc[0], periods=n ,freq=f"{fix_interval_nanos}ns"))
            #Substract a bit of time to stay before hbt elapse time
            depth_clr_dtime = deepcopy(fixed_range)
            depth_clr_dtime[1:] = np.array((fixed_range + pd.to_timedelta(np.ones(n)*clear_offset, unit='ns')))[1:]
            depth['time_exchange'] = fixed_range
            depth['time_coinapi'] = fixed_range
        for i in range(n):
            bids_i = pd.DataFrame().from_records(depth['bids'].iloc[i])
            if i == 0:
                init_id_bids = bids_i.shape[0]
            bids_i[src_cols] = depth[src_cols].iloc[i]
            bids_i['is_buy'] = 1
            evt[j].append(bids_i)
            asks_i = pd.DataFrame().from_records(depth['asks'].iloc[i]) 
            if i == 0:
                init_id_asks = asks_i.shape[0]
            asks_i[src_cols] = depth[src_cols].iloc[i]
            asks_i['is_buy'] = 0
            evt[j].append(asks_i)
        evt[j] = pd.concat(evt[j]).reset_index(drop=True)

        #Format
        evt[j]['exch_ts'] = pd.to_datetime(evt[j]['time_exchange']).astype('int64') / 1 
        evt[j]['local_ts'] = pd.to_datetime(evt[j]['time_coinapi']).astype('int64') / 1
        evt[j] = evt[j].rename(columns={'entry_px':'price', 'entry_sx':'size'})
        evt[j]['price'] = evt[j]['price'] /1
        evt[j]['size'] = evt[j]['size'] /1
        evt[j]['order_id'] = 0.0
        evt[j]['ival'] = 0.0
        evt[j]['fval'] = 0.0

        #Determine events
        evt[j]['ev'] = np.where((evt[j]['is_buy'] == 1), (DEPTH_EVENT | BUY_EVENT | EXCH_EVENT | LOCAL_EVENT) /1, (DEPTH_EVENT | SELL_EVENT | EXCH_EVENT | LOCAL_EVENT) /1)
        
        clear = pd.DataFrame(data=pd.to_datetime(depth_clr_dtime).astype('int64'), columns=['exch_ts'])
        clear['local_ts'] = clear['exch_ts']
        clear['price'] = 0.0
        clear['size'] = 0.0
        clear['order_id'] = 0.0
        clear['ival'] = 0.0
        clear['fval'] = 0.0
        clear['ev'] = (DEPTH_CLEAR_EVENT | EXCH_EVENT | LOCAL_EVENT)

        evt[j] = pd.concat([evt[j], clear])

        j += 1

    evt = pd.concat(evt).reset_index(drop=True)

    #Convert to hftbacktest data structure
    event_dtype = np.dtype(
    {'names' : ('ev', 'exch_ts', 'local_ts', 'px', 'qty', 'order_id', 'ival', 'fval'),
    'formats' : ('u8', 'i8', 'i8', 'f8', 'f8', 'u8', 'i8', 'f8'),
    'offsets':  [0, 8, 16, 24, 32, 40, 48, 56],
    'itemsize': 64,
    'aligned':  True,
    }
    )
    data = np.zeros(evt.shape[0], dtype=event_dtype)
    data['ev'] = evt['ev']
    data['exch_ts'] = evt['exch_ts']
    data['local_ts'] = evt['local_ts']
    data['px'] = evt['price'].astype(float)
    data['qty'] = evt['size'].astype(float)
    data = np.sort(data, order='exch_ts')
    

    #Initial depth
    init_depth_id = init_id_asks + init_id_bids

    return data, init_depth_id


def create_snapshot(data:np.ndarray, init_depth_id:int):
    """Create initial snaphot based on the position of the last event.
    Return the data after the last snapshot event and the snapshot events in a tuple
    Warning : works only on L2 derived data"""
    snap_evt = data[:init_depth_id]
    snap_evt['ev'] = np.where((snap_evt['ev'] - (DEPTH_EVENT | BUY_EVENT | EXCH_EVENT | LOCAL_EVENT )== 0), (DEPTH_SNAPSHOT_EVENT | BUY_EVENT | EXCH_EVENT | LOCAL_EVENT) /1, (DEPTH_SNAPSHOT_EVENT | SELL_EVENT | EXCH_EVENT | LOCAL_EVENT) /1)

    return (data[init_depth_id:], snap_evt)

def get_fees(venue:str, market_type:str=None, liquitity_taker:bool=None,mkt_fees_file:str='./exchanges_fees.json'):
    
    venue = venue.upper()

    with open(mkt_fees_file, 'r') as f:
        exch_dtls = loads(f.read())
    
    res = exch_dtls.get(venue)

    if not res:
        return res
    
    mk_type_exists = market_type is None

    if not mk_type_exists:    
        res = res.get(market_type)

    if not res:
        return res
    
    if not mk_type_exists and not liquitity_taker is None:
        res = res.get('taker' if liquitity_taker else 'maker')

    return res


def _closest_pairwise(a_times, df_target, col_target=None):
    """
    Core O(m log m + n log m) nearest-timestamp search, shared by both
    find_closest_timestamps and find_closest_across_dataframes.

    Parameters
    ----------
    a_times : np.ndarray[datetime64[ns]]
        Reference timestamps to match (already converted/extracted).
    df_target : pd.DataFrame
        DataFrame to search within.
    col_target : str, optional
        Column to use in df_target. Defaults to its first column.

    Returns
    -------
    (best_index_labels, best_diff_ms) : tuple of np.ndarray
    """
    col_target = col_target or df_target.columns[0]
    t_times = pd.to_datetime(df_target[col_target]).to_numpy(dtype='datetime64[ns]')

    if len(t_times) == 0:
        raise ValueError("A target dataframe has no rows to match against.")

    # Sort target's timestamps once, keep original index labels aligned
    sort_order = np.argsort(t_times)
    t_sorted = t_times[sort_order]
    t_index_sorted = df_target.index.to_numpy()[sort_order]

    # Binary search: insertion point of each reference timestamp into sorted target
    ins_idx = np.searchsorted(t_sorted, a_times)

    # Two candidate neighbors: the one at ins_idx (right) and ins_idx-1 (left)
    right_idx = np.clip(ins_idx, 0, len(t_sorted) - 1)
    left_idx = np.clip(ins_idx - 1, 0, len(t_sorted) - 1)

    right_diff = np.abs(t_sorted[right_idx] - a_times)
    left_diff = np.abs(t_sorted[left_idx] - a_times)

    use_right = right_diff <= left_diff
    best_pos = np.where(use_right, right_idx, left_idx)
    best_diff = np.where(use_right, right_diff, left_diff)

    return t_index_sorted[best_pos], best_diff / np.timedelta64(1, 'ms')


def find_closest_timestamps(df_a, df_b, col_a=None, col_b=None):
    """
    For each timestamp in df_a, find the index (in df_b) of the closest
    timestamp in df_b, and the time difference in milliseconds.

    Overall complexity: O((n + m) log m), vs O(n * m) for brute force.

    Returns
    -------
    pd.DataFrame indexed like df_a, with columns:
        - 'b_index' : the index label (from df_b.index) of the closest match
        - 'diff_ms' : absolute time difference in milliseconds
    """
    col_a = col_a or df_a.columns[0]
    a_times = pd.to_datetime(df_a[col_a]).to_numpy(dtype='datetime64[ns]')

    best_idx, best_diff = _closest_pairwise(a_times, df_b, col_b)

    return pd.DataFrame(
        {'b_index': best_idx, 'diff_ms': best_diff},
        index=df_a.index,
    )


def find_closest_across_dataframes(df_ref, other_dfs, col_ref=None, cols=None):
    """
    Generalization to N dataframes: for each timestamp in a single
    reference dataframe, find the closest timestamp's index and time
    difference (ms) in each of several other dataframes.

    Each target dataframe is handled independently with the same
    O(m log m) sort + O(log m) binary-search-per-row approach, so total
    complexity is O((n * k) log m) for k target dataframes of size ~m.

    Parameters
    ----------
    df_ref : pd.DataFrame
        The reference dataframe whose timestamps drive the lookup.
    other_dfs : dict[str, pd.DataFrame]
        Mapping of name -> dataframe to search, e.g.
        {'b': df_b, 'c': df_c, 'd': df_d}. Names are used as column
        prefixes in the result.
    col_ref : str, optional
        Timestamp column in df_ref. Defaults to its first column.
    cols : dict[str, str], optional
        Mapping of name -> column name for entries in other_dfs whose
        timestamp column isn't the first column.

    Returns
    -------
    pd.DataFrame indexed like df_ref, with two columns per target df:
        - '{name}_index'  : index label of the closest match in that df
        - '{name}_diff_ms': absolute time difference in milliseconds
    """
    cols = cols or {}
    col_ref = col_ref or df_ref.columns[0]
    ref_times = pd.to_datetime(df_ref[col_ref]).to_numpy(dtype='datetime64[ns]')

    result = pd.DataFrame(index=df_ref.index)
    for name, df_target in other_dfs.items():
        best_idx, best_diff = _closest_pairwise(ref_times, df_target, cols.get(name))
        result[f'{name}_index'] = best_idx
        result[f'{name}_diff_ms'] = best_diff

    return result