# 收藏我个人很喜欢的, 有点那么一点trick的代码小组件

import numpy as np
import pandas as pd
import xarray as xr

# 将 pd.DataFrame 转成N维 xr.DataArray
def dataframe_to_xarray(df:pd.DataFrame, dim_cols:list, data_col:str):
    ''' 
    @Description: 根据指定的轴名+数据, 转成n-dim的矩阵; 暂时只用到了 float 类型, maybe np.datetime64类型后续会补充
    @Parameters: dim_cols: 转成轴的列名; data_col: 数据所在的列名
    @Examples: a = pd.DataFrame({'x':[1,2]*15, 'y':[11,12,13]*10, 'z':[21,22,23,24,25]*6, 'value':np.random.rand(30)})
               dataframe_to_xarray(df=a, dim_cols=['x','y','z'], data_col='value')
    @Return: xr.DataArray
    @Author: lzeng
    ''' 
    ndf = df.copy()
    coord_list, idx_list = [], [] # coord相关的idx, 新的idx名称
    for col_name in dim_cols: # 对轴列进行排序
        uaxis, ndf[f'{col_name}_idx'] = np.unique(ndf[col_name].values.reshape(-1), return_inverse=True)
        coord_list.append(uaxis)
        idx_list.append(f'{col_name}_idx')
    res_data = np.full(ndf[dim_cols].nunique(), np.nan) # 使用idx生成 np.ndarray
    res_data[tuple(ndf[idx_name].values for idx_name in idx_list)] = ndf[data_col]
    res_xa = xr.DataArray(data = res_data, 
                          coords={col_name:coord_list[k] for k,col_name in enumerate(dim_cols)}, 
                          dims=dim_cols) # 加入轴的信息
    return res_xa


def last_x_non_nan(a, x=1, value_dtype='float'):
    ''' 
    @Description: 得到每一列最后x个非nan的值, 个数不足用nan填充
    @Parameters: a: N维np.ndarray, x: int
    @Examples: if non nan values of this column(sg.[1. nan 2. nan]) is less than x(sg.4), then return np.array([nan nan 1. 2.])
    @Return: np.ndarray
    @Author: lzeng
    ''' 
    assert a.ndim > 1 ,'matrix dimension less than 2' 
    mask = np.isnan(a)
    non_nan_count = np.cumsum(np.count_nonzero(~mask, axis=0))
    low_border = np.array([0] + list(non_nan_count[0:-1])) - 1
    rel_idx = np.arange(-x, 0)
    abs_idx = np.maximum(rel_idx[:, None] + non_nan_count, low_border)
    nan_idx = np.where(abs_idx == low_border, True, False)

    # get non nan array
    arr_raveled = a.ravel('F')
    arr_clear = arr_raveled[~np.isnan(arr_raveled)]
    if len(arr_clear) == 0:
        return np.full((x, a.shape[1]),np.nan)
    else:
        arr_result = arr_clear[abs_idx.ravel('F')].reshape((x, a.shape[1]), order='F')
        if value_dtype == 'float':
            arr_result[nan_idx] = np.nan
        elif value_dtype == 'datetime':
            arr_result[nan_idx] = np.datetime64('nat')
        else:
            assert False, "value_dtype not support!"
        return arr_result
