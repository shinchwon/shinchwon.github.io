# 2 장 Pandas
* 데이터 처리와 분석을 위한 라이브러리

* 행과 열로 이루어진 데이터 객체를 만들어 다룰 수 있음

* 대용량의 데이터들을 처리하는데 매우 편리

* pandas 자료구조

 **Series: 1차원
 ** DataFrame: 2차원
 **Panel: 3차원
* pandas 로딩


```python
import numpy as np # 보통 numpy와 함께 import
import pandas as pd
```

# 2.1 Pandas DataFrame
* 2차원 자료구조

* 행레이블/열레이블, 데이터로 구성됨

* 딕셔너리(dictionary)에서 데이터프레임 생성


```python
import pandas as pd
# 딕셔너리
data = {
    'year':[2016, 2017, 2018],
    'GDP rate': [2.8, 3.1, 3.0],
    'GDP': ['1.637M', '1.73M', '1.83M' ]
}
df = pd.DataFrame(data, index=data['year']) # index추가할 수 있음
print(df)
```

          year  GDP rate     GDP
    2016  2016       2.8  1.637M
    2017  2017       3.1   1.73M
    2018  2018       3.0   1.83M
    


```python
print("row labels:", df.index)
```

    row labels: Int64Index([2016, 2017, 2018], dtype='int64')
    


```python
print("column labels:", df.columns)
```

    column labels: Index(['year', 'GDP rate', 'GDP'], dtype='object')
    


```python
print("head:", df.head()) # print some lines in data
```

    head:       year  GDP rate     GDP
    2016  2016       2.8  1.637M
    2017  2017       3.1   1.73M
    2018  2018       3.0   1.83M
    

* csv 파일에서 데이터프레임 생성


```python
csv_data_df = pd.read_csv("C:/Users/user/Desktop/r practice/pop2019.csv")
print(csv_data_df.head())
```


    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    Cell In[6], line 1
    ----> 1 csv_data_df = pd.read_csv("C:/Users/user/Desktop/r practice/pop2019.csv")
          2 print(csv_data_df.head())
    

    File ~\anaconda3\lib\site-packages\pandas\util\_decorators.py:211, in deprecate_kwarg.<locals>._deprecate_kwarg.<locals>.wrapper(*args, **kwargs)
        209     else:
        210         kwargs[new_arg_name] = new_arg_value
    --> 211 return func(*args, **kwargs)
    

    File ~\anaconda3\lib\site-packages\pandas\util\_decorators.py:331, in deprecate_nonkeyword_arguments.<locals>.decorate.<locals>.wrapper(*args, **kwargs)
        325 if len(args) > num_allow_args:
        326     warnings.warn(
        327         msg.format(arguments=_format_argument_list(allow_args)),
        328         FutureWarning,
        329         stacklevel=find_stack_level(),
        330     )
    --> 331 return func(*args, **kwargs)
    

    File ~\anaconda3\lib\site-packages\pandas\io\parsers\readers.py:950, in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, squeeze, prefix, mangle_dupe_cols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, error_bad_lines, warn_bad_lines, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options)
        935 kwds_defaults = _refine_defaults_read(
        936     dialect,
        937     delimiter,
       (...)
        946     defaults={"delimiter": ","},
        947 )
        948 kwds.update(kwds_defaults)
    --> 950 return _read(filepath_or_buffer, kwds)
    

    File ~\anaconda3\lib\site-packages\pandas\io\parsers\readers.py:605, in _read(filepath_or_buffer, kwds)
        602 _validate_names(kwds.get("names", None))
        604 # Create the parser.
    --> 605 parser = TextFileReader(filepath_or_buffer, **kwds)
        607 if chunksize or iterator:
        608     return parser
    

    File ~\anaconda3\lib\site-packages\pandas\io\parsers\readers.py:1442, in TextFileReader.__init__(self, f, engine, **kwds)
       1439     self.options["has_index_names"] = kwds["has_index_names"]
       1441 self.handles: IOHandles | None = None
    -> 1442 self._engine = self._make_engine(f, self.engine)
    

    File ~\anaconda3\lib\site-packages\pandas\io\parsers\readers.py:1735, in TextFileReader._make_engine(self, f, engine)
       1733     if "b" not in mode:
       1734         mode += "b"
    -> 1735 self.handles = get_handle(
       1736     f,
       1737     mode,
       1738     encoding=self.options.get("encoding", None),
       1739     compression=self.options.get("compression", None),
       1740     memory_map=self.options.get("memory_map", False),
       1741     is_text=is_text,
       1742     errors=self.options.get("encoding_errors", "strict"),
       1743     storage_options=self.options.get("storage_options", None),
       1744 )
       1745 assert self.handles is not None
       1746 f = self.handles.handle
    

    File ~\anaconda3\lib\site-packages\pandas\io\common.py:856, in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
        851 elif isinstance(handle, str):
        852     # Check whether the filename is to be opened in binary mode.
        853     # Binary mode does not support 'encoding' and 'newline'.
        854     if ioargs.encoding and "b" not in ioargs.mode:
        855         # Encoding
    --> 856         handle = open(
        857             handle,
        858             ioargs.mode,
        859             encoding=ioargs.encoding,
        860             errors=errors,
        861             newline="",
        862         )
        863     else:
        864         # Binary mode
        865         handle = open(handle, ioargs.mode)
    

    FileNotFoundError: [Errno 2] No such file or directory: 'C:/Users/user/Desktop/r practice/pop2019.csv'



```python
print(csv_data_df.columns)
```

* 특정 변수의 추출


```python
#df에서 year변수의 값 추출
print(df['year']) 
# 또는 
```

* 부분추출


```python
print(csv_data_df[["sido", 'male']]) # DataFrame
```


```python
print(csv_data_df.loc[:3, ['sido', 'male']])
```


```python
print(csv_data_df.iloc[:3, :3])
```


```python
print (csv_data_df[csv_data_df['sido'] == '경상남도']) # 부울 인덱싱
```


```python
print(csv_data_df['female'].sum())
```


```python
print (df.describe()) # describe( )를 통해 기본적인 통계치를 모두 표시
```

             year  GDP rate
    count     3.0  3.000000
    mean   2017.0  2.966667
    std       1.0  0.152753
    min    2016.0  2.800000
    25%    2016.5  2.900000
    50%    2017.0  3.000000
    75%    2017.5  3.050000
    max    2018.0  3.100000
    


```python
print(csv_data_df.describe())
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[8], line 1
    ----> 1 print(csv_data_df.describe())
    

    NameError: name 'csv_data_df' is not defined


* 빈도


```python
# One-way contingency table
x=pd.crosstab(index=csv_data_df.sido, columns="count", margins=True)
print(x)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[9], line 2
          1 # One-way contingency table
    ----> 2 x=pd.crosstab(index=csv_data_df.sido, columns="count", margins=True)
          3 print(x)
    

    NameError: name 'csv_data_df' is not defined



```python
print('type of x=',type(x))
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[10], line 1
    ----> 1 print('type of x=',type(x))
    

    NameError: name 'x' is not defined



```python
print(x['count'])
# Two-way contingency table
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[11], line 1
    ----> 1 print(x['count'])
    

    NameError: name 'x' is not defined



```python
pd.crosstab(csv_data_df.sido, csv_data_df.female, margins=True)
# Three-way contingency table
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[12], line 1
    ----> 1 pd.crosstab(csv_data_df.sido, csv_data_df.female, margins=True)
    

    NameError: name 'csv_data_df' is not defined


# 2.2 Pandas 통계처리

df.sum(),
df.sum(axis='columns'),
df.mean(axis='columns', skipna=False),
df.cumsum()

# 2.3 Pandas plot
1. Boxplot


```python
df = pd.DataFrame({
         'unif': np.random.uniform(-3, 3, 20),
         'norm': np.random.normal(0, 1, 20)
    })
print(df.head())    
```

           unif      norm
    0 -1.146625 -0.497797
    1  1.718684  0.350536
    2 -2.400737  0.522445
    3 -1.999335  0.095268
    4 -1.161776  0.155345
    


```python
df.boxplot(column=['unif', 'norm'])
```




    <Axes: >




    
![png](output_29_1.png)
    


2. time series plot


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.DataFrame({
         'unif': np.random.uniform(-3, 3, 20),
         'norm': np.random.normal(0, 1, 20)
    })
# 플로팅
df.plot()
plt.show()

```


    
![png](output_31_0.png)
    



```python

```


```python

```


```python

```


```python

```
