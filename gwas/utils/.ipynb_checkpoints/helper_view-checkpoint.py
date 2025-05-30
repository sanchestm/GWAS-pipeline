import pandas as pd
from tqdm import tqdm
import mygene
import sys
import warnings

mg = mygene.MyGeneInfo()
tqdm.pandas()
sys.setrecursionlimit(10000)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
pn.extension()
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

na_values_4_pandas = ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'None', 'n/a', 'nan', 'null', 'UNK']