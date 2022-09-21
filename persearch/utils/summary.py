from datetime import datetime
from pathlib import Path
from typing import Union

from numpy import sqrt, ndenumerate
import pandas as pd

from persearch.utils import print_title


class Summary(object):
    """
    build a Summary instance for each test to write logs and summarize the results
    """
    def __init__(self, metrics: list = None, name: str = 'summary'):
        self.name = name
        self.metrics = metrics
        self.rst = []  # a list of pd.DataFrame that contains results
        self.next()

    def add(self, key: str, data: dict) -> None:
        self.rst[-1].loc[key] = data

    def next(self) -> None:
        if len(self.rst) > 0:
            self.write(self.rst[-1])
        self.rst.append(pd.DataFrame(columns=self.metrics))

    def write(self, rst: pd.DataFrame, dir_outp: str = '', suffix: str = '',
              use_timestamp: bool = True) -> None:
        time_stamp = datetime.now().strftime('%b%d%H%M%S') if use_timestamp else ''
        time_stamp = time_stamp.lower()
        suffix = '-' + suffix
        dir_outp = Path(dir_outp)
        dir_outp.mkdir(parents=True, exist_ok=True)
        file_name = '{}{}{}.csv'.format(self.name, suffix, time_stamp)
        file_path = dir_outp.joinpath(file_name)
        rst.to_csv(file_path)
        print('> output log {}'.format(file_path))

    def close(self):
        if len(self.rst) > 0:
            self.write(self.rst[-1])

    def summary(self, dir_inp: str, dir_outp: str = '../',
                keywords: Union[str, list] = None, fmt='{:.4f}+/-{:.3f}'):
        """search and stat the files in the prefix with all the keywords in the title"""
        # load log
        dir_inp = Path(dir_inp)
        assert dir_inp.exists()
        files = list(dir_inp.glob('*.csv'))
        keywords = [keywords] if isinstance(keywords, str) else keywords
        files_keep = [
            file for file in files if '-{}'.format(self.name) not in str(file)]

        for keyword in keywords:
            files_keyword = [
                file for file in files_keep if '{}-'.format(keyword) in str(file)]
            rst_lst = [pd.read_csv(f, index_col=0) for f in files_keyword]
            # statistics
            n_log = len(files_keyword)
            if n_log < 1:
                print('! no log {} qualified, jumped out.'.format(keyword))
                continue
            rst_avg = sum(rst_lst) / n_log
            rst_square = [rst * rst for rst in rst_lst]
            rst_var = sum(rst_square) - n_log * (rst_avg * rst_avg)
            rst_std = sqrt(rst_var)
            # make format and output
            rst_fmt = pd.DataFrame(index=rst_avg.index, columns=rst_avg.columns)
            for (r, c), v in ndenumerate(rst_avg):
                rst_fmt.iloc[r, c] = fmt.format(rst_avg.iloc[r, c], rst_std.iloc[r, c])
            
            print_title('SUMMARY {}'.format(keyword))
            print(rst_fmt)
            self.write(rst_fmt, dir_outp=dir_outp, suffix=keyword,
                       use_timestamp=False)
