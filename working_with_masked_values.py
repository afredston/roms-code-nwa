import pandas as pd
import os

if __name__ == '__main__':
    print(os.getcwd())
    if os.path.exists('data/hauls.csv'):
        print('Yes')
    df = pd.read_csv('data/full_haul_data_ver_2.csv')
    print(df)
    print(df.dtypes)
    zplk = df.iloc[12]['lg_zplk_seasonal']
    print(zplk)
    if zplk == '--':
        print('Yes')
