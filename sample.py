# -*- coding: utf-8 -*-
"""
Created on Wed May  2 21:51:06 2018

@author: Ravi Theja
"""

import pandas as pd
df = pd.read_json('train.json', lines=True)

df_s = df[df['Critic']=='Delta']
df_s.describe()
ans = df_s.hist(column='Stars', bins=8)