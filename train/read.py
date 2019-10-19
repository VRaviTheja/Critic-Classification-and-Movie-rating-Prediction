# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 21:10:22 2018

@author: Ravi Theja
"""

import json
'''
json_data=open('train.json').read()

data = json.loads(json_data)
print(type(json_data))
'''

def text():
    text_data = []
    with open('train/train.json') as f:
        i = 0
        for line in f:
            j_content = json.loads(line)
            j_str = json.dumps(j_content)
            i = i + 1
            text_data.append(j_content)
    return text_data
