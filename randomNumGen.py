# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 01:35:58 2016

@author: Edward
"""
import random

random.seed()

limit = int(input('I will give you a number between 1 and: '))

print('I chose: ' + str(random.randint(1,limit)))
