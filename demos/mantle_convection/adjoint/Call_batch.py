#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import subprocess
import csv
import pandas as pd


# %%
# Change these inputs
filename = 'Parameter_List_Adjoint_tutorial_Set_4.csv'
functional_series = './functionals/functional_Set_4_'

# df = pd.read_csv(filename)
# #checking the number of empty rows in th csv file
# print (df.isnull().sum())
# #Droping the empty rows
# modifiedDF = df.dropna()
# #Saving it to the csv file 
# modifiedDF.to_csv(filename,index=False)

with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if row[0]:
                if line_count == 0:
                    line_count += 1
                else:
                    # Minimisation Parameters
                    parameter_set = row[0]
                    iteration_limit = row[1]
                    initial_radius = row[2]
                    radius_growing_rate = row[3]
                    radius_shrinking_rate_negative_rho = row[4]
                    radius_shrinking_rate_positive_rho = row[5]
                    radius_shrinking_threshold = row[6]
                    radius_growing_threshold = row[7]
                    functional_file = functional_series + str(parameter_set) + '.txt'
                    subprocess.run(['python', 'Adjoint_batch.py', functional_file, iteration_limit, initial_radius, radius_growing_rate, radius_shrinking_rate_negative_rho, radius_shrinking_rate_positive_rho, radius_shrinking_threshold, radius_growing_threshold])
                    # print(line_count, functional_file, iteration_limit, initial_radius, radius_growing_rate, radius_shrinking_rate_negative_rho, radius_shrinking_rate_positive_rho, radius_shrinking_threshold, radius_growing_threshold)
                    # print(f'Processed parameter set{line_count}')
                    line_count += 1
        # print(f'Processed total {line_count-1} parameter sets.')


# %%




