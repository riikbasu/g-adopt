#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import subprocess
import csv
import pandas as pd
from adjoint_NK_batch_test_5_timsesteps import main
from multiprocessing import Pool


# %%
# # Change these inputs
# filename = 'N_K_testing_focussed.csv'

# # df = pd.read_csv(filename)
# # #checking the number of empty rows in th csv file
# # print (df.isnull().sum())
# # #Droping the empty rows
# # modifiedDF = df.dropna()
# # #Saving it to the csv file 
# # modifiedDF.to_csv(filename,index=False)

# with open(filename) as csv_file:
#         csv_reader = csv.reader(csv_file, delimiter=',')
#         line_count = 0
#         for row in csv_reader:
#             if row[0]:
#                 if line_count == 0:
#                     line_count += 1
#                 else:
#                     # Minimisation Parameters
#                     parameter_set = row[0]
#                     line_search_type = row[1]
#                     curvature_condition = row[2]
#                     abs_tol = row[3]
#                     rel_tol = row[4]
#                     iterations = row[5]
#                     functional_file = 'functional_NK_tests_focussed' + str(parameter_set) + '.txt'
#                     subprocess.run(['python', 'adjoint_NK_batch_test_5_timsesteps.py', functional_file, parameter_set, line_search_type, curvature_condition, abs_tol, rel_tol, iterations])
#                     # print(line_count, functional_file, iteration_limit, initial_radius, radius_growing_rate, radius_shrinking_rate_negative_rho, radius_shrinking_rate_positive_rho, radius_shrinking_threshold, radius_growing_threshold)
#                     # print(f'Processed parameter set{line_count}')
#                     line_count += 1
#         # print(f'Processed total {line_count-1} parameter sets.')


# %%
# Change these inputs
filename = 'N_K_testing_focussed.csv'

# df = pd.read_csv(filename)
# #checking the number of empty rows in th csv file
# print (df.isnull().sum())
# #Droping the empty rows
# modifiedDF = df.dropna()
# #Saving it to the csv file 
# modifiedDF.to_csv(filename,index=False)

functional_file_list = []
parameter_set_list = []
line_search_type_list = []
curvature_condition_list = []
abs_tol_list = []
rel_tol_list = []
iterations_list = []

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
                    parameter_set_list.append(parameter_set)
                    line_search_type = row[1]
                    line_search_type_list.append(line_search_type)
                    curvature_condition = row[2]
                    curvature_condition_list.append(curvature_condition)
                    abs_tol = row[3]
                    abs_tol_list.append(abs_tol)
                    rel_tol = row[4]
                    rel_tol_list.append(rel_tol)
                    iterations = row[5]
                    iterations_list.append(iterations)
                    functional_file = 'functional_NK_tests_focussed' + str(parameter_set) + '.txt'
                    functional_file_list.append(functional_file)
                    # subprocess.run(['python', 'adjoint_NK_batch_test_5_timsesteps.py', functional_file, parameter_set, line_search_type, curvature_condition, abs_tol, rel_tol, iterations])
                    # print(line_count, functional_file, iteration_limit, initial_radius, radius_growing_rate, radius_shrinking_rate_negative_rho, radius_shrinking_rate_positive_rho, radius_shrinking_threshold, radius_growing_threshold)
                    # print(f'Processed parameter set{line_count}')
                    line_count += 1
        # print(f'Processed total {line_count-1} parameter sets.')


# %%
if __name__ == '__main__':
    with Pool(12) as pool: # 12 parallel jobs
        results = pool.map(main, zip(functional_file_list, parameter_set_list, line_search_type_list, curvature_condition_list, abs_tol_list, rel_tol_list, iterations_list))
