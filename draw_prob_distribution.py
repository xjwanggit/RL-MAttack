import os
import pandas as pd
import matplotlib.pyplot as plt
import re

def extract_numeric_part(file_name):
    pattern = r'prob_(\d+)'
    matches = re.search(pattern, file_name)
    return int(matches.group(1))



# Path to the directory containing the CSV files
folder_path = r'./results/VBPR/Amazon_Man/RLRS/Subs_resnet50-Rec_resnet50/user_id12623/action_distribution'

# Get a list of all files in the directory
file_names = os.listdir(folder_path)
file_name_eps = []
file_name_index = []
for x in file_names:
    if 'eps' in x:
        file_name_eps.append(x)
    else:
        file_name_index.append(x)

# total_number = 0
# Iterate through each file

sorted_file_index_name = sorted(file_name_index, key=extract_numeric_part)
sorted_file_eps_name = sorted(file_name_eps, key=extract_numeric_part)

dict_file_index_name = {}
dict_file_eps_name = {}
pattern_index = r'(\d+)_(\d+)_index'
pattern_eps = r'(\d+)_(\d+)_eps'
for x in sorted_file_index_name:
    matches = re.search(pattern_index, x)
    if matches.group(1) not in dict_file_index_name.keys():
        dict_file_index_name[f'{matches.group(1)}'] = []
    dict_file_index_name[f'{matches.group(1)}'].append(x)

for x in sorted_file_eps_name:
    matches = re.search(pattern_eps, x)
    if matches.group(1) not in dict_file_eps_name.keys():
        dict_file_eps_name[f'{matches.group(1)}'] = []
    dict_file_eps_name[f'{matches.group(1)}'].append(x)



# Set font properties
plt.figure(figsize=(8, 5))
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.subplots_adjust(right=0.62)
for key, item in dict_file_index_name.items():
    if int(key) != 63657:
        continue
    # if len(item) != 300 or int(key) != 59844:
    #     continue
    total_number = 0
    for file_name in item:
        # Check if the file is a CSV file
        if file_name.endswith('.csv'):
            # Construct the full path to the CSV file
            file_path = os.path.join(folder_path, file_name)

            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv(file_path)

            # Plot the data from the CSV file
            x = df.iloc[:, 0]
            y = df.iloc[:, 1]
            plt.plot(x, y, label=file_name)  # Assuming x and y are column names in the CSV file
            total_number += 1
            if total_number % 5 == 0:
                # Add labels and legend
                if 'index' in file_name:
                    plt.xlabel('Target Class')
                    plt.ylabel('Action Probability')
                    plt.title('策略网络的类别选择概率')
                    plt.legend(loc=1, bbox_to_anchor=(1.76, 1))  # 这个bbox_to_anchor指的是legend的框框的右上角的位置，1.76指的是长度是绘制的图像的1.76倍


                    # Show the plot
                    if not os.path.exists('./action_distribution_figure/'):
                        os.makedirs('./action_distribution_figure/')
                    plt.savefig(f'./action_distribution_figure/{key}_index_{total_number // 5}.png')
                    # plt.show()
                    plt.figure(figsize=(8, 5))
                    plt.subplots_adjust(right=0.62)
                else:
                    plt.xlabel('Target Class')
                    plt.ylabel('Action Probability')
                    plt.title('策略网络的扰动选择概率')
                    plt.legend(loc=1, bbox_to_anchor=(1.76, 1))  # 这个bbox_to_anchor指的是legend的框框的右上角的位置，1.76指的是长度是绘制的图像的1.76倍


                    # Show the plot
                    if not os.path.exists('./action_distribution_figure/'):
                        os.makedirs('./action_distribution_figure/')
                    plt.savefig(f'./action_distribution_figure/{key}_eps_{total_number // 5}.png')
                    # plt.show()
                    plt.figure(figsize=(8, 5))
                    plt.subplots_adjust(right=0.62)