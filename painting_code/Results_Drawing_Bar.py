import os.path
import matplotlib.cm as cm
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def convert_to_decimal(value):
    if isinstance(value, str) and '%' in value:
        return float(value.replace('%', '')) / 100
    elif isinstance(value, str) and ('0' in value or '1' in value):
        return float(value)
    else:
        return value
dataset_name = r'Amazon Men'
recommender_system = ['VBPR', 'DVBPR', 'DeepStyle']
scenario = [1]  # scenario = ['scenario I', 'scenario II', 'scenario III']，三个场景
file_save__path = '../experiment_results/graph_bar'
file_save__path = os.path.join(file_save__path, dataset_name)
os.makedirs(file_save__path, exist_ok=True)

data_path = r"../results.xlsx"
sheet_name = r'Sheet2'
excel_data = pd.ExcelFile(data_path)
start_row_index, end_row_index = 4, 9
start_column_index, end_column_index = 4, 11

try:
    if data_path.endswith('.csv'):
        data = pd.read_csv(data_path)
    elif data_path.endswith('.xlsx'):
        data = pd.read_excel(data_path, sheet_name=sheet_name)
    else:
        raise ValueError("Unsupported file format. Please use CSV or Excel.")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

flag = False
data.iloc[:, 1] = data.iloc[:, 1].ffill()  # Forward-fill the first column
data.iloc[:, 2] = data.iloc[:, 2].ffill()  # Forward-fill the second column

filtered_data = data[data.iloc[:, 1] == dataset_name]
unique_recommender_systems = filtered_data.iloc[:, 2].dropna().unique()
non_null_indices = []
recommender_system_name = []

for system in unique_recommender_systems:
    # Get the first index where the system appears
    first_index = filtered_data[filtered_data.iloc[:, 2] == system].index[0]
    non_null_indices.append(first_index)  # Store the first index
    recommender_system_name.append(system)  # Store the system name
non_null_count = len(non_null_indices)
non_null_indices -= non_null_indices[0]


filtered_data = filtered_data.iloc[:, start_column_index:end_column_index]
filtered_data = filtered_data.applymap(convert_to_decimal)
attack_methods = filtered_data.iloc[:, 0].unique()
metrics = ['HR@1', 'HR@10', 'HR@100', 'AE', 'WPS', 'WSS']

num_methods = len(attack_methods)      # Number of attack methods
num_metrics = len(metrics)             # Number of metrics
bar_width = 0.15                       # Width of individual bars

cmap = cm.get_cmap('Set2', len(attack_methods))
plt.rcParams['font.family'] = 'DejaVu Serif'
for recommender_system_index in range(non_null_count):
    if (recommender_system_name[recommender_system_index] not in recommender_system) and recommender_system:
        continue
    fig, axes = plt.subplots(1, len(scenario), figsize=(14, 5))
    start_position = non_null_indices[recommender_system_index]
    handles, labels = None, None
    figure_index = 0
    for index in range(1, 4):
        if index not in scenario:
            continue
        if index > 0 and figure_index == 0:
            fig.set_size_inches(6, 3)
            ax = axes
        else:
            ax = axes[figure_index]
        figure_index += 1
        x = np.arange(num_metrics)   # X-axis positions for attack methods
        for idx, attack in enumerate(attack_methods):
            ax.bar(x + idx * bar_width, filtered_data.iloc[num_methods * (index - 1) + idx, 1:],
                   width=bar_width, label=attack, edgecolor='black', zorder=3, color=cmap(idx))
        if handles is None and labels is None:
            handles, labels = ax.get_legend_handles_labels()

        ax.set_xlabel('Metrics')
        ax.set_ylabel('Values')
        title_name = f"{recommender_system_name[recommender_system_index]} - {dataset_name}, Scenario {index}".replace(
            '1', '\u2160').replace('2', '\u2161').replace('3', '\u2162'  # Roman numerals I, II, III in Unicode
                                                          )
        ax.set_title(title_name)
        ax.set_xticks(x + bar_width * (num_methods - 1) / 2)  # Center the tick labels
        ax.set_xticklabels(metrics)  # Set metric names as x-axis labels
        ax.set_ylim(0, 1)
        ax.grid(axis='y', color='gray', linestyle='--', linewidth=0.7, zorder=0)

    # Create a shared legend using the collected handles and labels
    fig.legend(
        handles, labels, title='Attack Methods',
        loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=len(labels),
        frameon=False
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(os.path.join(file_save__path, f'{recommender_system_name[recommender_system_index]}.png'))
    plt.close(fig)
# else:
#     fig, axes = plt.subplots(1, 3, figsize=(12, 6))
#     start_position = 0
#     for index, ax in enumerate(axes, start=1):
#         x = np.arange(num_metrics)  # X-axis positions for attack methods
#         for idx, attack in enumerate(attack_methods):
#             ax.bar(x + idx * bar_width, filtered_data.iloc[start_position + idx, 1:],
#                    width=bar_width, label=attack, zorder=3, color=cmap(idx))
#         start_position = idx
#
#         ax.set_xlabel('Metrics')
#         ax.set_ylabel('Values')
#         ax.set_title(f'{recommender_system} - {dataset_name}, Scenario {index}')
#         ax.set_xticks(x + bar_width * (num_methods - 1) / 2)  # Center the tick labels
#         ax.set_xticklabels(metrics)  # Set metric names as x-axis labels
#         ax.set_ylim(0, 1)
#         ax.grid(axis='y', color='gray', linestyle='--', linewidth=0.7, zorder=0)
#     handles, labels = axes[0].get_legend_handles_labels()
#
#     fig.legend(
#         handles, labels, title='Attack Methods',
#         loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=len(labels),
#         frameon=False
#     )
#
#     plt.tight_layout(rect=[0, 0, 1, 0.95])
#
#     plt.savefig(os.path.join(file_save__path, f'{recommender_system}.png'))
#     plt.close(fig)

