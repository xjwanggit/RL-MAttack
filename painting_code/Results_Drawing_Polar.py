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
scenario = [3]  # scenario = ['scenario I', 'scenario II', 'scenario III']，三个场景
file_save__path = '../experiment_results/polar_figure'
file_save__path = os.path.join(file_save__path, dataset_name)
os.makedirs(file_save__path, exist_ok=True)
metrics = ['HR@1', 'HR@10', 'HR@100', 'AE', 'WPS', 'WSS']
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
flag = True

filtered_data = filtered_data.fillna(0)
if filtered_data.empty:
    print("No matching data found for the given dataset and recommender system.")
    exit()

filtered_data = filtered_data.iloc[:, start_column_index:end_column_index]
filtered_data = filtered_data.applymap(convert_to_decimal)
attack_methods = filtered_data.iloc[:, 0].unique()
metrics = ['HR@1', 'HR@10', 'HR@100', 'AE', 'WPS', 'WSS']

num_methods = len(attack_methods)      # Number of attack methods
num_metrics = len(metrics)             # Number of metrics

angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
angles += angles[:1]
plt.rcParams['font.family'] = 'DejaVu Serif'
for recommender_system_index in range(non_null_count):
    if (recommender_system_name[recommender_system_index] not in recommender_system) and recommender_system:
        continue
    fig, axes = plt.subplots(1, len(scenario), figsize=(14, 5), subplot_kw=dict(polar=True))
    start_position = non_null_indices[recommender_system_index]
    handles, labels = None, None
    figure_index = 0
    for index in range(1, 4):
        if index not in scenario:
            continue
        if index > 0 and figure_index == 0:
            fig.set_size_inches(10, 6)
            ax = axes
        else:
            ax = axes[figure_index]
        figure_index += 1
        for idx, attack in enumerate(attack_methods):
            metrics_value = filtered_data.iloc[(index - 1) * num_methods + idx, 1:].tolist()
            metrics_value += metrics_value[:1]
            ax.plot(angles, metrics_value, linewidth=2, linestyle='solid', label=attack)
            ax.fill(angles, metrics_value, alpha=0.25)
        ax.set_xticks(angles[:-1])  # Set metric labels (except the repeated last angle)
        ax.set_xticklabels(metrics, fontsize=12)
        ax.tick_params(pad=15)
        ax.set_ylim(0, 1)  # Normalize radial axis between 0 and 1
        ax.grid(color='gray', linestyle='--', linewidth=0.7)
        title_name = f"Polar Plot of {recommender_system_name[recommender_system_index]} - {dataset_name}, Scenario {index}".replace(
            '1', '\u2160').replace('2', '\u2161').replace('3', '\u2162')  # Roman numerals I, II, III in Unicode

        ax.set_title(title_name, pad=30)
        if handles is None and labels is None:
            handles, labels = ax.get_legend_handles_labels()

    # Create a shared legend using the collected handles and labels
    fig.legend(
        handles, labels, title='Attack Methods',
        loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=len(labels),
        frameon=False
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(os.path.join(file_save__path, f'{recommender_system_name[recommender_system_index]}.png'))
    plt.close(fig)
