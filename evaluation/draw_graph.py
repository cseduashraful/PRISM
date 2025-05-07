import pandas as pd
import matplotlib.pyplot as plt

color_map = {
    '128': 'magenta',
    '256': 'green',
    '512': 'red',
    '1024': 'purple',
    '2048': 'orange',
    '4096': 'brown',
    '8192': 'black',
    '16384': 'teal',
    '32768': 'crimson'
}

def draw_loss_graph(dataset, data):
    plt.figure(figsize=(14, 8))
    for model_name, model_data in data.items():
        batch_size = model_name.split('_')[-1]
        model_type = model_name.split('_')[0]#'TGN' if 'TGN' in model_name and 'SDA' not in model_name else 'SDA-TGN'
        cumulative_time = pd.Series(model_data['time']).cumsum()
        linestyle = ':' if model_type.startswith('TGN') else '-'
        plt.plot(cumulative_time, model_data['loss'], label=model_name, color=color_map[batch_size], linestyle=linestyle)

    plt.xlabel('Training Time (s)')
    plt.ylabel('Loss')
    plt.title('Training Loss vs. Training Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{dataset}_loss.pdf", bbox_inches='tight')
