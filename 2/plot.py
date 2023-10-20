import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# Function to read the matrices from the ".out" file
def read_matrices_from_file(filename):
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
            num_matrices = int(lines[0])  # Read the number of matrices
            matrices = []
            max_values = []

            matrix_lines = lines[1:]
            for i in range(num_matrices):
                matrix = []
                max_value = 0  # Initialize the max value for this matrix
                for line in matrix_lines[i * (len(matrix_lines) // num_matrices):(i + 1) * (len(matrix_lines) // num_matrices)]:
                    row = [float(num) for num in line.split()]
                    max_value = max(max_value, max(row))  # Update max value
                    matrix.append(row)
                matrices.append(np.array(matrix))
                max_values.append(max_value)
            
            return matrices, max_values, num_matrices
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        exit(1)

# Function to create an interactive heatmap
def create_interactive_heatmap(matrices, max_values):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.35)

    timestep_slider = Slider(ax=plt.axes([0.2, 0.1, 0.65, 0.03]), label='Timestep', valmin=0, valmax=len(matrices) - 1, valinit=0, valstep=1)

    def update(val):
        timestep = int(timestep_slider.val)
        ax.clear()
        ax.imshow(matrices[timestep], cmap='Reds', vmin=0, vmax=max_values[timestep], aspect=2)  # Use the corresponding max value
        ax.set_title(f'Matrix Visualization (Timestep {timestep}, {max_values[timestep]})')
        ax.set_xlabel('Columns')
        ax.set_ylabel('Rows')
        plt.draw()

    def next_timestep(event):
        nxt = timestep_slider.val + num_matrices // 10
        if nxt >= len(matrices):
            nxt = len(matrices) - 1
        timestep_slider.set_val(nxt)

    def prev_timestep(event):
        nxt = timestep_slider.val - num_matrices // 10
        if nxt <= 0:
            nxt = 0
        timestep_slider.set_val(nxt)

    next_button_ax = plt.axes([0.8, 0.02, 0.1, 0.04])
    next_button = Button(next_button_ax, 'Next')
    next_button.on_clicked(next_timestep)

    prev_button_ax = plt.axes([0.7, 0.02, 0.1, 0.04])
    prev_button = Button(prev_button_ax, 'Previous')
    prev_button.on_clicked(prev_timestep)

    timestep_slider.on_changed(update)
    update(0)  # Initialize the plot
    plt.show()

if __name__ == "__main__":
    filename = "out.txt"  # Replace with your actual filename

    matrices, max_values, num_matrices = read_matrices_from_file(filename)
    create_interactive_heatmap(matrices, max_values)
