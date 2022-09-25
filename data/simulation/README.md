# Use the followign command to generate simulated data (small and large)

- generate initial simulated Y and top level X values (numeric)

python generated_simulated_mvp_Y.py 5500 > mvp_y_fname

Two files will be generated: mvp_y_fname and top_x_vals


- python discretize_generate_Y.py 5500

This generates top_z_vals.npy


- python data_generation.py
