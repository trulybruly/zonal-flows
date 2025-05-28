import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from GOLEM_notebook_scripts.py import *

# Create the main figure and subplots
fig, axes = plt.subplots(5, 1, figsize=(10, 15), gridspec_kw={'height_ratios': [1, 1, 1, 1, 0.2]})
fig.subplots_adjust(hspace=0.5)

# Axes for the individual plots
ax_bias_distance = axes[0]
ax_lp_distance = axes[1]
ax_drp_distance = axes[2]
ax_vct = axes[3]
ax_slider = axes[4]

# Plot placeholders for the left column
plot_distance_between(None, plot_type='bias', ax=ax_bias_distance)
plot_distance_between(None, plot_type='lp', ax=ax_lp_distance)
plot_distance_between(None, plot_type='drp', ax=ax_drp_distance)
plot_VCT(ax=ax_vct)

# Add vertical lines to indicate the slider time
bias_line = ax_bias_distance.axvline(x=0, color='red', linestyle='--')
lp_line = ax_lp_distance.axvline(x=0, color='red', linestyle='--')
drp_line = ax_drp_distance.axvline(x=0, color='red', linestyle='--')
vct_line = ax_vct.axvline(x=0, color='red', linestyle='--')

# Create a subplot for Golem_geometry
fig_geometry, ax_geometry = plt.subplots(figsize=(5, 5))
geometry_plot = Golem_geometry(None, ax=ax_geometry)

# Slider for time adjustment
slider_ax = plt.axes([0.15, 0.05, 0.75, 0.04])
time_slider = Slider(slider_ax, 'Time', valmin=0, valmax=100, valinit=0, valfmt='%1.2f')

def update(val):
    time = time_slider.val

    # Update vertical lines on the left column plots
    bias_line.set_xdata(time)
    lp_line.set_xdata(time)
    drp_line.set_xdata(time)
    vct_line.set_xdata(time)

    # Update Golem_geometry plot
    Golem_geometry(time, ax=ax_geometry)

    # Redraw the figure
    fig.canvas.draw_idle()
    fig_geometry.canvas.draw_idle()

time_slider.on_changed(update)

plt.show()
