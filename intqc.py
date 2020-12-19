#!/usr/bin/env python3
"""
====================================
Filename:       intqc.py
Author:         Jonathan Delgado 
Description:    InteractiveQuantumCircuit, a script which creates a GUI for
                manipulating and working with quantum states
====================================
"""
"""
    TODO:
Add sliders for moving T gates?
Add the option when gathering new entropy data for "fix state" versus "fix 
    circuit" which would either make the state the same (as currently) and get 
    new random circuits, or vice versa
"""

#------------- Imports -------------#

from tkinter import ttk
import tkinter

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

import numpy as np

import sys, os
import randqc
import qstate_imager

#------------- Matplotlib Settings -------------#
# plt.rc('text', usetex=True)

# Curves
plt.rcParams['lines.linewidth'] = 1.7  # line width in points
plt.rcParams['lines.linestyle'] = '--'
# Fonts
plt.rcParams['font.family']  = 'serif'
plt.rcParams['font.serif'] = [ 'CMU Serif', 'stix' ]
# {100, 200, 300, normal, 500, 600, bold, 800, 900}
plt.rcParams['font.weight']  = '500'
plt.rcParams['font.size']    = 12.0
# Axis/Axes
plt.rcParams['axes.linewidth'] = 0.9 # line width of border
plt.rcParams['xtick.minor.visible'] = plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['xtick.direction'] = plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = plt.rcParams['xtick.bottom'] = plt.rcParams['ytick.left'] = plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.major.size'] = plt.rcParams['ytick.major.size'] = 7
plt.rcParams['xtick.minor.size'] = plt.rcParams['ytick.minor.size'] = 3
# Legend
plt.rcParams['legend.framealpha'] = 0.8 # legend transparency, range: [0,1]

#------------- Code to limit threads -------------#
# _THREAD_LIMIT=1
_THREAD_LIMIT = "1"
# export OMP_NUM_THREADS=$_THREAD_LIMIT
os.environ["OMP_NUM_THREADS"] = _THREAD_LIMIT
# export OPENBLAS_NUM_THREADS=$_THREAD_LIMIT
os.environ["OPENBLAS_NUM_THREADS"] = _THREAD_LIMIT
# export OPENBLAS_NUM_THREADS=$_THREAD_LIMIT
os.environ["MKL_NUM_THREADS"] = _THREAD_LIMIT
# export VECLIB_MAXIMUM_THREADS=$_THREAD_LIMIT
os.environ["VECLIB_MAXIMUM_THREADS"] = _THREAD_LIMIT
# export NUMEXPR_NUM_THREADS=$_THREAD_LIMIT
os.environ["NUMEXPR_NUM_THREADS"] = _THREAD_LIMIT

_progress = 0

def indicate_gates(ax, instructions, gate='T'):
    """
        Helper function that marks on a plot where gates are being applied,
        typically used to mark T gate applications
    
        Args:
            instructions (str): the string encoding the instructions for
                the circuit
            gate (str): the type of gate to indicate
    
        Returns:
            none (none): none
    
    """
    instructions_list = randqc.parse_instructions(instructions)
    gate_placement = 0
    for gate_type, gate_count in instructions_list:
        if gate_type == gate:
            # Check if there are any subsequent T gates to simply increase
                # the width
            # The width will be the number of T_gates we apply
            # Added one for width and emphasis
            ax.axvspan(gate_placement, gate_placement + gate_count + 1, 0, 1, alpha=0.1)

        gate_placement += gate_count

import time
def _update_progress_bar(bar, master, update_amount, BAR_LEN):
    """
        Function that handles updating the progress bar.
    
    """
    global _progress
    _progress += update_amount
    # Converts the progress into a percentage
    status = _progress / BAR_LEN * 100

    bar['value'] = status

    # time.sleep(1) # Uncomment to see the bar more clearly
    master.update()

def main():

    def _quit():
        """
            Provides functionality to quit button.
        
        """
        # Stops mainloop
        master.quit()
        # This is necessary on Windows to prevent fatal Python Error: 
            # PyEval_RestoreThread: NULL tstate
        master.destroy()
        # Quits out of Platypus
        print("QUITAPP\n")

    def _progress_bar_init():
        bar_master.wm_title('Loading...')

        bar_master.style = ttk.Style()
        bar_master.style.theme_use('classic')

        bar_window_width = 200
        bar_window_height = 40

        width_screen = bar_master.winfo_screenwidth() # width of the screen
        height_screen = bar_master.winfo_screenheight() # height of the screen
        # Calculate (x, y) coordinates for the Tk bar_master window
        x = (width_screen/2) - (bar_window_width/2)
        y = (height_screen/2) - (bar_window_height/2)
        # Set the dimensions of the screen and where it is placed
        bar_master.geometry('%dx%d+%d+%d' % (bar_window_width, bar_window_height, x, y))

    def _window_init():
        """
            Tkinter window preparation.
        
        """
        master.wm_title('Interactive Quantum Circuits')
        # Change the styling to match MacOS
        # print(ttk.Style().theme_names())
        # master.style = ttk.Style()
        # master.style.theme_use('aqua')

        width_screen = master.winfo_screenwidth() # width of the screen
        height_screen = master.winfo_screenheight() # height of the screen
        # Calculate (x, y) coordinates for the Tk master window
        x = (width_screen/2) - (width/2)
        y = (height_screen/2) - (height/2)
        # Set the dimensions of the screen and where it is placed
        master.geometry('%dx%d+%d+%d' % (width, height, x, y))

    def _init_axes():
        """
            Handles initialization of plots and axes. Note this function is
            only called once. Any behavior that is clear with axes.clear needs 
            to be placed in the respect _update function.
        
        """

        # Hide their axes
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax2.axes.xaxis.set_visible(False)
        ax2.axes.yaxis.set_visible(False)

        # Makes the canvas take up all of the space and be resizable
        canvas.get_tk_widget().grid(row=0, column=0, columnspan=4, sticky="nsew")
        for i in range(ncols):
            tkinter.Grid.columnconfigure(master, i, weight=1)

    def _update():
        """
            Updates the contents of the interactive window based on input.
        
        """

        # Clears any previous state in the axes
        ax.clear(); ax2.clear()

        # Clear counter data for entropy data when changing state
        nonlocal counter, progress
        counter = 1

        ### Get the contents of input on update ###
        N = N_slider.get()
        nonlocal num_entropy_curves
        nonlocal initial_state; nonlocal output_state
        num_entropy_curves = entropy_curves_slider.get()
        bipartition = N // 2
        q0_label = q0_button_var.get()
        instructions = instructions_entry.get()
        renyi_parameter = float(renyi_entry.get())

        # Input read
        if not _INITIALIZED:
            _update_progress_bar(progress_bar, bar_master, 1, PROGRESS_BAR_LEN)

        # Gets the location of the entropy data
        entropy_file_path = randqc.get_entropy_file_path(N, q0_label, instructions, N//2, renyi_parameter=renyi_parameter, root=temp_path)

        # Clear any previous data gathered
        if os.path.exists(entropy_file_path):
            os.remove(entropy_file_path)

        # saves the initial state
        initial_state = randqc.parse_state_string(N, q0_label)
        image_initial_state = qstate_imager.state_to_image(initial_state, bipartition=bipartition, cmap=cmap)

        # First image generated
        if not _INITIALIZED:
            _update_progress_bar(progress_bar, bar_master, 1, PROGRESS_BAR_LEN)

        # Runs the initial state through the circuit
        output_state = randqc.randomqc(N=N, instructions=instructions, q0_label=q0_label, q0=initial_state, entropy_flag=True, save_path=temp_path, renyi_parameter=renyi_parameter)

        # Completed default circuit
        if not _INITIALIZED:
            _update_progress_bar(progress_bar, bar_master, 1, PROGRESS_BAR_LEN)

        image_output_state = qstate_imager.state_to_image(output_state, bipartition=bipartition, cmap=cmap)

        # Output image generated
        if not _INITIALIZED:
            _update_progress_bar(progress_bar, bar_master, 1, PROGRESS_BAR_LEN)

        # Set their titles
        ax.set_title(f'Initial state: {q0_label}')
        ax2.set_title(f'Output of circuit: {instructions}')

        # Plot the images
        ax.imshow(image_initial_state)
        ax2.imshow(image_output_state)

        # Images drawn
        if not _INITIALIZED:
            _update_progress_bar(progress_bar, bar_master, 1, PROGRESS_BAR_LEN)

        fig.suptitle(f'Number of qubits: {N}')
        
        canvas.draw()

        # Canvas updated
        if not _INITIALIZED:
            _update_progress_bar(progress_bar, bar_master, 1, PROGRESS_BAR_LEN)

        # Handles clearing and plotting of first entropy curve
        _plot_entropy(instructions, entropy_file_path)

        # Entropy plotted
        if not _INITIALIZED:
            _update_progress_bar(progress_bar, bar_master, 1, PROGRESS_BAR_LEN)

        # Plots subsequent curves as data is continually gathered
        master.after(after_timer, lambda: _update_entropy(N, q0_label, instructions, initial_state, renyi_parameter, entropy_file_path))

    def _plot_entropy(instructions, entropy_file_path):
        """
            Loads the entropy data and plots it
        
        """
        nonlocal progress

        # Clears any previous entropy plot
        ax3.clear()

        # Added one to account for entropy of initial state
        num_gates = randqc.get_total_gates(instructions) + 1

        entropy_data = np.loadtxt(entropy_file_path)

        # Entropy data loaded
        if not _INITIALIZED:
            _update_progress_bar(progress_bar, bar_master, 1, PROGRESS_BAR_LEN)

        ax3.plot(np.arange(num_gates), entropy_data, color='red')
        # Mark the location of the T gates
        indicate_gates(ax3, instructions)

        ax3.set_title('Entropy versus gates')
        ax3.set_xlabel('Gates')
        # Make the plot square to match the images for even qubit numbers
        ax3.set_box_aspect(1)

        canvas.draw()

    def _update_entropy(N, q0_label, instructions, initial_state, renyi_parameter, entropy_file_path):
        """
            Continues to gather and plot entropy data even while window has 
            loaded
        
        """

        nonlocal counter

        num_gates = randqc.get_total_gates(instructions) + 1

        # Code executed once all entropy curves are drawn, here we can 
            # calculate items like the mean and variance of the entropy
        if counter == num_entropy_curves and num_entropy_curves > 1:
            # We are going to plot the mean of all of the curves, but don't
                # bother doing it if we only have want one curve
            entropy_data = np.loadtxt(entropy_file_path)
            mean_entropy = entropy_data.mean(axis=0)
            
            ax3.plot(np.arange(num_gates), mean_entropy, color='black', label='Mean entropy')

            # Plots the "standard deviation" as a width to the mean
            # plt.fill_between(
            #     np.arange(num_gates),
            #     mean_entropy + np.std(entropy_data, axis=0),
            #     mean_entropy - np.std(entropy_data, axis=0),
            #     color='green',
            #     alpha=0.8,
            #     label='standard deviation'
            # )
            # Plots the standard deviation as its own curve
            # ax3.plot(
            #     np.arange(num_gates),
            #     np.std(entropy_data, axis=0),
            #     color='green',
            #     label='standard deviation'
            # )

            ax3.legend(loc='lower right')

            canvas.draw()
            counter += 1
            return

        # We've plotted all of the curves and their mean
        if counter > num_entropy_curves:
            return
        
        counter += 1

        randqc.randomqc(N=N, instructions=instructions, q0_label=q0_label, q0=initial_state, entropy_flag=True, save_path=temp_path, renyi_parameter=renyi_parameter)

        # Reload the new entropy data
        entropy_data = np.loadtxt(entropy_file_path)

        # Plot the last set of data added to it
        ax3.plot(np.arange(num_gates), entropy_data[-1], color='blue', alpha=0.2)

        # Update the canvas with the newly plotted curve
        canvas.draw()
        master.after(after_timer, lambda: _update_entropy(N, q0_label, instructions, initial_state, renyi_parameter, entropy_file_path))

    def _update_cmap(cmap_choice):
        nonlocal cmap
        cmap = cmaps[cmap_choice]
        # Recreate the images and re-plot them
        ax.imshow(qstate_imager.state_to_image(initial_state, cmap=cmap))
        ax2.imshow(qstate_imager.state_to_image(output_state, cmap=cmap))

        # Redraw the canvas
        canvas.draw()

    #------------- Parameters -------------#

    # Initialized app flag
    _INITIALIZED = False

    master = tkinter.Tk()
    # Hide the master window until it's ready
    master.withdraw()

    ### Prepare progress bar ###
    # Variable for tracking progress
    progress = 0 
    PROGRESS_BAR_LEN = 18
    bar_master = tkinter.Toplevel()
    progress_bar = ttk.Progressbar(bar_master, orient=tkinter.HORIZONTAL, length=100, mode='determinate')

    progress_bar.pack(fill='both', expand=True)
    _progress_bar_init()
    _update_progress_bar(progress_bar, bar_master, 0, PROGRESS_BAR_LEN)

    # Window size
    width = 1200; height = 700
    # Location of temporary folder for entropy data storage and other files
    temp_path = 'temp'
    # Sleep timer for generating for after loop
    after_timer = 20

    num_entropy_curves = 10
    ncols = 3

    # Increase the scope of the states
    initial_state = output_state = None
    _window_init()

    # Window created
    _update_progress_bar(progress_bar, bar_master, 1, PROGRESS_BAR_LEN)

    fig, (ax, ax2, ax3) = plt.subplots(ncols=ncols)
    # A tk.DrawingArea
    canvas = FigureCanvasTkAgg(fig, master=master)
    counter = 1

    _update_progress_bar(progress_bar, bar_master, 1, PROGRESS_BAR_LEN)

    #------------- Generate input fields -------------#

    # Separates sections of input
    row = 1

    ### Number of qubits input ###
    ttk.Label(master, text='Number of qubits:').grid(row=row, column=0)
    N_slider = tkinter.Scale(master, from_=2, to=12, orient=tkinter.HORIZONTAL)
    N_slider.grid(row=row, column=1)
    # Set the default value
    N_slider.set(8)

    # N slider made
    _update_progress_bar(progress_bar, bar_master, 1, PROGRESS_BAR_LEN)

    ### Colormap choice ###
    # For more colormaps please look at:
        # https://matplotlib.org/3.3.3/tutorials/colors/colormaps.html
    cmaps = {
        'Viridis':cm.viridis,
        'Gray': cm.gray,
        'Plasma':cm.plasma,
        'Inferno':cm.inferno,
        'Magma':cm.magma,
        'Cividis':cm.cividis,
        'Cubehelix':cm.cubehelix,
    }
    default_cmap = 'Viridis'
    # default_cmap = 'None'

    opt_var = tkinter.StringVar(master)
    opt_var.set(default_cmap)

    # Default cmap
    cmap = cmaps[default_cmap]

    cmap_menu = tkinter.OptionMenu(master, opt_var, *cmaps.keys(), command=_update_cmap)
    cmap_menu.grid(row=row, column=2)

    # Cmap menu made and set
    _update_progress_bar(progress_bar, bar_master, 1, PROGRESS_BAR_LEN)

    # We're entering a new section of inputs
    row += 1

    # ### Initial state input ###
    q0_button_var = tkinter.StringVar()
    zero_state_button = tkinter.Radiobutton(master, text='Zero state', variable=q0_button_var, value='zero')
    zero_state_button.grid(row=row, column=0)
    # zero_state_button.select()

    prod_state_button = tkinter.Radiobutton(master, text='Product state', variable=q0_button_var, value='prod')
    prod_state_button.grid(row=row, column=1)
    prod_state_button.select()

    # Adds a radio button for random phase states
    # tkinter.Radiobutton(master, text='Phase state', variable=q0_button_var, value='phase').grid(row=row, column=2)

    tkinter.Radiobutton(master, text='Random Haar state', variable=q0_button_var, value='rand').grid(row=row, column=2)

    # Input state buttons made
    _update_progress_bar(progress_bar, bar_master, 1, PROGRESS_BAR_LEN)

    row += 1

    ### Circuit instructions input ###
    ttk.Label(master, text='Circuit instructions:').grid(row=row, column=0)
    instructions_entry = ttk.Entry(master)
    # # Sets the default circuit instructions
    instructions_entry.insert(10, 'Clx50;Tx1;Clx50')    
    instructions_entry.grid(row=row, column=1)

    # Circuit instructions input made
    _update_progress_bar(progress_bar, bar_master, 1, PROGRESS_BAR_LEN)

    row += 1

    ### Renyi parameter input ###
    ttk.Label(master, text='Renyi Entropy:').grid(row=row, column=0)
    renyi_entry = ttk.Entry(master)
    # Sets the default circuit instructions
    renyi_entry.insert(10, '1')    
    renyi_entry.grid(row=row, column=1)

    # Entropy parameter input made
    _update_progress_bar(progress_bar, bar_master, 1, PROGRESS_BAR_LEN)

    row += 1

    ### Number of entropy curves input ###
    ttk.Label(master, text='Number entropy curves:').grid(row=row, column=0)
    entropy_curves_slider = tkinter.Scale(master, from_=1, to=100, orient=tkinter.HORIZONTAL)
    entropy_curves_slider.set(num_entropy_curves)
    entropy_curves_slider.grid(row=row, column=1)

    # Entropy curves slider made
    _update_progress_bar(progress_bar, bar_master, 1, PROGRESS_BAR_LEN)

    row += 1

    # Quit button and run button for running the simulation
    ttk.Button(master=master, text='Run', command=_update).grid(row=row, column=1)
    ttk.Button(master=master, text="Quit", command=_quit).grid(row=row, column=2)

    # Buttons made
    _update_progress_bar(progress_bar, bar_master, 1, PROGRESS_BAR_LEN)

    _init_axes()

    # Axes initialized
    _update_progress_bar(progress_bar, bar_master, 1, PROGRESS_BAR_LEN)

    _update()

    # We've initialized the GUI for the first time and are ready to start
        # the mainloop
    _INITIALIZED = True
    bar_master.destroy()

    # Reveal the prepared window
    master.style = ttk.Style()
    master.style.theme_use('aqua')
    master.deiconify()
    master.mainloop()

if __name__ == '__main__':
    main()