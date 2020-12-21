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

import sys, os
import numpy as np
from tkinter import ttk
import tkinter as tk

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

import time
# For generating .gif
from PIL import Image

import randqc, qstate_imager

_VERSION = 0.01

#------------- Matplotlib Settings -------------#
# plt.rc('text', usetex=True)

# Curves
plt.rcParams['lines.linewidth'] = 1.7  # line width in points
plt.rcParams['lines.linestyle'] = '--'
# Fonts
plt.rcParams['font.family']  = 'serif'
# plt.rcParams['font.serif'] = [ 'CMU Serif', 'stix' ]
plt.rcParams['font.serif'] = [ 'stix' ]
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

def center_window(master, width, height):
        screen_width = master.winfo_screenwidth()
        screen_height = master.winfo_screenheight()
        # Calculate (x, y) coordinates for the Tk window
        x = (screen_width/2) - (width/2)
        y = (screen_height/2) - (height/2)
        # Set the dimensions of the screen and where it is placed
        master.geometry('%dx%d+%d+%d' % (width, height, x, y))

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

class ProgressWindow:
    # Object which will handle creating a Tkinter Progress window

    def __init__(self, MAX_VALUE, width=200, height=40):
        # Must know maximum to calculate percentages
        self._MAX_VALUE = MAX_VALUE
        # Initialize value property
        self.value = 0
        self.master = tk.Toplevel()
        self.bar = ttk.Progressbar(self.master, orient=tk.HORIZONTAL, length=100, mode='determinate')
        self.width = width; self.height = height

        self._style()
        center_window(self.master, self.width, self.height)

    def _style(self):
        """
            Handles designing the bar.
        
        """
        self.bar.pack(fill='both', expand=True)
        self.master.wm_title('Loading...')
        self.master.style = ttk.Style()
        self.master.style.theme_use('classic')

    def update(self, amount=1):
        """
            Main function for interacting with the bar. Handles updating the 
            progress.
        
        """
        self.value += amount

        # print( f'Progress update counter: \n{self.value}' )
        status = self.value / self._MAX_VALUE * 100
        self.bar['value'] = status
        # time.sleep(1) # Uncomment to see the progress bar update slowly

        # Updates the window itself
        self.master.update()

    def finish(self):
        self.master.destroy()


class IntQC:
    # An Interactive Quantum Circuit object. The window application itself

    def __init__(self):
        
        # Window must be created early to generate TopLevel for progress window
        self.master = tk.Tk()
        # Hide the master window until it's ready
        self.master.withdraw()

        # Communicate the state of initialization for a progress bar
        self.pw = ProgressWindow(11)
        # Windows have been created
        self.pw.update()

        #------------- Parameters -------------#

        self.width = 1200; self.height = 720
        # Relative path to folder holding temporary entropy data storage and
            # other files.
        self.temp_path = 'temp'
        # Sleep timer for generating data in for after loop
        self.after_timer = 20
        self.num_entropy_curves = 10
        # Number of columns for GUI elements
        self.ncols = 3

        # The default parameters
        self.N = 8
        self.bipartition = self.N // 2
        self.instructions = 'Clx50;Tx1;Clx50'
        self.renyi_parameter = 1

        ### Colormap choice ###
        # For more colormaps please look at:
            # https://matplotlib.org/3.3.3/tutorials/colors/colormaps.html
        self.cmaps = {
            'Viridis':cm.viridis,
            'Gray': cm.gray,
            'Afmhot':cm.afmhot,
            'Jet':cm.jet,
            'Plasma':cm.plasma,
            'Inferno':cm.inferno,
            'Magma':cm.magma,
            'Cividis':cm.cividis,
            'Cubehelix':cm.cubehelix,
        }

        self.image_maps = ['|Psi|^2', '|Psi|']
        self.image_map = self.image_maps[0]

        #------------- Initializing window -------------#

        self.master.wm_title(f'Interactive Quantum Circuits - V{_VERSION}')
        center_window(self.master, self.width, self.height)

        # Window has been geometry has been handled
        self.pw.update()

        self.fig, (self.ax, self.ax2, self.ax3) = plt.subplots(ncols=self.ncols)
        # Creates the canvas for drawing to the Tk GUI
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)

        # Canvas created
        self.pw.update()

        self._create_widgets()
        self._init_axes()

        self._update()
        # First _update iteration completed
        self.pw.update()

        # Listen for slider input
        self.state_block_slider.configure(command=self._updated_state_block_slider)

        # We've initialized the GUI for the first time and are ready to start
            # the mainloop
        self.pw.finish()

        # Style and show prepared window
        self.master.style = ttk.Style()
        self.master.style.theme_use('aqua')
        self.master.deiconify()
        self.master.mainloop()

    def _create_widgets(self):
        """
            Handles loading GUI elements for interaction.
        
        """
        # Counter for the element row we are currently on
        row = 1
        ### Number of qubits input ###
        ttk.Label(self.master, text='Number of qubits:').grid(row=row, column=0)
        self.N_slider = tk.Scale(self.master, from_=2, to=16, orient=tk.HORIZONTAL, command=self._updated_N)
        self.N_slider.grid(row=row, column=1)
        # Set the default value
        self.N_slider.set(self.N)

        # N slider made
        self.pw.update()

        ### Color map menu ###
        default_cmap = 'Viridis'
        self.cmap = self.cmaps[default_cmap]
        cmaps_var = tk.StringVar(self.master)
        cmaps_var.set(default_cmap)
        tk.OptionMenu(self.master, cmaps_var, *self.cmaps.keys(), command=self._updated_cmap).grid(row=row, column=2)

        # Cmap menu made and set
        self.pw.update()

        # We're entering a new row of inputs
        row += 1

        ### State post block selector ###
        ttk.Label(self.master, text='State after circuit block:').grid(row=row, column=0)
        # The number of circuit blocks
        num_blocks = randqc.get_number_of_blocks(self.instructions)
        self.state_block_slider = tk.Scale(self.master, from_=1, to=num_blocks, orient=tk.HORIZONTAL)
        self.state_block_slider.grid(row=row, column=1)
        # Set the default value to show the output of the complete circuit
        self.state_block_slider.set(num_blocks)

        # State block slider generated
        self.pw.update()

        ### Number input for easy selection of state post block ###
        self.state_block_entry = tk.Entry(self.master, width=3)
        # Add listener to the entry to move slider and update the image
        self.state_block_entry.bind('<KeyRelease>', self._updated_state_block)
        self.state_block_entry.insert(0, str(num_blocks))
        self.state_block_entry.grid(row=row, column=2)


        row += 1

        ### Initial state input ###
        self.initial_state_button_var = tk.StringVar(self.master)
        self.zero_radio_button = tk.Radiobutton(self.master, text='Zero state', variable=self.initial_state_button_var, value='zero')
        self.zero_radio_button.grid(row=row, column=0)

        self.prod_radio_button = tk.Radiobutton(self.master, text='Product state', variable=self.initial_state_button_var, value='prod')
        self.prod_radio_button.grid(row=row, column=1)
        self.prod_radio_button.select()

        self.haar_radio_button = tk.Radiobutton(self.master, text='Random Haar State', variable=self.initial_state_button_var, value='rand')
        self.haar_radio_button.grid(row=row, column=2)

        # Input state buttons made
        self.pw.update()

        row += 1

        ### Circuit instructions input ###
        ttk.Label(self.master, text='Circuit instructions:').grid(row=row, column=0)
        self.instructions_entry = ttk.Entry(self.master)
        # Sets the default circuit instructions
        self.instructions_entry.insert(0, self.instructions)
        self.instructions_entry.grid(row=row, column=1)

        # Circuit instructions input made
        self.pw.update()

       ### Drop down to choose how to view state images ###
        image_maps_var = tk.StringVar(self.master)
        image_maps_var.set(self.image_map)
        image_maps_menu = tk.OptionMenu(self.master, image_maps_var, *self.image_maps, command=self._updated_image_map)
        image_maps_menu.grid(row=row, column=2)

        # Image map menu generated
        self.pw.update()

        row += 1

        ### Renyi parameter input ###
        ttk.Label(self.master, text='Renyi Entropy:').grid(row=row, column=0)
        self.renyi_entry = ttk.Entry(self.master, width=3)
        # Sets the default Renyi parameter
        self.renyi_entry.insert(0, str(self.renyi_parameter))
        self.renyi_entry.grid(row=row, column=1)

        # Entropy parameter input made
        self.pw.update()

        row += 1

        ### Number of entropy curves input ###
        ttk.Label(self.master, text='Number of entropy curves:').grid(row=row, column=0)
        self.entropy_curves_slider = tk.Scale(self.master, from_=1, to=25, orient=tk.HORIZONTAL)
        self.entropy_curves_slider.set(self.num_entropy_curves)
        self.entropy_curves_slider.grid(row=row, column=1)

        # Entropy curves slider made
        self.pw.update()

        row += 1

        # Quit and Run buttons for controlling the simulation
        ttk.Button(master=self.master, text='Quit', command=self._quit).grid(row=row, column=0)
        ttk.Button(master=self.master, text='Run', command=self._update).grid(row=row, column=1)
        # Export gif button to create an animation of state evolution
        ttk.Button(master=self.master, text='Export GIF', command=self._export_gif).grid(row=row, column=2)

    def _init_axes(self):
        """
            Handles initialization of plots and axes. Note this function is
            only called once. Any behavior that is clear with axes.clear needs 
            to be placed in the respect _update function.
        
        """

        # Hide their axes
        self.ax.axes.xaxis.set_visible(False)
        self.ax.axes.yaxis.set_visible(False)
        self.ax2.axes.xaxis.set_visible(False)
        self.ax2.axes.yaxis.set_visible(False)

        # Makes the canvas take up all of the space and be resizable
        self.canvas.get_tk_widget().grid(row=0, column=0, columnspan=4, sticky="nsew")
        for i in range(self.ncols):
            tk.Grid.columnconfigure(self.master, i, weight=1)        

    def _update(self):
        """
            Updates the content of the interactive window based on state of GUI
            elements.
        
        """
        ### Clear plots and counter data ###
        self.ax.clear(); self.ax2.clear()
        self.entropy_curve_counter = 1

        ### Update properties with GUI elements ###
        self.N = self.N_slider.get()
        self.bipartition = self.N // 2
        self.initial_state_label = self.initial_state_button_var.get()
        self.instructions = self.instructions_entry.get()
        self.renyi_parameter = float(self.renyi_entry.get())
        self.num_entropy_curves = self.entropy_curves_slider.get()

        # Gets the location of the entropy data
        entropy_file_path = randqc.get_entropy_file_path(self.N, self.initial_state_label, self.instructions, self.bipartition, renyi_parameter=self.renyi_parameter, root=self.temp_path)
        # Clears any previous data gathered
        if os.path.exists(entropy_file_path): os.remove(entropy_file_path)

        self.states = randqc.randomqc(N=self.N, instructions=self.instructions, q0_label=self.initial_state_label, q0=randqc.parse_state_string(self.N, self.initial_state_label), entropy_flag=True, save_path=self.temp_path, renyi_parameter=self.renyi_parameter)

        self.state_1 = self.states[0]
        self.state_2 = self.states[-1]

        # Pretends an image map has been changed to generate matrices then
            # images from states (while abiding to current image map)
            # and hence updates the images drawn.
        self._updated_image_map(self.image_map)

        # Update the block view slider to be at the end of the circuit
        num_blocks = randqc.get_number_of_blocks(self.instructions)
        self.state_block_slider.configure(to=num_blocks)
        self.state_block_slider.set(num_blocks)
        # Clear the text of the entry on update
        self.state_block_entry.delete(0, tk.END)
        self.state_block_entry.insert(0, num_blocks)

        self.ax.set_title(f'Initial state: {self.initial_state_label}')
        self.ax2.set_title(f'Output of circuit: {self.instructions}')
        self.fig.suptitle(f'Number of qubits: {self.N}')
        self.canvas.draw()

        # Handles clearing and plotting of first entropy curve
        self._plot_entropy(entropy_file_path)

        # Plots subsequent curves as data is continually gathered
        self.master.after(self.after_timer, lambda: self._update_entropy(entropy_file_path))

    def _plot_entropy(self, entropy_file_path):
        """
            Loads the entropy data and plots it.
        
        """
        # Clear any previous entropy plot
        self.ax3.clear()

        # Added one to account for entropy of initial state
        num_gates = randqc.get_total_gates(self.instructions) + 1

        entropy_data = np.loadtxt(entropy_file_path)

        self.ax3.plot(np.arange(num_gates), entropy_data, color='red')
        # Mark the location of the T gates
        indicate_gates(self.ax3, self.instructions)

        self.ax3.set_title('Entropy versus gates')
        self.ax3.set_xlabel('Gates')
        # Make the plot square to match the images for even qubit numbers
        self.ax3.set_box_aspect(1)

        self.canvas.draw()

    def _update_entropy(self, entropy_file_path):

        num_gates = randqc.get_total_gates(self.instructions) + 1
        
        # Code executed once all entropy curves are drawn, here we can 
            # calculate items like the mean and variance of the entropy
        if self.entropy_curve_counter == self.num_entropy_curves and self.num_entropy_curves > 1:
            # We are going to plot the mean of all of the curves, but don't
                # bother doing it if we only have one curve
            entropy_data = np.loadtxt(entropy_file_path)
            mean_entropy = entropy_data.mean(axis=0)
            
            self.ax3.plot(np.arange(num_gates), mean_entropy, color='black', label='Mean entropy')

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

            self.ax3.legend(loc='lower right')

            self.canvas.draw()
            self.entropy_curve_counter += 1
            return

        # We've plotted all of the curves and their mean
        if self.entropy_curve_counter >= self.num_entropy_curves:
            return
        
        self.entropy_curve_counter += 1

        # Generate more data
        randqc.randomqc(N=self.N, instructions=self.instructions, q0_label=self.initial_state_label, q0=self.state_1, entropy_flag=True, save_path=self.temp_path, renyi_parameter=self.renyi_parameter)

        # Reload the new entropy data
        entropy_data = np.loadtxt(entropy_file_path)

        # Plot the last set of data added to it
        self.ax3.plot(np.arange(num_gates), entropy_data[-1], color='blue', alpha=0.2)

        # Update the canvas with the newly plotted curve
        self.canvas.draw()
        self.master.after(self.after_timer, lambda: self._update_entropy(entropy_file_path))

    def _quit(self):
        """
            Provides functionality to the quit button.
        
        """
        # Stops mainloop
        self.master.quit()
        # This is necessary on Windows to prevent fatal Python Error: 
            # PyEval_RestoreThread: NULL tstate
        self.master.destroy()

    def _update_state_1(self, image):
        """
            Handles updating the first state view window.

        """
        self.ax.imshow(image)
        # Apply changes to canvas by redrawing it
        self.canvas.draw()

    def _update_state_2(self, image):
        """
            Handles updating the second state view window.

        """
        self.ax2.imshow(image)
        self.canvas.draw()

    def state_to_image(self, state):
        """
            Checks image modifiers such as image maps and colormaps to
            generate an accurate image representing the provided state.
        
        """
        matrix = qstate_imager.state_to_matrix(state, bipartition=self.bipartition)

        # Check image maps
        if self.image_map == self.image_maps[0]:
            # No manipulations needed
            pass
        elif self.image_map == self.image_maps[1]:
            # We want the moduli not the square moduli of probabilities
            matrix **= 0.5

        # Check colormaps
        image = qstate_imager.matrix_to_image(matrix, cmap=self.cmap)

        return image

    def _export_gif(self):
        folder = 'gifs'
        # Make the folder if it doesn't exist
        if not os.path.exists(folder):
            os.makedirs(folder)

        fname = os.path.join(folder, f'N({self.N})-q0({self.initial_state_label})-inst[{self.instructions}]' + randqc.get_serial())

        # Filter out blocks of circuits with only T gates to avoid the effect
            # of a longer pause on the image animation since the image will
            # generally be unchanged after only T gates
        filtered_instructions_indices = [0] + [
            i + 1 for i, block in enumerate(
                randqc.parse_instructions(self.instructions)
            ) if block[0] != 'T'
            # ) if block[0] != 'T' or block[1] > self.N
        ]

        first_image, *remaining = [
            # Change the resampling method to avoid blurring interpolation
                # and resize the image to make it viewable
            self.state_to_image(self.states[filtered_instructions_indices[i]]).resize((512, 512), resample=Image.NEAREST)
            for i in range(len(filtered_instructions_indices))
        ]

        first_image.save(fp=fname + '.gif', format='GIF', append_images=remaining, save_all=True, duration=1000, loop=0, dpi=80)
        print('GIF saved successfully.')

    #------------- Listener functions for updated GUI elements -------------#

    def _updated_N(self, *args):
        """
            N slider has been updated.
        
        """
        # Read the N loaded but don't save it since we may not use this
        N = self.N_slider.get()
        # Disable the random Haar radio button for N larger than 12
        if N > 12:
            self.haar_radio_button.configure(state=tk.DISABLED)
        else:
            self.haar_radio_button.configure(state=tk.NORMAL)

    def _updated_cmap(self, cmap_choice):
        """
            Colormap choice has been updated.
        
        """
        self.cmap = self.cmaps[cmap_choice]
        self._update_state_1(qstate_imager.state_to_image(self.state_1, cmap=self.cmap))
        self._update_state_2(qstate_imager.state_to_image(self.state_2, cmap=self.cmap))

    def _updated_image_map(self, image_map_choice):
        self.image_map = image_map_choice

        self._update_state_1(self.state_to_image(self.state_1))
        self._update_state_2(self.state_to_image(self.state_2))

    def _updated_state_block_slider(self, *args):
        block_num = self.state_block_slider.get()
        # Update the text entry to match the slider
        self.state_block_entry.delete(0, tk.END)
        self.state_block_entry.insert(0, block_num)
        # Update the image
        self._update_state_2(self.state_to_image(self.states[block_num]))

    def _updated_state_block(self, *args):
        # Try to convert the input into an integer, if it's not one
            # then ignore it
        try:
            num = int(self.state_block_entry.get())
        except:
            return

        # Update the slider to match
        # The listener on the block is sufficient to do the rest of the
            # work.
        self.state_block_slider.set(num)



def main():
    IntQC()

if __name__ == '__main__':
    main()