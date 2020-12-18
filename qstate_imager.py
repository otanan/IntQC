#!/usr/bin/env python3
"""
====================================
Filename:       qstate_imager.py
Author:         Jonathan Delgado 
Description:    Script that handles the logic of taking quantum states as
                Qobj's and converts them into images including changing bases
                and manipulating other pieces of the image generation process
====================================
"""

######################## Imports ########################

import numpy as np

import pickle

# Image generation
from PIL import Image
from matplotlib import cm
from matplotlib.colors import ListedColormap

import qutip as qt

# Ensures the grayscale colormap is gray linearly.
GRAY_CMAP = ListedColormap(np.array([np.linspace(0, 1, 256), np.linspace(0, 1, 256), np.linspace(0, 1, 256), np.full((256,), 1)]).T)

#------------- State properties and saving -------------#

def get_N(q):
    """
        Helper function for calculating N from a passed in state, q
    
        Args:
            q (qutip.Qobj): the quantum state
    
        Returns:
            N (int): the number of qubits in the system
    
    """
    # return len(q.dims[0])
    # More general, works with normal numpy arrays
    return int(np.log2(q.shape[0]))

def qobj_from_array(data):
    """
        Instantiates a proper qutip.Qobj from given data.
    
        Args:
            data (numpy.ndarray): the components of the state
    
        Returns:
            q (qutip.Qobj): the state
    
    """
    N = get_N(data)
    dims = [[2] * N, [1] * N]

    return qt.Qobj(data, dims=dims, shape=data.shape, type='ket', isherm=False, copy=False, isunitary=False)

def save_state(q, fname):
    """
        Takes a state and saves it to fname in a compressed format.
    
        Args:
            q (qutip.Qobj): the state to be saved
            fname (str): the file name
    
        Returns:
            none (None): none
    
    """
    # qt.qsave(q, fname)
    # After some elementary testing, .npz was preferred to save the states due
        # to file size
    np.savez_compressed(fname, array1=q)

def load_state(fname, data_only=False):
    """
        Function that handles loading of quantum state with legacy code for
        handling states saved previously using qutip's saving functionality.
    
        Args:
            fname (str): the file name
            data_only (bool): whether to only load the data instead of 
                returning an entire qutip.Qobj
    
        Returns:
            q (qutip.Qobj/numpy.ndarray): the loaded state
    
    """
    file_type = fname.split('.')[-1]

    if file_type == 'qu':
        # Copied from qutip source and adjusted to remove print statement
        # Source: http://qutip.org/docs/latest/modules/qutip/fileio.html#qload
        return pickle.load(open(fname, 'rb'), encoding='latin1')
    elif file_type == 'npz':
        # If we are only interested in returning the state as a numpy.ndarray
            # and are not interested in converting it to a qutip.Qobj.
        if data_only:
            return np.load(fname)['array1']

        return qobj_from_array(np.load(fname)['array1'])

    print('Unsupported file extension for loading. Returning None...')
    return None

#------------- State manipulations -------------#

def partition_state(q, partition):
    """
        Generalizes the bipartition by converting the state into a 
        multi-dimensional array.
    
        Args:
            q (qutip.Qobj): the state to partition
            partition (list): the list of indices to partition the state into,
                the list is expected to be in increasing order and its last 
                element cannot be larger than the number of qubits in the 
                system. An example partitioning for an N=8 state would be (2, 
                4, 6) to equally break the state into a 4-dimensional array
                each with 4 elements (2 qubits). Note that a partitioning list
                of length n will result in an (n+1)-dimensional array.
    
        Returns:
            partitioned_state (numpy.ndarray): the partitioned state
    
    """
    N = get_N(q)

    # In the case of a bipartition, only a single number is passed
    if type(partition) is int:
        partition = [partition]

    # If the last element of the partition exceeds the number of qubits
    if partition[-1] > N:
        print('Invalid partitioning of state. Partitioning must not exceed number of qubits.')
        return

    # We need to include the number of qubits to the partition
    partition = np.append(partition, N)
    # The partitioning before served as a list of indices determining where
        # to cut off the state, but now we need to use this list to keep track
        # of how big each dimension of the array needs to be
    # Note that diff will make us lose the first element so we need to
        # re-include it
    new_shape = 2**np.append(partition[0], np.diff(partition))

    # In the case of a bipartition we take the state, as a row and wrap it 
        # over as to fit into the dimensions of the matrix, i.e. if the system 
        # has 3 qubits then the state will have 2**3 components, if we choose 
        # a bipartition of 1 then the dimensions of the matrix will be 
        # 2^1x2^(3-1)= 2x4  so q[0] will be mapped to A[0, 0] and q[5] will be 
        # mapped to A[1,1]
    # The squaring means we use the square norms which represent probabilities
        # as opposed to, say, the norm of each component
    return np.reshape(q, new_shape)

def state_to_matrix(q, bipartition=None):
    """
        Converts quantum state to a matrix with real values in [0,1] which can 
        represent gray-scale values. Done by calculating the square norm of 
        each amplitude.
    
        Args:
            q (qutip.Qobj): the quantum state itself
            bipartition (int/None): the bipartition to use
    
        Returns:
            matrix (numpy.ndarray): the real valued matrix
    
    """
    if bipartition is None:
        bipartition = get_N(q) // 2

    return np.abs(partition_state(q, bipartition))**2

#------------- Image Operations -------------#

def state_to_image(q, bipartition=None, cmap=GRAY_CMAP):
    """
        Function that takes an quantum state, and generates the corresponding 
        image based on the given bipartition.
    
        Args:
            q (qObj): the state to be converted to an image
            bipartition (int): the bipartition to be used, related to the 
                dimensions of the image produced
            cmap (matplotlib.cm, None): the colormap, left gray-scale if given 
                None
    
        Returns:
            none (None): none
    
    """
    return matrix_to_image(
        state_to_matrix(q, bipartition=bipartition),
        cmap=cmap
    )

def matrix_to_image(A, cmap=GRAY_CMAP):
    """
        Converts a matrix of nonnegative real numbers to a gray-scale or 
        colored image depending on a colormap. Although since each component 
        is only one real number associated to each pixel, the exact colors 
        themselves are for visualization purposes only and are not information 
        from the matrix itself. Each matrix is renormalized to the interval [0, 1] to prevent larger images from being "dimmer".

    """
    # Normalize the matrix to fill the entire gray-scale range
    image_matrix = A / A.max()
    # Color it using the provided colormap
    image_matrix = cmap(image_matrix)        

    image_matrix *= 255
    return Image.fromarray(image_matrix.astype(np.uint8))

def save_state_as_image(q, colored, fname, bipartition=None):
    """
        Function that takes an input state, generates the corresponding image and saves it to path with file name passed as argument.
    
        Args:
            q (qObj): the state to be converted to an image
            fname (string): the path and name of the image to be saved
            bipartition (int): the bipartition to be used, related to the 
                dimensions of the image produced
    
        Returns:
            none (none): none
    
    """
    # Converts the quantum state to the corresponding image matrix
        # converts the image matrix to the image, then saves the output
        # to the path and filename provided
    file_type = '.png'
    # Colormaps: https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
    cmap = cm.cubehelix if colored else GRAY_CMAP # {cm.viridis, cm.plasma}
    state_to_image(q, bipartition, cmap=cmap).save(fname + file_type)

def qubit_change_of_basis(q, basis='x'):
    """
        Change of basis function that calculates the change of basis to
        each qubit then applies the corresponding transformation to the
        entire state q
    
        Args:
            q (qObj): the state
            basis (string): the new basis, pass in "rand" in order to use a
                random qubit basis
    
        Returns:
            q_new_basis (qObj): the state in the new basis
    
    """
    N = get_N(q)

    # The original basis
    computational_basis = np.array([ [1,0],
                                    [0,1] ])

    # Pick the new basis
    if basis == 'x':
        new_basis = np.array( [
                            computational_basis[0] + computational_basis[1],
                            computational_basis[0] - computational_basis[1]
                        ]).T / np.sqrt(2)
    elif basis == 'rand':
        new_basis = qt.rand_unitary_haar(2)
    else:
        print('Please provide a valid basis to change to.')
        print('Returning the state unchanged...')
        return q

    # Construct the change of basis operator for the entire state
    tensored_new_basis = new_basis
    for _ in range(N - 1):
        tensored_new_basis = np.kron(tensored_new_basis, new_basis)
    # Returns the object as a proper qObj
    return qobj_from_array(np.matmul(tensored_new_basis, q))

def main():
    print('qstate_imager.py')

    # path = '/Users/otanan/Google Drive/Notability/Research - Hamma/mlq/projects/randomqc/scripts/results/demo/N(18)-q0(exp prod)-inst[Clx300;Tx20;Clx300]/states-N(18)-q0(exp prod)-inst[Clx300;Tx20;Clx300]-Universal'

if __name__ == '__main__':
    main()