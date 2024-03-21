"""
Utils for adding/multiplying models (lists of layers), making empty models etc.
Retrieving weights from a TF model results in a 2d list, each element is a layer
of the model, each element in the layer an array of values. 'Model' therefore 
refers to a 2d (ragged) list of arrays. Getting optimizer values returns a 1d
list of arrays, so 'Optim' refers to a 1d list.
"""
import numpy as np

        
def add_model_ws(a, b):
    """
    Returns model a + b (must be same shape).
    """
    r = []
    for i in range(len(a)):
        r.append([a[i][j] + b[i][j] for j in range(len(a[i]))])
    return r 
#Takes two models a and b (represented as 2D ragged lists of weight arrays) and returns their element-wise sum.
#

def minus_model_ws(a, b):
    """
    Returns model where a - b (must be same shape).
    """
    r = []
    for i in range(len(a)):
        r.append([a[i][j] - b[i][j] for j in range(len(a[i]))])
    return r 
#Takes two models a and b and returns their element-wise subtraction.

def multiply_model_ws(a, x):
    """
    Returns model a multiplied by constant x.
    """
    return [[(w * x).astype(w.dtype) for w in layer] for layer in a]
#akes a model a and a constant x and returns the model with all weights multiplied by x.


def divide_model_ws(a, x):
    """
    Returns model a divided by constant x.
    """
    return [[(w / x).astype(w.dtype) for w in layer] for layer in a]
#Takes a model a and a constant x and returns the model with all weights divided by x.

def zeros_like_model(a):
    """
    Returns list of layers, each element in layer a zero-array like those in a.
    """
    return [[np.zeros_like(w) for w in layer] for layer in a]
#Takes a model a and returns a model of the same shape with all zero arrays.

def add_optim_ws(a, b):
    """
    Returns optim a + b (must be same shape).
    """
    return [a[i] + b[i] for i in range(len(a))]
#Takes two optimizer vectors (1D lists of arrays) a and b and returns their element-wise sum.

def minus_optim_ws(a, b):
    """
    Returns optim a - b (must be same shape).
    """
    return [a[i] - b[i] for i in range(len(a))]
# Takes two optimizer vectors a and b and returns their element-wise subtraction.
    
def multiply_optim_ws(a, x):
    """
    Returns optim a multiplied by constant x.
    """
    return [(v * x).astype(v.dtype) for v in a]
#Takes an optimizer vector a and a constant x and returns the vector with all values multiplied by x.

def divide_optim_ws(a, x):
    """
    Returns optim a divided by constant x.
    """
    return [(v / x).astype(v.dtype) for v in a]
#Takes an optimizer vector a and a constant x and returns the vector with all values divided by x.
    
def zeros_like_optim(a):
    """
    Returns optim where each array is zeros like those in a.
    """
    return [np.zeros_like(v, dtype=v.dtype) for v in a]
#akes an optimizer vector a and returns a vector of zeros with the same shape and dtype.

def flatten_model(m):
    """
    Turns model m (2d ragged list of weight arrays) into a 1d list of arrays.
    """
    flat = []
    for layer in m:
        for w in layer:
            flat.append(w)
    return flat
#Takes a model m (2D ragged list of weight arrays) and flattens it into a 1D list of arrays.

def unflatten_model(m, ref):
    """
    Reshapes 1d list of arrays in m into 2d ragged list with same shape as ref.
    """
    unflat = []
    i = 0
    for layer in ref:
        unflat.append([])
        for w in layer:
            unflat[-1].append(m[i])
            i += 1
    return unflat
# Reshapes a flattened model m into a 2D ragged list with the same shape as a reference model ref.

def get_corr_optims(model):
    """
    TF does not seem to have an easy way of finding which optimizer arrays/
    values correspond to which weight array in the model. This function uses the
    name attributes TF generates for weights and optimizer values to return a 
    vector of length num_optim_params, where each value is the index of the 
    weight in the flattened model that that optimizer param is for. -1 values 
    indicate there was no matching weight (e.g. for the stored iteration number
    that Adam/RMSProp/etc use).
    
    Parameters:
    model (FedAvgModel):    model to generate match list for 
    
    Returns:
    vector of matches, see big description above
    """
    # tf generates unqiue names with / for scope ending with :i  
    w_names = [v.name for v in model.variables]
    w_names = [n.split(':')[0] for n in w_names]
    w_optim_names = [v.name for v in model.optimizer.variables()]
    
    # -1 means no corresponding weight value in model e.g. for iter num
    matches = -np.ones(len(w_optim_names), dtype=np.int32)
    
    # fimd the matching weight and place its index in matches
    for (i, w_optim_name) in enumerate(w_optim_names):
        for (j, w_name) in enumerate(w_names):
            if w_name in w_optim_name:
                matches[i] = j 
                break
        
    return matches

#what is ML model,what is layers ,how layers represent in matrix form
#what is optimizer vectors ?
#why we use this ?