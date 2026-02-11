import numpy as np
import pandas as pd
import re
from operator import itemgetter
import sys
sys.path.append("/cmnfs/home/j.lapin/projects/shabaz/data")
from mass_scale import Scale, tiebreak
scale = Scale()
IMMONIUM_IONS = [key for key in scale.mass.keys() if (key[0]=='I')&(len(key)>1)]

amino_acid_list = list("AVILMFYWSTNQCUGPRHKDEU")
def contains_amino_acid(token):

    # remove modiffication, if necessary
    token_ = re.sub(r"\[UNIMOD:[0-9]{1,3}]", '', token)
    return True if token_ in amino_acid_list else False

def convert_to_unimod(modseq):
    modseq = re.sub("\+42.011", "[UNIMOD:1]", modseq)
    modseq = re.sub("\+57.021", "[UNIMOD:4]", modseq)
    modseq = re.sub("\+43.006", "[UNIMOD:5]", modseq)
    modseq = re.sub("\+0.984", "[UNIMOD:7]", modseq)
    modseq = re.sub("\-17.027", "[UNIMOD:28]", modseq)
    modseq = re.sub("\+15.995", "[UNIMOD:35]", modseq)
    return modseq

def IONS(
    max_length, 
    max_charge, 
    ion_series=['b', 'y'], 
    neutral_losses=[],
    internal_neutral_losses=[],
    add_immonium=False, 
    add_precursor=False,
    add_internals=False,
    isotope_degree=0,
    custom_adds=[],
):
    ions = []
    charges  = [f'^{c}' for c in range(1, max_charge+1, 1)]
    charges[0]=''
    neutral_losses_ =  [''] + neutral_losses
    isotopes = [f"+{i}i" if i>1 else ('+i' if i==1 else '') for i in range(0,isotope_degree+1,1)]
    for ion in ion_series:
        for length in range(1, max_length, 1):
            for charge in charges:
                for nl in neutral_losses_:
                    for isotope in isotopes:
                        
                        nl_ = '-' + nl if nl != '' else nl
                        ions.append(f"{ion}{length}{nl_}{charge}{isotope}")
    if add_immonium:
        ions.extend(IMMONIUM_IONS)
    if add_precursor:
        ions.extend(['p', 'p+i', 'p^2'])
    if add_internals:
        ions.extend([f"Int{start}>{1}" for start in range(1, max_length-1, 1)])
        ions.extend([f"Int{start}>{2}" for start in range(1, max_length-2, 1)])
        ions.extend([f"Int{start}>{3}" for start in range(1, max_length-3, 1)])
        if len(internal_neutral_losses) > 0:
            for nl in internal_neutral_losses:
                ions.extend([f"Int{start}>{2}-{nl}" for start in range(1, max_length-2, 1)])
                ions.extend([f"Int{start}>{3}-{nl}" for start in range(1, max_length-3, 1)])
    for addon in custom_adds:
        if addon not in ions:
            ions.append(addon)
    
    ions = np.array(ions)
    return ions

def calc_masses(modseq, charge, ions):
    return np.array([scale.calcmass(modseq, charge, ion) for ion in ions])

def match(split_aa_sequence, charge, real_masses, IONS_kwargs={}, threshold=10, spl=[400,500], breaktie=True):
    real_masses = np.array(real_masses)
    real_masses = real_masses[real_masses!=0]
    ions = IONS(len(split_aa_sequence), charge, **IONS_kwargs)
    modseq = "".join(split_aa_sequence)
    modseq = convert_to_unimod(modseq)
    possible = calc_masses(modseq, charge, ions)
    TP,_,_ = scale.match(possible, real_masses, thr=threshold, spl=spl)
    if breaktie:
        theor_df = pd.DataFrame({'ion': ions, 'mz': possible})
        TP = tiebreak(TP, theor_df, real_masses)
    found_ions = ions[TP[0]]
    found_mzs = real_masses[TP[1]]
    return found_ions, found_mzs, TP[1]

def sort_array(array, axis_index, descending=False, sub_values=None, first=None):
    ndim = len(array.shape)

    # Put sub value in
    if sub_values is not None:
        array[array==sub_values[0]] = sub_values[1]
    # Find argument sort
    if ndim==2:
        argsort = array[axis_index].argsort()
    elif ndim==3:
        argsort = array[:,axis_index].argsort()
    if descending:
        argsort = argsort[::-1]
    # Return original value
    if sub_values is not None:
        array[array==sub_values[1]] = sub_values[0]
    
    # Only take first values
    if first is not None:
        argsort = argsort[:first]
    
    if ndim==1:
        return array[argsort]
    elif ndim==2:
        return array[:, argsort]
    elif ndim==3:
        return np.take_along_axis(array, argsort[:,None], -1)
    elif ndim=='last':
        return array[...,argsort]
