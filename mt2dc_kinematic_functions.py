###############################################
# Mt2dc: An auxiliary file, in which are defined basic kinematic functions required in the numerical minimiser code (\textit{cf.} 2 below). Such functions (1) extract the $x, y, z-$momenta and energy from pT, $\eta$, $\phi$ and mass information, (2) calculate the transverse mass, momentum and energy of single or parent particles, using scalar and 4-vector input variables and (3) extract the values of the W and top terms from the numerical optimiser after it has been run. 
###############################################

import ROOT
import numpy as np 
import math 
import scipy.optimize as so 

# FUNCTIONS 
# Basic Variable Functions 
def mass_scalarCalc(px, py, pz, E): 
    """Calculate the mass of a particle, using scalar input variables.
    """
    m_squared = E**2 - (px**2 + py**2 + pz**2)
    
    if m_squared > 0:
        return np.sqrt(m_squared)
    return 0 

def ET_scalarCalc(m, pT):
    """Calculate the transverse energy of a particle, using scalar input variables.
    """
    ET = math.sqrt(max(0, m**2+pT**2)) 
    return ET

def mT_arrayCalc(p4_vis_array, p4_invis_array):
    """Calculate the transverse mass of a parent particle, using array input variables.
    Note: p4_(in)vis_array = [px, py, pz, E].
    """ 
    # Extract mass information
    m_vis = max(0, mass_scalarCalc(p4_vis_array[0], p4_vis_array[1], p4_vis_array[2], p4_vis_array[3]))
    m_invis = max(0, mass_scalarCalc(p4_invis_array[0], p4_invis_array[1], p4_invis_array[2], p4_invis_array[3]))
    
    # Get transverse momentum vectors 
    pT_vis_vec = np.array([p4_vis_array[0], p4_vis_array[1]])  
    pT_invis_vec = np.array([p4_invis_array[0], p4_invis_array[1]])
    
    # Extract energy information 
    ET_vis = ET_scalarCalc(m_invis, np.linalg.norm(pT_vis_vec)) 
    ET_invis = ET_scalarCalc(m_invis, np.linalg.norm(pT_invis_vec)) 
   
    mT = math.sqrt(max(0, m_vis**2 + m_invis**2 + 2*(ET_vis*ET_invis - np.dot(pT_vis_vec, pT_invis_vec))))
    return mT 

# TLorentz Module Functions 
def extract_Px(pT, eta, phi, mass): 
    """Extract Px, from scalar input variables. 
    """
    Px = pT*np.cos(phi) 
    return Px 

def extract_Py(pT, eta, phi, mass):
    """Extract Py, from scalar input variables.
    """
    Py = pT*np.sin(phi) 
    return Py 
    
def extract_Pz(pT, eta, phi, mass):
    """Extract Pz, from scalar input variables.
    """ 
    theta = 2*np.arctan(np.exp(-eta)) # polar angle 
    Pz = pT/np.tan(theta)
    return Pz 
    
def extract_E(pT, eta, phi, mass):
    """Extract E, from scalar input variables.
    """
    Pz = extract_Pz(pT, eta, phi, mass)
    E = math.sqrt(mass**2 + (pT**2 + Pz**2))
    return E 

# Optimisation Functions 
def get_alpha_term(vis_sideA_array, vis_sideB_array, met, invis_sideA_2vec):
    invis_sideA_array = np.array([invis_sideA_2vec[0], invis_sideA_2vec[1], 0, 
                                  np.sqrt(invis_sideA_2vec[0]**2 + invis_sideA_2vec[1]**2)]) 
    alpha_term_1 = mT_arrayCalc(vis_sideA_array[-1], invis_sideA_array) # mT(lA, pT_A)
    alpha_term_2 = mT_arrayCalc(vis_sideB_array[-1], met-invis_sideA_array) # mT(TB, pT_B) 
    alpha_term = max(alpha_term_1, alpha_term_2)
    return alpha_term 

def get_beta_term(vis_sideA_array, vis_sideB_array, met, invis_sideA_2vec):
    invis_sideA_array = np.array([invis_sideA_2vec[0], invis_sideA_2vec[1], 0, 
                                  np.sqrt(invis_sideA_2vec[0]**2 + invis_sideA_2vec[1]**2)]) 
    beta_term_1 = mT_arrayCalc(vis_sideA_array[0] + vis_sideA_array[-1], invis_sideA_array) # mT(lATA, pT_A)
    beta_term_2 = mT_arrayCalc(vis_sideB_array[0] + vis_sideB_array[-1], met-invis_sideA_array) # mT(TBbB, pt_B)
    beta_term = max(beta_term_1, beta_term_2) 
    return beta_term 
