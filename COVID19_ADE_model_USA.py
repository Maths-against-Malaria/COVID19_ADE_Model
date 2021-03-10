#!/usr/bin/env python
# coding: utf-8

#############################################################################
# Library importation
#############################################################################

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

#############################################################################
# Parameter values
#############################################################################

# Total population
N = 331e6

# Total number of daily infections from the outside
lext = 50

# Durations of each disease stage E, P, I, L 
DE = 3.7
DP = 1
DI = 5
DL = 5

# Number of Erlang stages per disease stage
NE = 16
NP = 16
NI = 16
NL = 16

# Contagiousness at the different disease stages
cP = 0.5
cI = 1
cL = 0.5

# Denominator of Beta
cD = cP * DP + cI * DI + cL * DL

# Drifting rates
epsilon = NE / DE
varphi = NP / DP
gamma = NI / DI
delta = NL / DL

# Rate at which vaccination takes effect
alpha_vec = [1 / 28, 1 / 42]

# Rate at which individuals become vaccinated
nu_vec = [0, 1 / 180]  # , 1 / 240, 1 / 300]

# Proportion of unvaccinable indivaduals
pNV_vec = [0.6]  # [0, 0.20, 0.25, 0.30, 0.40, 0.60, 0.75]

# Day at which the vaccination campaign starts
tVaccin = [330, 345, 360, 370]

# Possible herd immunity thresholds (HITs)
HIT_vec = [0]

# Proportion fADE_S, fNI_S, fPI_S, and fR_S of susceptible individuals who become ADE, NI, PI, R after vaccination
fVac = np.array([[0.01, 0.02, 0.03, 0.94]])  # ,             # Good vaccine
                 #[0.01, 0.04, 0.05, 0.90],             # Medium
                 #[0.02, 0.10, 0.10, 0.78],             # Low
                 #[0.02, 0.24, 0.24, 0.50]])            # Poor vaccine

# Probability of showing symptoms
f_sick = 0.58

# Proportion of individuals that are isolated
fiso = 0.48

# Probability of dying from the disease
f_dead = 0.04

# Number of individuals in quarantine per 10,000 individuals
Qmax = 30 * N / 10000

# Effectiveness of home isolation
phome = 0.75

# Total simulation time
sim_time = 900

# NUmber of initial infections
N_init_inf = 75

# Period of sustainability of isolation measures (till the end of the simulation)
tiso1_vec = [0, 20]
tiso2 = sim_time

# Sustainability period for the general distancing measures
tdista = 50   # March    10, 2020
tdistb = 115  # May      14
tdistc = 190  # July     28
tdistd = 255  # October  01
tdiste = 290  # November 05
tdistf = 309  # November 24
tdistg = 316  # December 01
tdisth = 325  # December 10
tdisti = 335  # December 20
tdistj = 354  # January  08, 2021
tdistk = 450  # April    15

# Reproduction number
a = 0.35
R0bar = 3.2
tR0max = 335

# General contact reduction first lockdown, between the lockdowns, and the partial lockdown
pcont_a_vec = [0, 0.55]  # White house ban of gatherings (More than 10 persons)
pcont_b_vec = [0, 0.22]  # State economic reopening + Black Lives Matter (BLM) protest
pcont_c_vec = [0, 0.55]  # Experts warn for lockdowns necessity
pcont_d_vec = [0, 0.45]  # BLM continues + schools reopening + campaigns
pcont_e_vec = [0, 0.65]  # Lockdown
pcont_f_vec = [0, 0.55]  # Thanksgiving period (more travels)
pcont_g_vec = [0, 0.60]  # Less flights
pcont_h_vec = [0, 0.70]  # Lockdowns (few flights)
pcont_i_vec = [0, 0.55]  # Christmas period (family gatherings & more travels)
pcont_j_vec = [0, 0.65]  # Post Christmas period


# Probability of showing symptoms or dying when partially immunized (PI)
fPI_sick = 0.50
fPI_dead = 0.02

# Probability of showing symptoms or dying when being Antibody Dependent Enhanced (ADE)
fADE_sick = 0.92
fADE_dead_vec = [0.07, 0.2]  # 0.10, 0.15, 0.20]

# Probability of showing symptoms or dying while waiting for the vaccine to have effect
fIstar_sick = 0.58
fIstar_dead = 0.07
fLstar_sick = 0.58
fLstar_dead = 0.07

# Extra f parameters fUplus_I, fItilde_I, fItilde_L, fLtilde_L
fUplus_I_val = 0.005
fItilde_I = 0.95
fItilde_L = 0.95
fLtilde_L = 0.99

# Fraction at which the force of infection of PI and ADE individuals is reduced
P_pi = 0.5
P_ADE = 0.5

# Empty lists for the index dictionary
compartements = []
upperscripts = []

#############################################################################
# Functions definition
#############################################################################

def f_parameter(subscript, upperscript):
    if subscript == 'sick':
        if upperscript in ['', 'V', 'NI']:
            return f_sick
        elif upperscript == 'PI':
            return fPI_sick
        elif upperscript == 'ADE':
            return fADE_sick
        elif upperscript == 'Istar':
            return fIstar_sick
        elif upperscript == 'Lstar':
            return fLstar_sick
        else:
            print("Error: enter a correct values for the subscript and upperscript")
    elif subscript == 'dead':
        if upperscript == '':
            return f_dead
        elif upperscript == 'PI':
            return fPI_dead
        elif upperscript == 'ADE':
            return fADE_dead
        elif upperscript == 'Istar':
            return fIstar_dead
        elif upperscript == 'Lstar':
            return fLstar_dead
        else:
            print("Error: enter a correct values for the subscript and upperscript")
    elif subscript == 'S':
        if upperscript == 'R':
            return fR_S
        elif upperscript == 'NI':
            return fNI_S
        elif upperscript == 'PI':
            return fPI_S
        elif upperscript == 'ADE':
            return fADE_S
        else:
            print("Error: enter a correct values for the subscript and upperscript")
    elif subscript == 'E':
        if upperscript == 'R':
            return fR_E
        elif upperscript == 'NI':
            return fNI_E
        elif upperscript == 'PI':
            return fPI_E
        elif upperscript == 'ADE':
            return fADE_E
        else:
            print("Error: enter a correct values for the subscript and upperscript")
    elif subscript == 'P':
        if upperscript == 'R':
            return fR_P
        elif upperscript == 'NI':
            return fNI_P
        elif upperscript == 'PI':
            return fPI_P
        elif upperscript == 'ADE':
            return fADE_P
        else:
            print("Error: enter a correct values for the subscript and upperscript")
    elif subscript == 'I':
        if upperscript == 'R':
            return fR_I
        elif upperscript == 'NI':
            return fNI_I
        elif upperscript == 'PI':
            return fPI_I
        elif upperscript == 'ADE':
            return fADE_I
        elif upperscript == 'Itilde':
            return fItilde_I
        else:
            print("Error: enter a correct values for the subscript and upperscript")
    elif subscript == 'L':
        if upperscript == 'R':
            return fR_L
        elif upperscript == 'NI':
            return fNI_L
        elif upperscript == 'PI':
            return fPI_L
        elif upperscript == 'ADE':
            return fADE_L
        elif upperscript == 'Itilde':
            return fItilde_L
        elif upperscript == 'Ltilde':
            return fLtilde_L
        else:
            print("Error: enter a correct values for the subscript and upperscript")

def indexfunction(NE, NP, NI, NL):
    """
    This Code  Build a map which associate to a population name of the model  an
    index, in order to create a list that will record the dynamics of the populations.

    We first created a dictionary named "index" which will has as keys the names of
    populations and as  vallues the associated index.

    The illustration of the model can be found at
    
    We decided to go row wise to attribute the index corresponding to eah population.
    
    To do so, we first created
    
    compartments: a list that contains 3-uplets. Each 3-uplets are made from the combination of:
    names of the stage [S, E, P, I, L, D, RIn, RIm] of the disease, the number of different classes at the corresponding disease stage
    and the corresponding number of Erlang stage associated with it.
    
    upperscript: a list cointaining all the upperscripts used to descript the different
    populations. this is list is strategically sorted because some stages don't
    end up with all the different upperscripts. We sorted this list by the frequency
    of use of the upperscript. for instance, in our model the upperscript 'NI'
    is almost used in all the stages.

    output desired ==> index : dictionary
    
    
    """

    global compartments, upperscripts

    compartments = [('S', 5, 0), ('E', 5, NE), ('P', 5, NP), ('I', 10, NI), ('L', 13, NL), ('D', 0, 0), ('RInf', 0, 0),
                    ('RVac', 0, 0)]  # E, P have 5 compartments
    upperscripts = ['NV', 'U', 'V', 'NI', 'PI', 'ADE', 'U-', 'U+', 'IV', 'Itilde', 'Istar', 'LV', 'Ltilde', 'Lstar']
    index = dict()
    ind = 0
    for i in compartments:
        notation = i[0]
        # in case, there is no upperscripts we just update the index
        if i[1] == 0:
            index[notation] = ind
            ind += 1
        else:
            for j in upperscripts[: i[1] + 1]:
                # Remember that the compartments I and L don't have the upperscript (U)
                # we are ignoring to concatenate U to the compartments I and L
                exception = ((i[0] in ['I', 'L']) and (j in ['U'])) or (not (i[0] in ['S']) and (j in ['NV']))
                if exception:
                    continue
                notation = i[0] + '^' + j
                # in case, there is no subscripts we just update the index
                if i[2] == 0:
                    index[notation] = ind
                    ind += 1
                else:
                    for k in list(range(1, i[2] + 1)):
                        notation = i[0] + '^' + j + '_' + str(k)
                        index[notation] = ind
                        ind += 1
    return index

# reproduction number with seasonal fluctuations and effects of general distancing
def R0(t):
    x = R0bar * (1 + a * np.cos(2 * np.pi * (t - tR0max) / 365))
    return x

# Beta parameter
def beta(t, inp_var1, inp_var2):
    return inp_var1 * R0(t) / inp_var2

# Contact reduction
def pCont(t):
    if tdista <= t < tdistb:
        return pcont_a
    elif tdistb <= t < tdistc:
        return pcont_b
    elif tdistc <= t < tdistd:
        return pcont_c
    elif tdistd <= t < tdiste:
        return pcont_d
    elif tdiste <= t < tdistf:
        return pcont_e
    elif tdistf <= t < tdistg:
        return pcont_f
    elif tdistg <= t < tdisth:
        return pcont_g
    elif tdisth <= t < tdisti:
        return pcont_h
    elif tdisti <= t < tdistj:
        return pcont_i
    elif tdistj <= t < tdistk:
        return pcont_j
    else:
        return 0

# Time dependent nu
def func_Nu(t, inp_var):
    if t >= tVac:
        return inp_var
    else:
        return 0

def func_fUplus_I(t, inp_var):
    if t >= tVac:
        return inp_var
    else:
        return 0

# Total number of individuals in the different divisions of population
def popsum(compartment, upperscript, Nerlangs, t, pop):
    x = 0
    for k in range(1, Nerlangs + 1):
        x = x + pop[index[compartment + '^' + upperscript + '_' + str(k)]]
    return x


# Total number of individuals that can effectively infect susceptible individuals
def popeff(compartment, upperscript, Nerlangs, t, pop, fUplus_I, fUplus_sick):
    return popsum(compartment, upperscript, Nerlangs, t, pop) - popiso(compartment, upperscript, Nerlangs, t,
                                                                       pop, fUplus_I, fUplus_sick) - phome * pophome(compartment, upperscript,
                                                                                                                     Nerlangs, t, pop, fUplus_I, fUplus_sick)


# Thus the numbers of symptomatic infections in the fully contagious states and the number of individuals in the late
# infectious states
def popsick(compartment, upperscript, Nerlangs, t, pop, fUplus_I, fUplus_sick):
    return f_parameter(subscript='sick', upperscript=upperscript) * popsum(compartment, upperscript, Nerlangs, t, pop)


# Force of infection is defined by
def l(t, pop, fUplus_I, fUplus_sick):
    lambdaP, lambdaI, lambdaL = 0, 0, 0

    for i in ['U', 'V', 'NI', 'PI', 'ADE']:
        lambdaP = lambdaP + popsum('P', i, NP, t, pop)

    for i in ['U-', 'IV', 'Itilde']:
        lambdaI = lambdaI + popsum('I', i, NI, t, pop)

    for i in ['V', 'U+', 'NI', 'PI', 'ADE', 'Istar']:
        lambdaI = lambdaI + popeff('I', i, NI, t, pop, fUplus_I, fUplus_sick)

    for i in ['U-', 'IV', 'Itilde', 'LV', 'Ltilde']:
        lambdaL = lambdaL + popsum('L', i, NL, t, pop)

    for i in ['V', 'U+', 'NI', 'PI', 'ADE', 'Istar', 'Lstar']:
        lambdaL = lambdaL + popeff('L', i, NL, t, pop, fUplus_I, fUplus_sick)

    return (beta(t, cP, cD) * lambdaP + beta(t, cI, cD) * lambdaI + beta(t, cL, cD) * lambdaL + lext) * (
            1 - pCont(t)) / N


# Total number of individuals isolated in general quarantine wards
def Q(t, pop, fUplus_I, fUplus_sick):
    Q1 = (fUplus_sick * fiso + (1 - fUplus_sick)) * (
            popsum('I', 'U+', NI, t, pop) + popsum('L', 'U+', NL, t, pop)) + fiso * (
                 popsick('I', 'V', NI, t, pop, fUplus_I, fUplus_sick) + popsick('L', 'V', NL, t, pop, fUplus_I, fUplus_sick)) + fiso * (
                 popsick('I', 'NI', NI, t, pop, fUplus_I, fUplus_sick) + popsick('L', 'NI', NL, t, pop, fUplus_I, fUplus_sick))
    Q2 = fiso * (popsick('I', 'PI', NI, t, pop, fUplus_I, fUplus_sick) + popsick('L', 'PI', NL, t, pop, fUplus_I, fUplus_sick)) + fiso * (
            popsick('I', 'ADE', NI, t, pop, fUplus_I, fUplus_sick) + popsick('L', 'ADE', NL, t, pop, fUplus_I, fUplus_sick)) + fiso * (
                 popsick('I', 'Istar', NI, t, pop, fUplus_I, fUplus_sick) + popsick('L', 'Istar', NL, t, pop, fUplus_I, fUplus_sick)) + fiso * popsick('L','Lstar',NL, t, pop, fUplus_I, fUplus_sick)
    return Q1 + Q2


# Total number of individuals isolated in general quarantine wards
def popiso(compartment, upperscript, Nerlangs, t, pop, fUplus_I, fUplus_sick):
    if upperscript == 'U+':
        if (tiso1 <= t <= tiso2) and (Q(t, pop, fUplus_I, fUplus_sick) <= Qmax):
            return (fUplus_sick * fiso + (1 - fUplus_sick)) * popsum(compartment, upperscript, Nerlangs, t, pop)
        elif (tiso1 <= t <= tiso2) and (Q(t, pop, fUplus_I, fUplus_sick) > Qmax):
            return (fUplus_sick * fiso + (1 - fUplus_sick)) * popsum(compartment, upperscript, Nerlangs, t, pop) * (
                    Qmax / Q(t, pop, fUplus_I, fUplus_sick))
        else:
            return 0
    elif upperscript in ['NI', 'PI', 'ADE', 'Istar']:
        if (tiso1 <= t <= tiso2) and (Q(t, pop, fUplus_I, fUplus_sick) <= Qmax):
            return fiso * popsick(compartment, upperscript, Nerlangs, t, pop, fUplus_I, fUplus_sick)
        elif (tiso1 <= t <= tiso2) and (Q(t, pop, fUplus_I, fUplus_sick) > Qmax):
            return fiso * popsick(compartment, upperscript, Nerlangs, t, pop, fUplus_I, fUplus_sick) * (Qmax / Q(t, pop, fUplus_I, fUplus_sick))
        else:
            return 0
    elif upperscript == 'Lstar' and compartment == 'L':
        if (tiso1 <= t <= tiso2) and (Q(t, pop, fUplus_I, fUplus_sick) <= Qmax):
            return fiso * popsick(compartment, upperscript, Nerlangs, t, pop, fUplus_I, fUplus_sick)
        elif (tiso1 <= t <= tiso2) and (Q(t, pop, fUplus_I, fUplus_sick) > Qmax):
            return fiso * popsick(compartment, upperscript, Nerlangs, t, pop, fUplus_I, fUplus_sick) * (Qmax / Q(t, pop, fUplus_I, fUplus_sick))
        else:
            return 0
    else:
        return 0


# Total number of individuals  isolated at home
def pophome(compartment, upperscript, Nerlangs, t, pop, fUplus_I, fUplus_sick):
    if upperscript == 'U+':
        if (tiso1 <= t <= tiso2) and (Q(t, pop, fUplus_I, fUplus_sick) > Qmax):
            return (fUplus_sick * fiso + (1 - fUplus_sick)) * popsum(compartment, upperscript, Nerlangs, t, pop) * (
                    1 - Qmax / Q(t, pop, fUplus_I, fUplus_sick))
        else:
            return 0
    elif upperscript in ['V', 'NI', 'PI', 'ADE', 'Istar']:
        if (tiso1 <= t <= tiso2) and (Q(t, pop, fUplus_I, fUplus_sick) > Qmax):
            return popsick(compartment, upperscript, Nerlangs, t, pop, fUplus_I, fUplus_sick) * fiso * (1 - Qmax / Q(t, pop, fUplus_I, fUplus_sick))
        else:
            return 0
    elif upperscript == 'Lstar' and compartment == 'L':
        if (tiso1 <= t <= tiso2) and (Q(t, pop, fUplus_I, fUplus_sick) > Qmax):
            return popsick(compartment, upperscript, Nerlangs, t, pop, fUplus_I, fUplus_sick) * fiso * (1 - Qmax / Q(t, pop, fUplus_I, fUplus_sick))
        else:
            return 0
    else:
        return 0


## Mapping the compatments
index = indexfunction(NE, NP, NI, NL)


#############################################################################
######################################################## System of ODEs
#############################################################################

# The dynamic of susceptible individuals is described by the following differential equations
def dSNV(t, pop, lambdat, fUplus_I, fUplus_sick):
    x = - lambdat * pop[index['S^NV']]
    return x


def dSU(t, pop, lambdat, fUplus_I, fUplus_sick):
    x = - lambdat * pop[index['S^U']] - func_Nu(t, nu) * pop[index['S^U']]
    return x


def dSV(t, pop, lambdat, fUplus_I, fUplus_sick):
    x = func_Nu(t, nu) * pop[index['S^U']] - lambdat * pop[index['S^V']] - alpha * pop[index['S^V']]
    return x


def dSNI(t, pop, lambdat, fUplus_I, fUplus_sick):
    x = alpha * fNI_S * pop[index['S^V']] - lambdat * pop[index['S^NI']]
    return x


def dSPI(t, pop, lambdat, fUplus_I, fUplus_sick):
    x = alpha * fPI_S * pop[index['S^V']] - lambdat * pop[index['S^PI']]
    return x


def dSADE(t, pop, lambdat, fUplus_I, fUplus_sick):
    x = alpha * fADE_S * pop[index['S^V']] - lambdat * pop[index['S^ADE']]
    return x


# The dynamic of latent individuals is described by the following differential equations
def dEU(t, k, pop, lambdat, fUplus_I, fUplus_sick):
    if k == 1:
        x = lambdat * pop[index['S^U']] - func_Nu(t, nu) * pop[index['E^U_' + str(k)]] - epsilon * pop[
            index['E^U_' + str(k)]]
        return x
    else:
        x = epsilon * pop[index['E^U_' + str(k - 1)]] - epsilon * pop[index['E^U_' + str(k)]] - func_Nu(t, nu) * pop[
            index['E^U_' + str(k)]]
        return x


def dEV(t, k, pop, lambdat, fUplus_I, fUplus_sick):
    if k == 1:
        x = lambdat * pop[index['S^V']] + func_Nu(t, nu) * pop[index['E^U_' + str(k)]] - epsilon * pop[
            index['E^V_' + str(k)]] - alpha * pop[index['E^V_' + str(k)]]
        return x
    else:
        x = epsilon * pop[index['E^V_' + str(k - 1)]] + func_Nu(t, nu) * pop[index['E^U_' + str(k)]] - epsilon * pop[
            index['E^V_' + str(k)]] - alpha * pop[index['E^V_' + str(k)]]
        return x


def dENI(t, k, pop, lambdat, fUplus_I, fUplus_sick):
    if k == 1:
        x = lambdat * (pop[index['S^NV']] + pop[index['S^NI']]) + alpha * fNI_E * pop[
            index['E^V_' + str(k)]] - epsilon * pop[index['E^NI_' + str(k)]]
        return x
    else:
        x = epsilon * pop[index['E^NI_' + str(k - 1)]] + alpha * fNI_E * pop[index['E^V_' + str(k)]] - epsilon * pop[
            index['E^NI_' + str(k)]]
        return x


def dEPI(t, k, pop, lambdat, fUplus_I, fUplus_sick):
    if k == 1:
        x = P_pi * lambdat * pop[index['S^PI']] + alpha * fPI_E * pop[index['E^V_' + str(k)]] - epsilon * pop[
            index['E^PI_' + str(k)]]
        return x
    else:
        x = epsilon * pop[index['E^PI_' + str(k - 1)]] + alpha * fPI_E * pop[index['E^V_' + str(k)]] - epsilon * pop[
            index['E^PI_' + str(k)]]
        return x


def dEADE(t, k, pop, lambdat, fUplus_I, fUplus_sick):
    if k == 1:
        x = P_ADE * lambdat * pop[index['S^ADE']] + alpha * fADE_E * pop[index['E^V_' + str(k)]] - epsilon * pop[
            index['E^ADE_' + str(k)]]
        return x
    else:
        x = epsilon * pop[index['E^ADE_' + str(k - 1)]] + alpha * fADE_E * pop[index['E^V_' + str(k)]] - epsilon * pop[
            index['E^ADE_' + str(k)]]
        return x


# The dynamic of prodromal individuals is described by the following differential equations
def dPU(t, k, pop, lambdat, fUplus_I, fUplus_sick):
    if k == 1:
        x = epsilon * pop[index['E^U_' + str(NE)]] - varphi * pop[index['P^U_' + str(k)]] - func_Nu(t, nu) * pop[
            index['P^U_' + str(k)]]
        return x
    else:
        x = varphi * pop[index['P^U_' + str(k - 1)]] - varphi * pop[index['P^U_' + str(k)]] - func_Nu(t, nu) * pop[
            index['P^U_' + str(k)]]
        return x


def dPV(t, k, pop, lambdat, fUplus_I, fUplus_sick):
    if k == 1:
        x = epsilon * pop[index['E^V_' + str(NE)]] + func_Nu(t, nu) * pop[index['P^U_' + str(k)]] - varphi * pop[
            index['P^V_' + str(k)]] - alpha * pop[index['P^V_' + str(k)]]
        return x
    else:
        x = varphi * pop[index['P^V_' + str(k - 1)]] + func_Nu(t, nu) * pop[index['P^U_' + str(k)]] - varphi * pop[
            index['P^V_' + str(k)]] - alpha * pop[index['P^V_' + str(k)]]
        return x


def dPNI(t, k, pop, lambdat, fUplus_I, fUplus_sick):
    if k == 1:
        x = epsilon * pop[index['E^NI_' + str(NE)]] + alpha * fNI_P * pop[index['P^V_' + str(k)]] - varphi * pop[
            index['P^NI_' + str(k)]]
        return x
    else:
        x = varphi * pop[index['P^NI_' + str(k - 1)]] + alpha * fNI_P * pop[index['P^V_' + str(k)]] - varphi * pop[
            index['P^NI_' + str(k)]]
        return x


def dPPI(t, k, pop, lambdat, fUplus_I, fUplus_sick):
    if k == 1:
        x = epsilon * pop[index['E^PI_' + str(NE)]] + alpha * fPI_P * pop[index['P^V_' + str(k)]] - varphi * pop[
            index['P^PI_' + str(k)]]
        return x
    else:
        x = varphi * pop[index['P^PI_' + str(k - 1)]] + alpha * fPI_P * pop[index['P^V_' + str(k)]] - varphi * pop[
            index['P^PI_' + str(k)]]
        return x


def dPADE(t, k, pop, lambdat, fUplus_I, fUplus_sick):
    if k == 1:
        x = epsilon * pop[index['E^ADE_' + str(NE)]] + alpha * fADE_P * pop[index['P^V_' + str(k)]] - varphi * pop[
            index['P^ADE_' + str(k)]]
        return x
    else:
        x = varphi * pop[index['P^ADE_' + str(k - 1)]] + alpha * fADE_P * pop[index['P^V_' + str(k)]] - varphi * pop[
            index['P^ADE_' + str(k)]]
        return x

    # The dynamic of fully infectious individuals is described by the following differential equations


def dIUplus(t, k, pop, lambdat, fUplus_I, fUplus_sick):
    if k == 1:
        x = varphi * (f_sick + (1 - f_sick) * fUplus_I) * pop[index['P^U_' + str(NP)]] - gamma * pop[
            index['I^U+_' + str(k)]]
        return x
    else:
        x = gamma * pop[index['I^U+_' + str(k - 1)]] - gamma * pop[index['I^U+_' + str(k)]]
        return x


def dIUminus(t, k, pop, lambdat, fUplus_I, fUplus_sick):
    if k == 1:
        x = varphi * (1 - f_sick - fUplus_I + f_sick * fUplus_I) * pop[index['P^U_' + str(NP)]] - gamma * pop[
            index['I^U-_' + str(k)]] - func_Nu(t, nu) * pop[index['I^U-_' + str(k)]]
        return x
    else:
        x = gamma * pop[index['I^U-_' + str(k - 1)]] - gamma * pop[index['I^U-_' + str(k)]] - func_Nu(t, nu) * pop[
            index['I^U-_' + str(k)]]
        return x


def dIV(t, k, pop, lambdat, fUplus_I, fUplus_sick):
    if k == 1:
        x = varphi * pop[index['P^V_' + str(NP)]] - gamma * pop[index['I^V_' + str(k)]] - alpha * pop[
            index['I^V_' + str(k)]]
        return x
    else:
        x = gamma * pop[index['I^V_' + str(k - 1)]] - gamma * pop[index['I^V_' + str(k)]] - alpha * pop[
            index['I^V_' + str(k)]]
        return x


def dINI(t, k, pop, lambdat, fUplus_I, fUplus_sick):
    if k == 1:
        x = varphi * pop[index['P^NI_' + str(NP)]] + alpha * fNI_I * pop[index['I^V_' + str(k)]] - gamma * pop[
            index['I^NI_' + str(k)]]
        return x
    else:
        x = gamma * pop[index['I^NI_' + str(k - 1)]] + alpha * fNI_I * pop[index['I^V_' + str(k)]] - gamma * pop[
            index['I^NI_' + str(k)]]
        return x


def dIPI(t, k, pop, lambdat, fUplus_I, fUplus_sick):
    if k == 1:
        x = varphi * pop[index['P^PI_' + str(NP)]] + alpha * fPI_I * pop[index['I^V_' + str(k)]] - gamma * pop[
            index['I^PI_' + str(k)]]
        return x
    else:
        x = gamma * pop[index['I^PI_' + str(k - 1)]] + alpha * fPI_I * pop[index['I^V_' + str(k)]] - gamma * pop[
            index['I^PI_' + str(k)]]
        return x


def dIADE(t, k, pop, lambdat, fUplus_I, fUplus_sick):
    if k == 1:
        x = varphi * pop[index['P^ADE_' + str(NP)]] + alpha * fADE_I * pop[index['I^V_' + str(k)]] - gamma * pop[
            index['I^ADE_' + str(k)]]
        return x
    else:
        x = gamma * pop[index['I^ADE_' + str(k - 1)]] + alpha * fADE_I * pop[index['I^V_' + str(k)]] - gamma * pop[
            index['I^ADE_' + str(k)]]
        return x


def dIIV(t, k, pop, lambdat, fUplus_I, fUplus_sick):
    if k == 1:
        x = func_Nu(t, nu) * pop[index['I^U-_' + str(k)]] - gamma * pop[index['I^IV_' + str(k)]] - alpha * pop[
            index['I^IV_' + str(k)]]
        return x
    else:
        x = gamma * pop[index['I^IV_' + str(k - 1)]] + func_Nu(t, nu) * pop[index['I^U-_' + str(k)]] - gamma * pop[
            index['I^IV_' + str(k)]] - alpha * pop[index['I^IV_' + str(k)]]
        return x


def dIItilde(t, k, pop, lambdat, fUplus_I, fUplus_sick):
    if k == 1:
        x = alpha * fItilde_I * pop[index['I^IV_' + str(k)]] - gamma * pop[index['I^Itilde_' + str(k)]]
        return x
    else:
        x = gamma * pop[index['I^Itilde_' + str(k - 1)]] + alpha * fItilde_I * pop[index['I^IV_' + str(k)]] - gamma * \
            pop[index['I^Itilde_' + str(k)]]
        return x


def dIIstar(t, k, pop, lambdat, fUplus_I, fUplus_sick):
    if k == 1:
        x = alpha * (1 - fItilde_I) * pop[index['I^IV_' + str(k)]] - gamma * pop[index['I^Istar_' + str(k)]]
        return x
    else:
        x = gamma * pop[index['I^Istar_' + str(k - 1)]] + alpha * (1 - fItilde_I) * pop[
            index['I^IV_' + str(k)]] - gamma * pop[index['I^Istar_' + str(k)]]
        return x

    # The dynamic of late infectious individuals is described by the following differential equations


def dLUplus(t, k, pop, lambdat, fUplus_I, fUplus_sick):
    if k == 1:
        x = gamma * pop[index['I^U+_' + str(NI)]] - delta * pop[index['L^U+_' + str(k)]]
        return x
    else:
        x = delta * pop[index['L^U+_' + str(k - 1)]] - delta * pop[index['L^U+_' + str(k)]]
        return x


def dLUminus(t, k, pop, lambdat, fUplus_I, fUplus_sick):
    if k == 1:
        x = gamma * pop[index['I^U-_' + str(NI)]] - delta * pop[index['L^U-_' + str(k)]] - func_Nu(t, nu) * pop[
            index['L^U-_' + str(k)]]
        return x
    else:
        x = delta * pop[index['L^U-_' + str(k - 1)]] - delta * pop[index['L^U-_' + str(k)]] - func_Nu(t, nu) * pop[
            index['L^U-_' + str(k)]]
        return x


def dLV(t, k, pop, lambdat, fUplus_I, fUplus_sick):
    if k == 1:
        x = gamma * pop[index['I^V_' + str(NI)]] - delta * pop[index['L^V_' + str(k)]] - alpha * pop[
            index['L^V_' + str(k)]]
        return x
    else:
        x = delta * pop[index['L^V_' + str(k - 1)]] - delta * pop[index['L^V_' + str(k)]] - alpha * pop[
            index['L^V_' + str(k)]]
        return x


def dLNI(t, k, pop, lambdat, fUplus_I, fUplus_sick):
    if k == 1:
        x = gamma * pop[index['I^NI_' + str(NI)]] + alpha * fNI_L * pop[index['L^V_' + str(k)]] - delta * pop[
            index['L^NI_' + str(k)]]
        return x
    else:
        x = delta * pop[index['L^NI_' + str(k - 1)]] + alpha * fNI_L * pop[index['L^V_' + str(k)]] - delta * pop[
            index['L^NI_' + str(k)]]
        return x


def dLPI(t, k, pop, lambdat, fUplus_I, fUplus_sick):
    if k == 1:
        x = gamma * pop[index['I^PI_' + str(NI)]] + alpha * fPI_L * pop[index['L^V_' + str(k)]] - delta * pop[
            index['L^PI_' + str(k)]]
        return x
    else:
        x = delta * pop[index['L^PI_' + str(k - 1)]] + alpha * fPI_L * pop[index['L^V_' + str(k)]] - delta * pop[
            index['L^PI_' + str(k)]]
        return x


def dLADE(t, k, pop, lambdat, fUplus_I, fUplus_sick):
    if k == 1:
        x = gamma * pop[index['I^ADE_' + str(NI)]] + alpha * fADE_L * pop[index['L^V_' + str(k)]] - delta * pop[
            index['L^ADE_' + str(k)]]
        return x
    else:
        x = delta * pop[index['L^ADE_' + str(k - 1)]] + alpha * fADE_L * pop[index['L^V_' + str(k)]] - delta * pop[
            index['L^ADE_' + str(k)]]
        return x


def dLIV(t, k, pop, lambdat, fUplus_I, fUplus_sick):
    if k == 1:
        x = gamma * pop[index['I^IV_' + str(NI)]] - delta * pop[index['L^IV_' + str(k)]] - alpha * pop[
            index['L^IV_' + str(k)]]
        return x
    else:
        x = delta * pop[index['L^IV_' + str(k - 1)]] - delta * pop[index['L^IV_' + str(k)]] - alpha * pop[
            index['L^IV_' + str(k)]]
        return x


def dLItilde(t, k, pop, lambdat, fUplus_I, fUplus_sick):
    if k == 1:
        x = gamma * pop[index['I^Itilde_' + str(NI)]] + alpha * fItilde_L * pop[index['L^IV_' + str(k)]] - delta * pop[
            index['L^Itilde_' + str(k)]]
        return x
    else:
        x = delta * pop[index['L^Itilde_' + str(k - 1)]] + alpha * fItilde_L * pop[index['L^IV_' + str(k)]] - delta * \
            pop[index['L^Itilde_' + str(k)]]
        return x


def dLIstar(t, k, pop, lambdat, fUplus_I, fUplus_sick):
    if k == 1:
        x = gamma * pop[index['I^Istar_' + str(NI)]] + alpha * (1 - fItilde_L) * pop[index['L^IV_' + str(k)]] - delta * \
            pop[index['L^Istar_' + str(k)]]
        return x
    else:
        x = delta * pop[index['L^Istar_' + str(k - 1)]] + alpha * (1 - fItilde_L) * pop[
            index['L^IV_' + str(k)]] - delta * pop[index['L^Istar_' + str(k)]]
        return x


def dLLV(t, k, pop, lambdat, fUplus_I, fUplus_sick):
    if k == 1:
        x = func_Nu(t, nu) * pop[index['L^U-_' + str(k)]] - delta * pop[index['L^LV_' + str(k)]] - alpha * pop[
            index['L^LV_' + str(k)]]
        return x
    else:
        x = delta * pop[index['L^LV_' + str(k - 1)]] + func_Nu(t, nu) * pop[index['L^U-_' + str(k)]] - delta * pop[
            index['L^LV_' + str(k)]] - alpha * pop[index['L^LV_' + str(k)]]
        return x


def dLLtilde(t, k, pop, lambdat, fUplus_I, fUplus_sick):
    if k == 1:
        x = alpha * fLtilde_L * pop[index['L^LV_' + str(k)]] - delta * pop[index['L^Ltilde_' + str(k)]]
        return x
    else:
        x = delta * pop[index['L^Ltilde_' + str(k - 1)]] + alpha * fLtilde_L * pop[index['L^LV_' + str(k)]] - delta * \
            pop[index['L^Ltilde_' + str(k)]]
        return x


def dLLstar(t, k, pop, lambdat, fUplus_I, fUplus_sick):
    if k == 1:
        x = alpha * (1 - fLtilde_L) * pop[index['L^LV_' + str(k)]] - delta * pop[index['L^Lstar_' + str(k)]]
        return x
    else:
        x = delta * pop[index['L^Lstar_' + str(k - 1)]] + alpha * (1 - fLtilde_L) * pop[
            index['L^LV_' + str(k)]] - delta * pop[index['L^Lstar_' + str(k)]]
        return x

    # The death toll over time is described by the following equations


def dD(t, pop, lambdat, fUplus_I, fUplus_sick):
    dD1 = fUplus_sick * f_dead * pop[index['L^U+_' + str(NL)]] + f_sick * f_dead * pop[
        index['L^V_' + str(NL)]] + f_sick * f_dead * pop[index['L^NI_' + str(NL)]] + fPI_sick * fPI_dead * pop[
              index['L^PI_' + str(NL)]]
    dD2 = fADE_sick * fADE_dead * pop[index['L^ADE_' + str(NL)]] + fIstar_sick * fIstar_dead * pop[
        index['L^Istar_' + str(NL)]] + fLstar_sick * fLstar_dead * pop[index['L^Lstar_' + str(NL)]]
    return delta * (dD1 + dD2)


# The dynamic in the recovered population is described by the following equations
# Immune or Protected ('Im': for Immunize to avoid mixup with 'I' and 'NI') fully from the disease when vaccinated
def dRVac(t, pop, lambdat, fUplus_I, fUplus_sick):
    dR1 = alpha * fR_S * pop[index['S^V']] + alpha * fR_E * popsum('E', 'V', NE, t, pop) + alpha * fR_P * popsum('P','V', NP, t, pop) \
          + alpha * fR_I * popsum('I', 'V', NI, t, pop) + alpha * fR_L * popsum('L', 'V', NL, t, pop)
    return dR1


# Undergo infection episode then recover from the infection and may or may not be vaccinated at a point in time
def dRInf(t, pop, lambdat, fUplus_I, fUplus_sick):
    dR2 = delta * (1 - fUplus_sick * f_dead) * pop[index['L^U+_' + str(NL)]]
    dR3 = delta * pop[index['L^U-_' + str(NL)]] + delta * (1 - f_sick * f_dead) * pop[
        index['L^V_' + str(NL)]] + delta * (1 - f_sick * f_dead) * pop[index['L^NI_' + str(NL)]] + delta * (
                  1 - fPI_sick * fPI_dead) * pop[index['L^PI_' + str(NL)]] + delta * (1 - fADE_sick * fADE_dead) * \
          pop[index['L^ADE_' + str(NL)]]
    dR4 = delta * pop[index['L^IV_' + str(NL)]] + delta * pop[index['L^Itilde_' + str(NL)]] + delta * (
            1 - fIstar_sick * fIstar_dead) * pop[index['L^Istar_' + str(NL)]] + delta * pop[
              index['L^LV_' + str(NL)]] + delta * pop[index['L^Ltilde_' + str(NL)]] + delta * (
                  1 - fLstar_sick * fLstar_dead) * pop[index['L^Lstar_' + str(NL)]]
    return dR2 + dR3 + dR4

#############################################################################
# Solving the ODEs
#############################################################################

ODEName = {'S^NV': dSNV, 'S^U': dSU, 'S^V': dSV, 'S^NI': dSNI, 'S^PI': dSPI, 'S^ADE': dSADE, 'E^U': dEU, 'E^V': dEV,
           'E^NI': dENI, 'E^PI': dEPI, 'E^ADE': dEADE,
           'P^U': dPU, 'P^V': dPV, 'P^NI': dPNI, 'P^PI': dPPI, 'P^ADE': dPADE, 'I^U+': dIUplus, 'I^U-': dIUminus,
           'I^V': dIV, 'I^NI': dINI, 'I^PI': dIPI, 'I^ADE': dIADE, 'I^IV': dIIV,
           'I^Itilde': dIItilde, 'I^Istar': dIIstar, 'L^U+': dLUplus, 'L^U-': dLUminus, 'L^V': dLV, 'L^NI': dLNI,
           'L^PI': dLPI, 'L^ADE': dLADE, 'L^IV': dLIV,
           'L^Itilde': dLItilde, 'L^Istar': dLIstar, 'L^LV': dLLV, 'L^Ltilde': dLLtilde, 'L^Lstar': dLLstar, 'D': dD,
           'RInf': dRInf, 'RVac': dRVac}
notation_list_k = list(index.keys())[6:-3]
notation_list_without_k = list(index.keys())[:6] + list(index.keys())[-3:]
notation_list_k_split = [(notation.split('_')[0], int(notation.split('_')[1])) for notation in notation_list_k]

def f(t, pop, alpha):
    """This function solves the ODES"""
    fUplus_I = func_fUplus_I(t, fUplus_I_const)
    fUplus_sick = f_sick / (f_sick + (1 - f_sick) * fUplus_I)

    dpop = [0 for i in np.arange(len(index))]
    lambdat = l(t, pop, fUplus_I, fUplus_sick)

    dpop1 = [ODEName[notation](t, pop, lambdat, fUplus_I, fUplus_sick) for notation in notation_list_without_k]
    dpop2 = [ODEName[notation[0]](t, notation[1], pop, lambdat, fUplus_I, fUplus_sick) for notation in notation_list_k_split]

    dpop = dpop1[:6] + dpop2 + dpop1[-3:]

    return dpop

#############################################################################
# Simulation
#############################################################################

# Initializing the populations
pop0 = [0 for i in np.arange(len(index))]

# One initial fully infected individual in U-
pop0[index['I^U-_1']] = N_init_inf

# Initial number of susceptible
S_init = N - N_init_inf

# Time points extracted from the simulation
tspan = np.linspace(0, sim_time)

colonne = ["susceptible", "infected", "recovered_Inf", "recovered_Vac", "recovered_total", "dead", "incidence",
           "ADE_Level", "waiting_time_Bf_vaccination",
           "waiting_time_Bf_effect", "Vaccine_Start", "proportion_Unvaccinable", "Lethal", "HIT", "simulation_step"]

df = pd.DataFrame(columns=colonne)

is_HIT_and_vaccine = False  # Assessing HIT and vaccine?

for mm in range(len(pNV_vec)):
    pNV = pNV_vec[mm]

    for pp in range(len(HIT_vec)):

        HIT = HIT_vec[pp]
        # Initial number of susceptible and unvaccinated individuals (U)
        pop0[index['S^U']] = (1 - pNV) * S_init * (1 - HIT)

        # Initial number of recovered individuals (R)
        pop0[index['RVac']] = (1 - pNV) * S_init * HIT

        # Initial number of susceptible individuals who will not be vaccinated (NV)
        pop0[index['S^NV']] = pNV * S_init

        if HIT != 0:  # Assessing HIT
            if not is_HIT_and_vaccine:  # no interventions
                nu = 0
                
                fUplus_I_const = 0
                
                alpha = 0

                tiso1 = tiso1_vec[0]

                pcont_a = pcont_a_vec[0]
                pcont_b = pcont_b_vec[0]
                pcont_c = pcont_c_vec[0]
                pcont_d = pcont_d_vec[0]
                pcont_e = pcont_e_vec[0]
                pcont_f = pcont_f_vec[0]
                pcont_g = pcont_g_vec[0]
                pcont_h = pcont_h_vec[0]
                pcont_i = pcont_i_vec[0]
                pcont_j = pcont_j_vec[0]

                # Proportion of susceptible individuals who become NI, PI, ADE, and recovered R after vaccination
                fADE_S, fNI_S, fPI_S, fR_S = [0, 0, 0, 0]

                # Proportion of individuals at E stage who become NI, PI, ADE, and recovered R after vaccination
                fADE_E, fNI_E, fPI_E, fR_E = [0, 0, 0, 0]

                # Proportion of individuals at P stage who become NI, PI, ADE, and recovered R after vaccination
                fADE_P, fNI_P, fPI_P, fR_P = [0, 0, 0, 0]

                # Proportion of individuals at I stage who become NI, PI, ADE, and recovered R after vaccination
                fADE_I, fNI_I, fPI_I, fR_I = [0, 0, 0, 0]

                # Proportion of individuals at L stage who become NI, PI, ADE, and recovered R after vaccination
                fADE_L, fNI_L, fPI_L, fR_L = [0, 0, 0, 0]

                # Vaccine onset
                tVac = 0

                # Proportion of dead in ADE individuals (no ADE in this case as no vaccine intervention)
                fADE_dead = 0

                df1 = pd.DataFrame(columns=colonne)  # Empty data frame to save the simulated data

                # Solving the system of ODEs
                soln = solve_ivp(lambda t, y: f(t, y, alpha), [0, sim_time], pop0, method="RK45",
                                 dense_output=True)

                # Saving the simulation results into a data frame

                # Column: Susceptibles
                SNv = soln.y[index['S^NV']]
                SU = soln.y[index['S^U']]
                SV = soln.y[index['S^V']]
                SNI = soln.y[index['S^NI']]
                SPI = soln.y[index['S^PI']]
                SADE = soln.y[index['S^ADE']]

                long = len(SNv)

                infec_E = [0 for i in range(long)]
                infec_P = [0 for i in range(long)]
                infec_I = [0 for i in range(long)]
                infec_L = [0 for i in range(long)]
                infec = [0 for i in range(long)]

                Susc = np.add(np.add(SNv.tolist(), SU.tolist()), SV.tolist())

                Susc2 = np.add(np.add(np.add(np.add(SNv.tolist(), SU.tolist()), SNI.tolist()), SPI.tolist()), SADE.tolist())

                # Column infected
                for upperscript in ['U', 'V', 'NI', 'PI', 'ADE']:
                    for k in range(1, NE + 1):
                        infec_E = [m + n for m, n in
                                   zip(infec_E, soln.y[index['E^' + upperscript + '_' + str(k)]])]
                        infec_P = [m + n for m, n in
                                   zip(infec_P, soln.y[index['P^' + upperscript + '_' + str(k)]])]

                for upperscript in ['U+', 'U-', 'V', 'NI', 'PI', 'ADE', 'IV', 'Itilde', 'Istar']:
                    for k in range(1, NI + 1):
                        infec_I = [m + n for m, n in
                                   zip(infec_I, soln.y[index['I^' + upperscript + '_' + str(k)]])]

                for upperscript in ['U+', 'U-', 'V', 'NI', 'PI', 'ADE', 'IV', 'Itilde', 'Istar', 'LV',
                                    'Ltilde', 'Lstar']:
                    for k in range(1, NL + 1):
                        infec_L = [m + n for m, n in
                                   zip(infec_L, soln.y[index['L^' + upperscript + '_' + str(k)]])]

                infec = np.add(np.add(infec_E, infec_P), np.add(infec_I, infec_L))

                # Column incidence
                cnt = 0
                lbda = []
                for t in soln.t.tolist():
                    fUplus_I = func_fUplus_I(t, fUplus_I_const)
                    fUplus_sick = f_sick / (f_sick + (1 - f_sick) * fUplus_I)
                    lbda.append(l(t, soln.y[:, cnt], fUplus_I, fUplus_sick))
                    cnt += 1

                incid = np.multiply(lbda, Susc2)

                # Column Recovered 'RInf'
                RecovInf = soln.y[index['RInf']]

                # Column Recovered 'RVac'
                RecoVac = soln.y[index['RVac']]

                # Column Total recovery
                Recov = np.add(RecovInf.tolist(), RecoVac.tolist())

                # Column Death toll
                Dead = soln.y[index['D']]

                # Column: ADE Level
                ADE_Lev = ["None" for i in range(long)]

                # Column: Vaccination rate
                VR = ["None" for i in range(long)]

                # Column: Vaccination effectiveness rate
                VER = ["None" for i in range(long)]

                # Column: Vaccination
                Vstart = ["None" for i in range(long)]

                # Column: Proportion of unvaccinable
                if pNV != 0:
                	Vunvac = [str(pNV) for i in range(long)]
                else:
                	Vunvac = [str(0) for i in range(long)]

                # Column: Lethal
                Vlethal = ["None" for i in range(long)]

                # Column: HIT (Herd Immunity Threshold)
                VHIT = [str(HIT) for i in range(long)]

                # Column: simulation time
                sim_step = soln.t.tolist()

                # saving the simulated data
                df1 = pd.DataFrame({"susceptible": Susc, "infected": infec, "recovered_Inf": RecovInf,
                                    "recovered_Vac": RecoVac, "recovered_total": Recov,
                                    "dead": Dead, "incidence": incid, "ADE_Level": ADE_Lev,
                                    "waiting_time_Bf_vaccination": VR,
                                    "waiting_time_Bf_effect": VER, "Vaccine_Start": Vstart,
                                    "proportion_Unvaccinable": Vunvac, "Lethal": Vlethal,
                                    "HIT": VHIT, "simulation_step": sim_step})

                # saving the final dataset
                df = df.append(df1, ignore_index=True)

            else:  # Assessing effect of immunity level vs vaccination rate
                tiso1 = tiso1_vec[1]

                pcont_a = pcont_a_vec[1]
                pcont_b = pcont_b_vec[1]
                pcont_c = pcont_c_vec[1]
                pcont_d = pcont_d_vec[1]
                pcont_e = pcont_e_vec[1]
                pcont_f = pcont_f_vec[1]
                pcont_g = pcont_g_vec[1]
                pcont_h = pcont_h_vec[1]
                pcont_i = pcont_i_vec[1]
                pcont_j = pcont_j_vec[1]

                for ii in range(len(nu_vec)):
                    nu = nu_vec[ii]

                    if nu == 0:
                        fUplus_I_const = 0
                    else:
                    	fUplus_I_const = fUplus_I_val

                    for jj in range(len(alpha_vec)):
                        alpha = alpha_vec[jj]

                        for kk in range(fVac.shape[0]):

                            # Proportion of susceptible individuals who become NI, PI, ADE, and recovered R after
                            # vaccination
                            fADE_S, fNI_S, fPI_S, fR_S = list(fVac[kk])

                            # Proportion of individuals at E stage who become NI, PI, ADE, and recovered R after
                            # vaccination
                            fADE_E, fNI_E, fPI_E, fR_E = list(fVac[kk])

                            # Proportion of individuals at P stage who become NI, PI, ADE, and recovered R after
                            # vaccination
                            fADE_P, fNI_P, fPI_P, fR_P = list(fVac[kk])

                            # Proportion of individuals at I stage who become NI, PI, ADE, and recovered R after
                            # vaccination
                            fADE_I, fNI_I, fPI_I, fR_I = list(fVac[kk])

                            # Proportion of individuals at L stage who become NI, PI, ADE, and recovered R after
                            # vaccination
                            fADE_L, fNI_L, fPI_L, fR_L = list(fVac[kk])

                            df1 = pd.DataFrame(columns=colonne)  # Empty data frame to save the simulated data

                            for ll in range(len(tVaccin)):
                                tVac = tVaccin[ll]

                                for nn in range(len(fADE_dead_vec)):
                                    fADE_dead = fADE_dead_vec[nn]

                                    # Solving the system of ODEs
                                    soln = solve_ivp(lambda t, y: f(t, y, alpha), [0, sim_time], pop0, method="RK45",
                                                     dense_output=True)

                                    # Saving the simulation results into a data frame

                                    # Column: Susceptibles
                                    SNv = soln.y[index['S^NV']]
                                    SU = soln.y[index['S^U']]
                                    SV = soln.y[index['S^V']]
                                    SNI = soln.y[index['S^NI']]
                                    SPI = soln.y[index['S^PI']]
                                    SADE = soln.y[index['S^ADE']]

                                    long = len(SNv)

                                    infec_E = [0 for i in range(long)]
                                    infec_P = [0 for i in range(long)]
                                    infec_I = [0 for i in range(long)]
                                    infec_L = [0 for i in range(long)]
                                    infec = [0 for i in range(long)]

                                    Susc = np.add(np.add(SNv.tolist(), SU.tolist()), SV.tolist())

                                    Susc2 = np.add(np.add(np.add(np.add(SNv.tolist(), SU.tolist()), SNI.tolist()), SPI.tolist()), SADE.tolist())

                                    # Column infected
                                    for upperscript in ['U', 'V', 'NI', 'PI', 'ADE']:
                                        for k in range(1, NE + 1):
                                            infec_E = [m + n for m, n in
                                                       zip(infec_E, soln.y[index['E^' + upperscript + '_' + str(k)]])]
                                            infec_P = [m + n for m, n in
                                                       zip(infec_P, soln.y[index['P^' + upperscript + '_' + str(k)]])]

                                    for upperscript in ['U+', 'U-', 'V', 'NI', 'PI', 'ADE', 'IV', 'Itilde', 'Istar']:
                                        for k in range(1, NI + 1):
                                            infec_I = [m + n for m, n in
                                                       zip(infec_I, soln.y[index['I^' + upperscript + '_' + str(k)]])]

                                    for upperscript in ['U+', 'U-', 'V', 'NI', 'PI', 'ADE', 'IV', 'Itilde', 'Istar',
                                                        'LV',
                                                        'Ltilde', 'Lstar']:
                                        for k in range(1, NL + 1):
                                            infec_L = [m + n for m, n in
                                                       zip(infec_L, soln.y[index['L^' + upperscript + '_' + str(k)]])]

                                    infec = np.add(np.add(infec_E, infec_P), np.add(infec_I, infec_L))

                                    # Column incidence
                                    cnt = 0
                                    lbda = []
                                    for t in soln.t.tolist():
                                        fUplus_I = func_fUplus_I(t, fUplus_I_const)
                                        fUplus_sick = f_sick / (f_sick + (1 - f_sick) * fUplus_I)
                                        lbda.append(l(t, soln.y[:, cnt], fUplus_I, fUplus_sick))
                                        cnt += 1

                                    incid = np.multiply(lbda, Susc2)

                                    # Column Recovered 'RInf'
                                    RecovInf = soln.y[index['RInf']]

                                    # Column Recovered 'RVac'
                                    RecoVac = soln.y[index['RVac']]

                                    # Column Total recovery
                                    Recov = np.add(RecovInf.tolist(), RecoVac.tolist())

                                    # Column Death toll
                                    Dead = soln.y[index['D']]

                                    # Column: ADE Level
                                    if kk == 0:
                                        ADE_Lev = ["Good" for i in range(long)]
                                    elif kk == 1:
                                        ADE_Lev = ["Medium" for i in range(long)]
                                    elif kk == 2:
                                        ADE_Lev = ["Low" for i in range(long)]
                                    else:
                                        ADE_Lev = ["Poor" for i in range(long)]

                                    # Column: Vaccination rate
                                    if nu != 0:
                                        VR = [str(1 / nu) for i in range(long)]
                                    else:
                                        VR = [str(0) for i in range(long)]

                                    # Column: Vaccination effectiveness rate
                                    VER = [str(1 / alpha) for i in range(long)]

                                    # Column: Vaccination
                                    Vstart = [str(tVac) for i in range(long)]

                                    # Column: Proportion of unvaccinable
                                    if pNV != 0:
                                        Vunvac = [str(pNV) for i in range(long)]
                                    else:
                                        Vunvac = [str(0) for i in range(long)]

                                    # Column: Lethal
                                    Vlethal = [str(fADE_dead) for i in range(long)]

                                    # Column: HIT (Herd Immunity Treshold)
                                    VHIT = [str(HIT) for i in range(long)]

                                    # Column: simulation time
                                    sim_step = soln.t.tolist()

                                    # saving the simulated data
                                    df1 = pd.DataFrame(
                                        {"susceptible": Susc, "infected": infec, "recovered_Inf": RecovInf,
                                         "recovered_Vac": RecoVac, "recovered_total": Recov,
                                         "dead": Dead, "incidence": incid, "ADE_Level": ADE_Lev,
                                         "waiting_time_Bf_vaccination": VR,
                                         "waiting_time_Bf_effect": VER, "Vaccine_Start": Vstart,
                                         "proportion_Unvaccinable": Vunvac, "Lethal": Vlethal,
                                         "HIT": VHIT, "simulation_step": sim_step})

                                    # saving the final dataset
                                    df = df.append(df1, ignore_index=True)

        else:  # Not assessing HIT effect

            tiso1 = tiso1_vec[1]

            pcont_a = pcont_a_vec[1]
            pcont_b = pcont_b_vec[1]
            pcont_c = pcont_c_vec[1]
            pcont_d = pcont_d_vec[1]
            pcont_e = pcont_e_vec[1]
            pcont_f = pcont_f_vec[1]
            pcont_g = pcont_g_vec[1]
            pcont_h = pcont_h_vec[1]
            pcont_i = pcont_i_vec[1]
            pcont_j = pcont_j_vec[1]

            for ii in range(len(nu_vec)):
                nu = nu_vec[ii]
                if nu == 0:
                    fUplus_I_const = 0
                else:
                    fUplus_I_const = fUplus_I_val

                for jj in range(len(alpha_vec)):
                    alpha = alpha_vec[jj]

                    for kk in range(fVac.shape[0]):

                        # Proportion of susceptible individuals who become NI, PI, ADE, and recovered R after
                        # vaccination
                        fADE_S, fNI_S, fPI_S, fR_S = list(fVac[kk])

                        # Proportion of individuals at E stage who become NI, PI, ADE, and recovered R after
                        # vaccination
                        fADE_E, fNI_E, fPI_E, fR_E = list(fVac[kk])

                        # Proportion of individuals at P stage who become NI, PI, ADE, and recovered R after
                        # vaccination
                        fADE_P, fNI_P, fPI_P, fR_P = list(fVac[kk])

                        # Proportion of individuals at I stage who become NI, PI, ADE, and recovered R after
                        # vaccination
                        fADE_I, fNI_I, fPI_I, fR_I = list(fVac[kk])

                        # Proportion of individuals at L stage who become NI, PI, ADE, and recovered R after
                        # vaccination
                        fADE_L, fNI_L, fPI_L, fR_L = list(fVac[kk])

                        df1 = pd.DataFrame(columns=colonne)  # Empty data frame to save the simulated data

                        for ll in range(len(tVaccin)):
                            tVac = tVaccin[ll]

                            for nn in range(len(fADE_dead_vec)):
                                fADE_dead = fADE_dead_vec[nn]

                                # Solving the system of ODEs
                                soln = solve_ivp(lambda t, y: f(t, y, alpha), [0, sim_time], pop0, method="RK45",
                                                 dense_output=True)

                                # Saving the simulation results into a data frame

                                # Column: Susceptibles
                                SNv = soln.y[index['S^NV']]
                                SU = soln.y[index['S^U']]
                                SV = soln.y[index['S^V']]
                                SNI = soln.y[index['S^NI']]
                                SPI = soln.y[index['S^PI']]
                                SADE = soln.y[index['S^ADE']]

                                long = len(SNv)

                                infec_E = [0 for i in range(long)]
                                infec_P = [0 for i in range(long)]
                                infec_I = [0 for i in range(long)]
                                infec_L = [0 for i in range(long)]
                                infec = [0 for i in range(long)]

                                Susc = np.add(np.add(SNv.tolist(), SU.tolist()), SV.tolist())

                                Susc2 = np.add(np.add(np.add(np.add(SNv.tolist(), SU.tolist()), SNI.tolist()), SPI.tolist()), SADE.tolist())

                                # Column infected
                                for upperscript in ['U', 'V', 'NI', 'PI', 'ADE']:
                                    for k in range(1, NE + 1):
                                        infec_E = [m + n for m, n in
                                                   zip(infec_E, soln.y[index['E^' + upperscript + '_' + str(k)]])]
                                        infec_P = [m + n for m, n in
                                                   zip(infec_P, soln.y[index['P^' + upperscript + '_' + str(k)]])]

                                for upperscript in ['U+', 'U-', 'V', 'NI', 'PI', 'ADE', 'IV', 'Itilde', 'Istar']:
                                    for k in range(1, NI + 1):
                                        infec_I = [m + n for m, n in
                                                   zip(infec_I, soln.y[index['I^' + upperscript + '_' + str(k)]])]

                                for upperscript in ['U+', 'U-', 'V', 'NI', 'PI', 'ADE', 'IV', 'Itilde', 'Istar', 'LV',
                                                    'Ltilde', 'Lstar']:
                                    for k in range(1, NL + 1):
                                        infec_L = [m + n for m, n in
                                                   zip(infec_L, soln.y[index['L^' + upperscript + '_' + str(k)]])]

                                infec = np.add(np.add(infec_E, infec_P), np.add(infec_I, infec_L))

                                # Column incidence
                                cnt = 0
                                lbda = []
                                for t in soln.t.tolist():
                                    fUplus_I = func_fUplus_I(t, fUplus_I_const)
                                    fUplus_sick = f_sick / (f_sick + (1 - f_sick) * fUplus_I)
                                    lbda.append(l(t, soln.y[:, cnt], fUplus_I, fUplus_sick))
                                    cnt += 1

                                incid = np.multiply(lbda, Susc2)

                                # Column Recovered 'RInf'
                                RecovInf = soln.y[index['RInf']]

                                # Column Recovered 'RVac'
                                RecoVac = soln.y[index['RVac']]

                                # Column Total recovery
                                Recov = np.add(RecovInf.tolist(), RecoVac.tolist())

                                # Column Death toll
                                Dead = soln.y[index['D']]

                                # Column: ADE Level
                                if kk == 0:
                                    ADE_Lev = ["Good" for i in range(long)]
                                elif kk == 1:
                                    ADE_Lev = ["Medium" for i in range(long)]
                                elif kk == 2:
                                    ADE_Lev = ["Low" for i in range(long)]
                                else:
                                    ADE_Lev = ["Poor" for i in range(long)]

                                # Column: Vaccination rate
                                if nu != 0:
                                    VR = [str(1 / nu) for i in range(long)]
                                else:
                                    VR = [str(0) for i in range(long)]

                                # Column: Vaccination effectiveness rate
                                VER = [str(1 / alpha) for i in range(long)]

                                # Column: Vaccination
                                Vstart = [str(tVac) for i in range(long)]

                                # Column: Proportion of unvaccinable
                                if pNV != 0:
                                    Vunvac = [str(pNV) for i in range(long)]
                                else:
                                    Vunvac = [str(0) for i in range(long)]

                                # Column: Lethal
                                Vlethal = [str(fADE_dead) for i in range(long)]

                                # Column: HIT (Herd Immunity Treshold)
                                VHIT = [str(HIT) for i in range(long)]

                                # Column: simulation time
                                sim_step = soln.t.tolist()

                                # saving the simulated data
                                df1 = pd.DataFrame({"susceptible": Susc, "infected": infec, "recovered_Inf": RecovInf,
                                                    "recovered_Vac": RecoVac, "recovered_total": Recov,
                                                    "dead": Dead, "incidence": incid, "ADE_Level": ADE_Lev,
                                                    "waiting_time_Bf_vaccination": VR,
                                                    "waiting_time_Bf_effect": VER, "Vaccine_Start": Vstart,
                                                    "proportion_Unvaccinable": Vunvac, "Lethal": Vlethal,
                                                    "HIT": VHIT, "simulation_step": sim_step})

                                # saving the final dataset
                                df = df.append(df1, ignore_index=True)

# Saving the data from the simulation
df.to_csv("COVID_19_ADE_simulation_USA.csv")
