
from scipy.integrate import solve_ivp
import numpy as np
from matplotlib import pyplot as plt
from math import exp, log

# Given Values:
F = 96485e3    # Faraday's constant, C/kmol equivalent charge
R = 8314.5     # Gas constant, J/kmol-K

k_fwd_star = 4.16307062e+7 # Chemical forward rate constant, m^4/kmol^2/s

C_elyte = 46.05    # Total (reference) elyte concentration, kmol/m3
C_Ni_s = 2.6e-08   # Total (reference) concentration of Ni surface sites, kmol/m2

X_H_Ni = 0.6
X_H2O_Ni = 0.2
X_Vac_Ni = 0.2
X_Vac_elyte = 0.08
X_Ox_elyte = 0.92

"Species standard-state thermo"
g_H_Ni_o = -7.109209e+07      # standard-state gibbs energy for H adsorbed on Ni surface (J/kmol)
g_H2O_Ni_o = -3.97403035e+08  # standard-state gibbs energy for H2O adsorbed on Ni surface (J/kmol)
g_Vac_Ni_o = 0.0              # standard-state gibbs energy for Ni surface vacancy (J/kmol)
g_Vac_elyte_o = 0.0           # standard-state gibbs energy for electrolyte oxide vacancy (J/kmol)
g_Ox_elyte_o = -2.1392135e+08 # standard-state gibbs energy for electrolyte oxide O2- (J/kmol)


Beta = 0.5
n = -2
delta_Phi_eq_an = 0.61
DeltaG_rxn_circ = g_Vac_Ni_o + g_H2O_Ni_o + g_Vac_elyte_o - 2*g_H_Ni_o - g_Ox_elyte_o
Prod_C_ac = (X_Vac_elyte*X_H2O_Ni*X_Vac_Ni)/(X_Ox_elyte*X_H_Ni**2)
phi_an = 0
phi_elyte_0 = 0.6
temps =np.array([600, 800, 1200])
i_Far_an = np.zeros_like(temps)
                                  
i_ext=20
                                       
SV_0 = [phi_elyte_0 - phi_an]
#SV_0 = np.array([phi_elyte_0 - phi_an, phi_ca_0 - phi_elyte_0])
time_span = np.array([0,1000])
C_dl_an = 1e4 # F/m2


def residual(t,SV):
    dSV_dt = np.zeros_like(SV)
    delta_phi = SV[0] - delta_Phi_eq_an
    k_fwd = k_fwd_star * exp(-Beta * n * F * delta_phi / (R * T)) #we need to recalc delta phi
    k_rev = k_rev_star * exp((1-Beta) * n * F * delta_phi / (R * T))
    i_Far_an = n * F * (k_fwd * (X_Ox_elyte*C_elyte*(X_H_Ni*C_Ni_s)**2)
                               - k_rev * (X_Vac_elyte*C_elyte*X_H2O_Ni*C_Ni_s*X_Vac_Ni*C_Ni_s))
    
    i_dl_an = i_ext - i_Far_an
    dSV_dt[0] = -i_dl_an/C_dl_an
    
    # print(i_dl_an)
    # print(dSV_dt)
    return dSV_dt,


# ### Case 1: vary temp

for T in temps:
    DeltaG_rxn = DeltaG_rxn_circ + R*T*log(Prod_C_ac)
    k_rev_star = k_fwd_star / exp(-DeltaG_rxn/(R*T)) / Prod_C_ac
    solution = solve_ivp(residual,time_span,SV_0,rtol=1e-4, atol=1e-6)
    # print(i_Far_an)

    f1=solution.t
    f2=np.transpose(solution.y)
    plt.plot(f1,f2)
plt.xlabel('Time (s)',fontsize=14)
plt.ylabel('Electric Potential (V)',fontsize=14)
plt.legend(['T=600 K','T=800 K','T=1200 K'])
plt.title('Cathode Double Layer Electric Potential at Various SOFC Operating Temperatures')
plt.show()

## Case 2: Vary external current

T=800
i_ext_array=np.array([5, 10, 15])
for i_ext in i_ext_array:
    solution = solve_ivp(residual,time_span,SV_0,rtol=1e-4, atol=1e-6)

    f1=solution.t
    f2=np.transpose(solution.y)
    plt.plot(f1,f2)
plt.xlabel('Time (s)',fontsize=14)
plt.ylabel('Electric Potential (V)',fontsize=14)
plt.legend(['i_ext=5A','i_ext=10A','i_ext=15A'])
plt.title('Cathode Double Layer Electric Potential at Various External Current Demands')

plt.show()
