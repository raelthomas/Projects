from scipy.integrate import solve_ivp
import numpy as np
from matplotlib import pyplot as plt
from math import exp, log
import itertools

# Given Values:
F = 96485e3  # Faraday's constant, s-A/kmol equivalent charge
R = 8314.5  # Gas constant, J/kmol-K

C_dl_an = 0.003 # F/m3

"Assumed molar fractions"
X_H_Ni = 0.9
X_Vac_Ni = 0.1
X_OH_YSZ = 0.4
X_O2ion_YSZ = 0.4
X_Vac_YSZ = 0.1
X_H2O_YSZ = 0.1


"Species standard-state thermo T 800C"
"Anode"
h_H_Ni_o = -9.1e6  # standard-state gibbs energy for H adsorbed on Ni surface (J/kmol)
s_H_Ni_o = 80.2e3  # standard-state gibbs energy for H adsorbed on Ni surface (J/kmol-K)
h_O2ion_YSZ_o = -122.3e6  # standard-state gibbs energy for O2- ion adsorbed on YSZ surface (J/kmol)
s_O2ion_YSZ_o = 128.9e3  # standard-state gibbs energy for O2- ion adsorbed on YSZ surface (J/kmol-K)
h_Vac_Ni_o = 20.0e6  # standard-state gibbs energy for a vacancy on Ni surface (J/kmol)
s_Vac_Ni_o = 33.1e3  # standard-state gibbs energy for a vacancy on Ni surface (J/kmol-K)
h_OH_YSZ_o = 47.8e6  # standard-state gibbs energy for OH adsorbed on YSZ surface (J/kmol)
s_OH_YSZ_o = 165.9e3  # standard-state gibbs energy for OH adsorbed on YSZ surface (J/kmol-K)
h_H2O_YSZ_o = -293.4e6  # standard-state gibbs energy for H2O adsorbed on YSZ surface (J/kmol)
s_H2O_YSZ_o = 177.0e3  # standard-state gibbs energy for H2) adsorbed on YSZ surface (J/kmol-K)

E_act_fwd_an_1 = 90.0e6  # Activation energy (J/kmol)
E_act_fwd_an_2 = 90.0e6  # Activation energy (J/kmol)
A_fwd_an_1 = 1e11  # (kmol-m-s)
A_fwd_an_2 = 1e10  # (kmol-m-s)
alpha_fwd_an_1 = 0.5
alpha_fwd_an_2 = 0.5

gamma_surf_an = 1e-9
Prod_C_ac_an_1 = (X_Vac_Ni * X_OH_YSZ) / (X_H_Ni * X_O2ion_YSZ)
Prod_C_ac_an_2 = (X_H2O_YSZ * X_Vac_YSZ) / (X_H_Ni * X_OH_YSZ)

temps = np.array([800 + 273.15, 900 + 273.15, 1000 + 273.15]) #[K]
i_Far_an = np.zeros_like(temps) #[A/m^2]

i_ext = 1000
SV_0 = np.array([0.8])
time_span = np.linspace(0, 0.00025, num=100)
i_ext_array = np.linspace(0, 3000, num=100)

n_an = -1
dSV_dt = np.zeros_like(SV_0)

temps_range = np.linspace(800+273, 1000+273, num=100)
G_an_1_app = []
G_an_2_app = []
G_an_tot_app = []
temps_range_plot = temps_range-273

for T in temps_range:
    g_H_Ni_o = h_H_Ni_o - T * s_H_Ni_o                          # [J/kmol]
    g_O2ion_YSZ_o = h_O2ion_YSZ_o - T * s_O2ion_YSZ_o
    g_Vac_Ni_o = h_Vac_Ni_o - T * s_Vac_Ni_o
    g_OH_YSZ_o = h_OH_YSZ_o - T * s_OH_YSZ_o
    g_H2O_YSZ_o = h_H2O_YSZ_o - T * s_H2O_YSZ_o

    Delta_g_circ_an_1 = g_Vac_Ni_o + g_OH_YSZ_o - (g_H_Ni_o + g_O2ion_YSZ_o)     # [J/kmol]
    Delta_g_circ_an_2 = g_H2O_YSZ_o + g_Vac_Ni_o - (g_H_Ni_o + g_OH_YSZ_o)

    k_star_fwd_an_1 = A_fwd_an_1 * exp(-E_act_fwd_an_1 / (R * T))  # DeCaluwe eq 13, [kmol-m-s]
    k_star_fwd_an_2 = A_fwd_an_2 * exp(-E_act_fwd_an_2 / (R * T))

    Delta_g_an_1 = Delta_g_circ_an_1 + R * T * log(Prod_C_ac_an_1)  # Fuller eq 2.10 [J/kmol]
    Delta_g_an_2 = Delta_g_circ_an_2 + R * T * log(Prod_C_ac_an_2)

    delta_g_an_tot = Delta_g_an_1 + Delta_g_an_2

    f1 = Delta_g_an_1/1e6
    f2 = Delta_g_an_2/1e6
    f3 = delta_g_an_tot/1e6
    G_an_1_app.append(f1)
    G_an_2_app.append(f2)
    G_an_tot_app.append(f3)

plt.plot(temps_range_plot, G_an_1_app)
plt.plot(temps_range_plot, G_an_2_app)
plt.plot(temps_range_plot, G_an_tot_app)
plt.legend(['Eq (1)', 'Eq (2)', 'Summation'])
plt.xlabel('Temperature (C)', fontsize=10)
plt.ylabel('Gibbs Free Energy (kJ/mol)', fontsize=10)
plt.title('Temperature Dependence of Gibbs Free Energy')
plt.show()

plt.plot(temps_range_plot, G_an_tot_app)
plt.xlabel('Temperature (C)', fontsize=10)
plt.ylabel('Gibbs Free Energy (kJ/mol)', fontsize=10)
plt.legend(['Summation'])
plt.title('Temperature Dependence of Gibbs Free Energy')
plt.show()

def residual(t, SV):
    delta_phi_an = SV[0]

    "temperature dependent"
    g_H_Ni_o = h_H_Ni_o - T * s_H_Ni_o                          # [J/kmol]
    g_O2ion_YSZ_o = h_O2ion_YSZ_o - T * s_O2ion_YSZ_o
    g_Vac_Ni_o = h_Vac_Ni_o - T * s_Vac_Ni_o
    g_OH_YSZ_o = h_OH_YSZ_o - T * s_OH_YSZ_o
    g_H2O_YSZ_o = h_H2O_YSZ_o - T * s_H2O_YSZ_o
    Delta_g_circ_an_1 = g_Vac_Ni_o + g_OH_YSZ_o - (g_H_Ni_o + g_O2ion_YSZ_o)     # [J/kmol]
    Delta_g_circ_an_2 = g_H2O_YSZ_o + g_Vac_Ni_o - (g_H_Ni_o + g_OH_YSZ_o)
    k_star_fwd_an_1 = A_fwd_an_1 * exp(-E_act_fwd_an_1 / (R * T))  # DeCaluwe eq 13, [kmol-m-s]
    k_star_fwd_an_2 = A_fwd_an_2 * exp(-E_act_fwd_an_2 / (R * T))

    Delta_g_an_1 = Delta_g_circ_an_1 + R * T * log(Prod_C_ac_an_1)  # Fuller eq 2.10 [J/kmol]
    Delta_g_an_2 = Delta_g_circ_an_2 + R * T * log(Prod_C_ac_an_2)

    "temperature and delta_phi dependent"
    k_fwd_an_1 = k_star_fwd_an_1 * exp(-alpha_fwd_an_1 * n_an * F * delta_phi_an / (R * T))  # DeCaluwe eq 15 [kmol-m-s]
    k_fwd_an_2 = k_star_fwd_an_2 * exp(-alpha_fwd_an_2 * n_an * F * delta_phi_an / (R * T))

    k_rev_an_1 = k_fwd_an_1 * exp(Delta_g_an_1 / (R * T)) * exp(n_an * F * delta_phi_an / (R * T)) #[kmol-m-s]
    k_rev_an_2 = k_fwd_an_2 * exp(Delta_g_an_2 / (R * T)) * exp(n_an * F * delta_phi_an / (R * T))
    q_dot_rop_an_1 = (k_fwd_an_1 * (X_H_Ni * X_O2ion_YSZ) - k_rev_an_1 * (X_Vac_Ni * X_OH_YSZ)) * gamma_surf_an ** 2  # DeCaluwe eq 17
    q_dot_rop_an_2 = (k_fwd_an_2 * (X_H_Ni * X_OH_YSZ) - k_rev_an_2 * X_H2O_YSZ * X_Vac_YSZ) * gamma_surf_an ** 2 #
    i_Far_an = F * (n_an * q_dot_rop_an_1 + n_an * q_dot_rop_an_2)
    i_dl_an = i_ext - i_Far_an
    dSV_dt[0] = -i_dl_an / C_dl_an

    return dSV_dt


### Case 1: vary temp

for T in temps:
    solution = solve_ivp(residual, time_span, SV_0, rtol=1e-10, atol=1e-10, method='BDF')
    SV_0 = solution.y[:, -1]
    f1 = solution.t
    f2 = np.transpose(solution.y[0])
    plt.plot(f1*100000, f2)
plt.xlabel('Time (ms)', fontsize=10)
plt.ylabel('Electric Potential (V)', fontsize=10)
plt.legend(['Anode, T=800 C', 'Anode, T=900 C', 'Anode, T=1000 C', ])
plt.title('Anode Double Layer Electric Potential')
plt.show()

## Case 2: Vary external current
f2_app = []
SV_0 = np.array([1.75])
for i_v in i_ext_array:
    T = 1073
    i_ext = i_v
    solution = solve_ivp(residual, time_span, SV_0, rtol=1e-9, atol=1e-4, method='BDF')
    f1 = i_ext_array
    f2 = solution.y[0, -1]
    f2_app.append(f2)
    f2_vec = np.stack(f2_app, axis=0)
plt.plot(f1, f2_vec)
plt.xlabel('External Current (A/m^2)', fontsize=10)
plt.ylabel('Electric Potential (V)', fontsize=10)
plt.title('Polarization Curve')
plt.show()  # change for current on bottom
