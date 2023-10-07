from numpy import array, load, log10
from scipy.interpolate import interpn
from Model import Jent_numerical_New

##-- Phyiscal Constants with units --##
nuc = 1e-6 # Continuous phase viscosity
g = 9.81; rhoc = 1000; sig = 0.072
wmeth = 2
zlam_min = 2 ## zlam: z/lambda
zlam_max = 3
Fr2_crt_PolyExtra = True
F_tab_NP = True
sector = 0
J_table=load("./data/J_star_table.npz")
zeroDTable=load("./data/Ent_table_org.npz")
n_Wee = J_table["n_Wee"]; Wee_lst = J_table["Wee_lst"]
n_Fr2 = J_table["n_Fr2"]; Fr2_lst = J_table["Fr2_lst"]
n_Eta = J_table["n_Eta"]; Eta_lst = J_table["Eta_lst"]
J_fin_table = J_table["J_fin_table"]

logWee_lst	=log10(Wee_lst)
logFr2_lst	=log10(Fr2_lst)
logEta_lst	=log10(Eta_lst)
logJ_tab	=log10(J_fin_table)

coords=(logWee_lst,logFr2_lst,logEta_lst)

## User input
kt = 0.7E+00
et = 2.71E+00 

L =kt**1.5/et
Eta = nuc**(0.75)/et**(0.25)/L
Wee = rhoc*et**(2.0/3.0)*L**(5.0/3.0)/sig
Fr2 = et**(2.0/3.0)*L**(-1.0/3.0)/g

print('----------------------------------------------------------------------------------')
print(' >>Inputs  kt:{:.3e} m^2/s^2, et:{:.3e} m^2/s^3 ==> L:{:.3e} m, Eta:{:.3e} m'.format(kt,et,L, Eta*L))
print(' >>Inputs  Wee:{:.3e}, Fr2:{:.3e}, Eta*:{:.3e}'.format(Wee,Fr2,Eta))
logWee  = log10(Wee);
logFr2  = log10(Fr2);
logEta  = log10(Eta);

J_dim=Jent_numerical_New(kt,et,nuc,g,rhoc,sig,zeroDTable,zlam_min,zlam_max,wmeth,Fr2_crt_PolyExtra,F_tab_NP,sector)
point = array([logWee, logFr2, logEta])
J_scl_int=10**(interpn(coords, logJ_tab, point))[0] * (et*L)**(1/3) 
print(' >>J: Exact:{:.5e}m/s, Interp:{:.5e}m/s, Rel err:{:6.2f}%'.format(J_dim,J_scl_int, (J_scl_int-J_dim)/
J_dim*100))