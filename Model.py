from numpy import sqrt, exp, log, pi, logspace, zeros, log10, interp, linspace, mod
from scipy.special import erfc
from scipy.integrate import quad, quadrature
from Utilities import findcLceta, ulambda_sq
from scipy.interpolate import interp2d
from numba import jit
#==============================================#
#                   New Model
#==============================================#
@jit(nopython=True, cache=True, nogil=True)
def ulam_nlam(logl,kt,et,nu,cL,cEta,lst,ul2_lst,output):
	l=exp(logl)
	ulamsq=interp(l,lst,ul2_lst)
	C=1.5;	p0=2.0;   be=5.2; k=2*pi/l
	L=kt**1.5/et; eta=(nu**3/et)**0.25
	fl  = (k*L/((k*L)**2+cL)**0.5)**(5.0/3.0+p0)
	feta= exp(-be*(((k*eta)**4+cEta**4)**0.25-cEta))
	Ek  = C*et**(2.0/3.0)*k**(-5.0/3.0)*fl*feta
	n_lam=24*Ek/(l**5*ulamsq)
	if output == 1: #numerator
		return sqrt(ulamsq)*n_lam * l
	elif output == 2: #denominator
		return n_lam * l
@jit(nopython=True, cache=True, nogil=True)
def ulam_nlam_nolog(l,kt,et,nu,cL,cEta,lst,ul2_lst,output):
	ulamsq=interp(l,lst,ul2_lst)
	C=1.5;	p0=2.0;   be=5.2; k=2*pi/l
	L=kt**1.5/et; eta=(nu**3/et)**0.25
	fl  = (k*L/((k*L)**2+cL)**0.5)**(5.0/3.0+p0)
	feta= exp(-be*(((k*eta)**4+cEta**4)**0.25-cEta))
	Ek  = C*et**(2.0/3.0)*k**(-5.0/3.0)*fl*feta
	n_lam=24*Ek/(l**5*ulamsq)
	if output == 1: #numerator
		return sqrt(ulamsq)*n_lam
	elif output == 2: #denominator
		return n_lam
####################################################################
@jit(nopython=True, cache=True, nogil=True)
def Ek_logint(logk,c1,c2,C,L,p0,beta,eta,et):
	k=exp(logk)
	return C*et**(2/3)*k**(-5.0/3.0)*(k*L/((k*L)**2+c1)**0.5)**(5.0/3.0+p0)*exp(-beta*(((k*eta)**4+c2**4)**0.25-c2))*k
@jit(nopython=True, cache=True, nogil=True)
def Ek_int(k,c1,c2,C,L,p0,beta,eta,et):
	return C*et**(2/3)*k**(-5.0/3.0)*(k*L/((k*L)**2+c1)**0.5)**(5.0/3.0+p0)*exp(-beta*(((k*eta)**4+c2**4)**0.25-c2))
####################################################################
# def get_rise_speed(l1,l2,kt,et,nu,cL,cEta,lst,ul2_lst,method):
def get_rise_speed(l1,l2,kt,et,nu,cL,cEta,method):
	if l1 > l2:
		# print('==WARNING: Length scale l1 should be smaller than l2!!!===')
		return 0
	if method == 1 : # Use second-order longitudinal structure function
		if l1 == l2:
			return 0
			# numerator   = quad(ulam_nlam, log(l1), log(l2), args=(kt,et,nu,cL,cEta,lst,ul2_lst,1),limit = 100)[0]
			# denominator = quad(ulam_nlam, log(l1), log(l2), args=(kt,et,nu,cL,cEta,lst,ul2_lst,2),limit = 100)[0]
			# numerator   = quadrature(ulam_nlam_nolog, l1, l2, args=(kt,et,nu,cL,cEta,lst,ul2_lst,1),vec_func=False,maxiter=100)[0]
			# denominator = quadrature(ulam_nlam_nolog, l1, l2, args=(kt,et,nu,cL,cEta,lst,ul2_lst,2),vec_func=False,maxiter=100)[0]
			# return numerator/denominator
		return -9999
	elif method == 2: # Use Energy spectrum
		L=kt**1.5/et;  eta=(nu**3/et)**0.25
		C=1.5;	p0=2.0;	beta=5.2

		# p1 = quadrature(Ek, 2*pi/l2, 2*pi/l1, args=(cL,cEta,C,L,p0,beta,eta),vec_func=False,maxiter=100)[0]
		# p1 = quad(Ek, 2*pi/l2, 2*pi/l1, args=(cL,cEta,C,L,p0,beta,eta),limit = 100)[0]
		p1 = quad(Ek_logint, log(2*pi/l2), log(2*pi/l1), args=(cL,cEta,C,L,p0,beta,eta,et),limit = 100,epsrel = 1e-5)[0]
		p1 = (p1 *2.0/3.0)**(0.5)
		return p1
####################################################################
#           Single depth version (z_c=3\lambda)
####################################################################
@jit(nopython=True, cache=True, nogil=True)
def Ent_Volume(zp,l,lst,ul2_lst,Reg,Bog,Weg,kt,et,cL,cEta,nu,g,circ_p,Vmax):
	if Reg<70:
		B=0
	elif Reg<2580:
		B=-3.4e-3+5.1e-5*Reg+(-9.8e-9)*Reg**2
	else:
		B=0.062
	if Bog<1:
		W=0
	elif Bog<50:
		W=1.5e-2+8.1e-5*Weg
	else:
		W=1
	# Rise velocity
	# ulam_z=sqrt(ulambda_sq(zp-l/2,kt,et,cL,cEta,nu,pope_spec=1.01)); wz=ulam_z/sqrt(2)
	ulam_z=sqrt(interp(zp-l/2,lst,ul2_lst)); wz=ulam_z/sqrt(2)
	Fr2=circ_p*wz/(l**2/4*g); fz=2.5*l/(zp-0.5*l)
	if Fr2*fz<0.4:
		F=0
	else:
		F=-0.43+1.2*Fr2*fz
	V_Ent=pi*l**3/4.0*F*B*W
	V_Ent=min(Vmax,V_Ent)
	return V_Ent
####################################################################
#           Multiple depth version (z_c=3\lambda)
####################################################################
@jit(nopython=True, cache=True, nogil=True)
def OneD_interp_weight(x,x_lst,isNP):
	if x_lst[1] < x_lst[0]:
		print('### Error: Input arry not mono increasing! ####')
		return (0, 0)
	ix=-1; d_lst = x_lst.size-1
	###### Inter/extrapolate "zoa" dimension ######
	if not(isNP): # Using linear extrapolation
		if x >= x_lst[d_lst-1]:
			ix=d_lst-1; x_lw=(x_lst[d_lst]-x)/(x_lst[d_lst]-x_lst[d_lst-1])
		elif x <= x_lst[0]:
			ix=0; 		x_lw=(x_lst[1] -x)/(x_lst[1] -x_lst[0])
	else: #  Performing N-P extrapolation (cap) for the table (takes in the original table)
		if x >= x_lst[d_lst-1]:
			ix=d_lst-1;	x_lw=0.0 
		elif x <= x_lst[0]:
			ix=0;		x_lw=1.0 
	if ix == -1: # within the original table range
		for i in range(d_lst):
			if x>=x_lst[i] and x<x_lst[i+1]:
				ix=i; x_lw=(x_lst[i+1]-x)/(x_lst[i+1]-x_lst[i])
				break
	if ix == -1:
		print('### Error: Extrapolation/interpolation not working properly! ####')
		return (0, 0)
	return (ix, x_lw)
@jit(nopython=True, cache=True, nogil=True)
def Fr_contribution(zcoa,Fr2,F_tab_NP,zcoa_lst,F_tab,Fr2_lst):
	(izcoa, zcoa_lw) = OneD_interp_weight(zcoa, zcoa_lst, F_tab_NP)
	F_lst=F_tab[izcoa,:]*zcoa_lw+F_tab[izcoa+1,:]*(1-zcoa_lw)
	###### Inter/extrapolate "Fr2" dimension ######
	(jfr, fr2_lw) = OneD_interp_weight(Fr2, Fr2_lst, F_tab_NP)
	F=max(0.0,F_lst[jfr]*fr2_lw+F_lst[jfr+1]*(1-fr2_lw))
	return F
@jit(nopython=True, cache=True, nogil=True)
def Re_contribution(zcoa,Refitcoefs,Reg):
	if zcoa < -13.502406389481719: # 1st root of the parabolic function
		return 0
	else:
		b=Refitcoefs[1]+Refitcoefs[4]*zcoa
		a=Refitcoefs[3]
		Re_crt = -b/2/a
		Reg_eff = min(Reg, Re_crt)
		B=Refitcoefs[0] + Refitcoefs[1]*Reg_eff + Refitcoefs[2]*zcoa + \
		Refitcoefs[3]*Reg_eff**2 + Refitcoefs[4]*Reg_eff*zcoa
		B=max(0.0,B)
		return B
@jit(nopython=True, cache=True, nogil=True)
def We_contribution(Bog,Weg):
	if Bog<1:
		W=0
	elif Bog<50:
		W=1.5e-2+8.1e-5*Weg
	else:
		W=1
	return W
#####################################################################
# multi value for "sector", deprecated
@jit(nopython=True, cache=True, nogil=True)
def para_sec(v, vmin, vmax):
	if v < vmin:
		return -1
	elif v >= vmin and v <= vmax:
		return 0
	elif v > vmax:
		return 1
@jit(nopython=True, cache=True, nogil=True)
def F_filter(zcoa,Fr2,sector):
	flg = False
	zcoa_sec = para_sec(zcoa, -6, -4)
	Fr2_sec = para_sec(Fr2, 0, 4)
	
	for i in range(sector.size):
		if (sector[i] == 1 and zcoa_sec == -1 and Fr2_sec == 0):
			flg = True
		if (sector[i] == 2 and zcoa_sec ==  0 and Fr2_sec == 0):
			flg = True
		if (sector[i] == 3 and zcoa_sec ==  1 and Fr2_sec == 0):
			flg = True
		if (sector[i] == 4 and zcoa_sec == -1 and Fr2_sec == 1):
			flg = True
		if (sector[i] == 5 and zcoa_sec ==  0 and Fr2_sec == 1):
			flg = True
		if (sector[i] == 6 and zcoa_sec ==  1 and Fr2_sec == 1):
			flg = True
		if (flg):
			return flg
	return flg
@jit(nopython=True, cache=True, nogil=True)
def Ent_Volume_intgrand_jit(logzp,l,zp_lst,wz_lst,g,circ_p,Reg,Bog,Weg,Refitcoefs,Fr2_lst,zcoa_lst,F_tab,F_tab_NP,sector):
	zp=exp(logzp)
	zcoa=-1*zp/(l/2)
	# ===== Reynolds number dependence =====
	B = Re_contribution(zcoa,Refitcoefs,Reg)
	# ===== Weber number dependence =====
	W = We_contribution(Bog,Weg)
	# ===== Froude number dependence =====
	# New Method
	wz=interp(zp,zp_lst,wz_lst)
	# wz = get_rise_speed(l,2*zp,kt,et,nu,cL,cEta,method=2)
	Fr2=circ_p*wz/(l**2/4*g)
	if sector >= 1:
		if sector >= 4 and Fr2 < 4:
			return 0
		elif sector <= 3 and Fr2 > 4:
			return 0
	F = Fr_contribution(zcoa,Fr2,F_tab_NP,zcoa_lst,F_tab,Fr2_lst)
	V_Ent=pi*l**3/6.0*F*B*W
	if V_Ent < 0:
		print('Error: Entrainment volume less than 0')
	return V_Ent*zp
####################################################################
@jit(nopython=True, cache=True, nogil=True)
def J_lambda_prep(l,lst,ul2_lst,kt,et,cL,cEta,nu,g,rhoc,sig):
	#---- Eddy velocity ----#
	ulamsq=interp(l,lst,ul2_lst)
	# ulamsq=ulambda_sq(l,kt,et,cL,cEta,nu,pope_spec=1.01)
	ulam  =sqrt(ulamsq)
	#---- Circulation (parallel component) ----#
	circ_p=pi*pi/4*l*ulam
	#---- Eddy lifetime
	tau_vort=l**(2.0/3.0)/et**(1.0/3.0)
	# tau_vort=ulamsq*0.5/et
	#---- MIT model input ----#
	Reg=circ_p/nu; 				Weg=circ_p**2*rhoc/(0.5*l*sig)
	Bog=g*(l/2)**2/(sig/rhoc)
	#---- Eddy number density ----#
	C=1.5;	p0=2.0;   be=5.2; k=2*pi/l
	L=kt**1.5/et; eta=(nu**3/et)**0.25
	fl  = (k*L/((k*L)**2+cL)**0.5)**(5.0/3.0+p0)
	feta= exp(-be*(((k*eta)**4+cEta**4)**0.25-cEta))
	Ek  = C*et**(2.0/3.0)*k**(-5.0/3.0)*fl*feta
	n_lam=24*Ek/(l**5*ulamsq)
	#---- Breakage probability ----#
	We = rhoc*l*ulamsq/sig #Weber number
	x = sqrt(2/We)
	return Reg,Bog,Weg,circ_p,n_lam,x,tau_vort
####################################################################
# Constants #
# See Fr2_crit_lst_getter_helper for details
zcoa_l  = 1.6057930839841816;	zcoa_r  = -1.6057930839841816;
base_l  = 0.19000483988061692;	base_r  = 0.35277970476728265;
slope_l = 0.20371010807162385;	slope_r = -0.06749286124910733;
@jit(nopython=True, cache=True, nogil=True)
def Fr2_crit_getter(l,zp,FrXcoefs,Fr2_crt_PolyExtra):
	zcoa=-1*zp/(l/2)
	zcoa_scl=(zcoa-FrXcoefs[7])/FrXcoefs[8]
	if Fr2_crt_PolyExtra or (zcoa>=-6 and zcoa<=-4): # Critical Fr2 (outside use original formulation)
		Fr2_crit =FrXcoefs[0]*zcoa_scl**6+FrXcoefs[1]*zcoa_scl**5+FrXcoefs[2]*zcoa_scl**4+\
		FrXcoefs[3]*zcoa_scl**3+FrXcoefs[4]*zcoa_scl**2+FrXcoefs[5]*zcoa_scl + FrXcoefs[6]
	else: # Critical Fr2 (outside use linear extrapolation)
		if zcoa>=-4:
			dzcoa = zcoa_scl-zcoa_l
			Fr2_crit  = base_l+slope_l*dzcoa
		elif zcoa<=-6:
			dzcoa = zcoa_scl-zcoa_r
			Fr2_crit  = base_r+slope_r*dzcoa
	return Fr2_crit
####################################################################
def Fr2_minus_Fr2_crit(zp,l,kt,et,nu,g,cL,cEta,circ_p,FrXcoefs,Fr2_crt_PolyExtra,sector):
	wz =  get_rise_speed(l,2*zp,kt,et,nu,cL,cEta,method=2)
	Fr2 = circ_p*wz/(l**2/4*g)
	if sector >= 4:
		Fr2crt = 4.0
	else:
		Fr2crt = Fr2_crit_getter(l, zp, FrXcoefs, Fr2_crt_PolyExtra)
	return Fr2 - Fr2crt
####################################################################
from scipy.optimize import brenth
def root_find(l,zp_min,zp_max,kt,et,nu,cL,cEta,circ_p,g,FrXcoefs,Fr2_crt_PolyExtra,sector):
	# See if there is root within the segment
	diff_left  = Fr2_minus_Fr2_crit(zp_min,l,kt,et,nu,g,cL,cEta,circ_p,FrXcoefs,Fr2_crt_PolyExtra,sector)
	diff_right = Fr2_minus_Fr2_crit(zp_max,l,kt,et,nu,g,cL,cEta,circ_p,FrXcoefs,Fr2_crt_PolyExtra,sector)
	if (diff_left*diff_right < 0.0 ):
		root = brenth(Fr2_minus_Fr2_crit,zp_min,zp_max,
					  args=(l,kt,et,nu,g,cL,cEta,circ_p,FrXcoefs,Fr2_crt_PolyExtra,sector),
					  rtol=1e-4)
		has_root = True
	else:
		has_root = False; root = -1.0
	return has_root, root
####################################################################
@jit(nopython=True, cache=True, nogil=True)
def zl_sector(zlam_min,zlam_max,sector):
	if sector < 1 : # skip sector selection
		return zlam_min, zlam_max
	else:
		i = mod(sector-1,3)
		if i==0:
			zl_min = max(3,zlam_min)
			zl_max = max(3,zlam_max)
		elif i == 1:
			zl_min = 2
			zl_max = 3
		elif i == 2:
			zl_min = min(2,zlam_min)
			zl_max = min(2,zlam_max)
		return zl_min,zl_max			
def int_seg_find(l,zl_min,zl_max,kt,et,nu,cL,cEta,FrXcoefs,circ_p,g,Fr2_crt_PolyExtra,sector):
	# Search for integration range based on Fr and Fr_crit
	# Search is divided into regions
	# Number of regions is determined by
	# whether Poly extrapolation for Fr2_crt is used (6 regions) or not (3 regions)
	# Fr **always** increases with depth
	if Fr2_crt_PolyExtra: # At most 6 regions
		zl_breakpoints=[2.03417342, 2.19577958, 2.56982698, 3.03826383, 3.11406141]; num_zl_bpts=5
	else: # At most 6 regions
		zl_breakpoints=[2.03417342, 2.19577958, 2.56982698]; num_zl_bpts=3
	zl_list=[zl_min]; zl_list_size=1
	for i in range(num_zl_bpts): # Find the range considering zl_min and zl_max
		if zl_breakpoints[i] > zl_min and zl_breakpoints[i] < zl_max:
			zl_list.append(zl_breakpoints[i]);	zl_list_size += 1
	zl_list.append(zl_max);	zl_list_size += 1
	
	roots_list=[]; num_roots=0;
	for i in range(zl_list_size-1): # Find the roots
		has_root, root = root_find(l,zl_list[i]*l,zl_list[i+1]*l,kt,et,nu,cL,cEta,
									circ_p,g,FrXcoefs,Fr2_crt_PolyExtra,sector)
		if has_root:
			roots_list.append(root); num_roots += 1;
	
	num_seg=0; zp_seg=[]
	roots_list.insert(0,zl_min*l)
	roots_list.append(zl_max*l)
	for i in range(num_roots+1):
		mid = 0.5*(roots_list[i] + roots_list[i+1])
		diff_mid = Fr2_minus_Fr2_crit(mid,l,kt,et,nu,g,cL,cEta,circ_p,FrXcoefs,Fr2_crt_PolyExtra,sector)
		if diff_mid > 0:
			# print("Seg found from {:.3e} to {:.3e} at {:.3e}".format(roots_list[i], roots_list[i+1], mid))
			num_seg = num_seg+1;
			zp_seg.append((roots_list[i], roots_list[i+1]))
	return num_seg, zp_seg
####################################################################
def J_lambda(l,lst,ul2_lst,kt,et,cL,cEta,nu,g,rhoc,sig,Refitcoefs,FrXcoefs,Fr2_lst,zcoa_lst,F_tab,zl_min,zl_max,Fr2_crt_PolyExtra,F_tab_NP,sector):
	Reg,Bog,Weg,circ_p,n_lam,x,tau_vort=J_lambda_prep(l,lst,ul2_lst,kt,et,cL,cEta,nu,g,rhoc,sig)
	# Figure out number of segments in integration
	num_seg, zp_seg = int_seg_find(l,zl_min,zl_max,kt,et,nu,cL,cEta,FrXcoefs,circ_p,g,Fr2_crt_PolyExtra,sector)
	if num_seg == 0:
		return 0
	V_int = 0
	# Integrate in each segment
	for iseg in range(num_seg):
		nwz = 50 # in each depth segment
		zp_lst=linspace(zp_seg[iseg][0],zp_seg[iseg][1],nwz)
		wz_lst=zeros(nwz)
		for iz in range(nwz):
			wz_lst[iz] = get_rise_speed(l,2*zp_lst[iz],kt,et,nu,cL,cEta,method=2)
		V_int=V_int+quad(Ent_Volume_intgrand_jit, log(zp_seg[iseg][0]), log(zp_seg[iseg][1]), \
		                 args=(l,zp_lst,wz_lst,g,circ_p,Reg,Bog,Weg,Refitcoefs,Fr2_lst,zcoa_lst,F_tab,F_tab_NP,sector), \
		                 limit = 100,epsrel = 1e-5)[0]
	#---- Breakage probability ----#
	PB=erfc(x)+2/sqrt(pi)*x*exp(-(x**2))
	J_lam=n_lam*PB*V_int/tau_vort
	return J_lam
####################################################################
# When lambda is too large the resulting Fr2 is too small and \
# therefore results in unnecessary integral range.
def max_lambda(kt,et,nu,g,cL,cEta,zlam_min,zlam_max,FrXcoefs,Fr2_crt_PolyExtra):
	L=kt**1.5/et
	if not(zlam_min<=2 or zlam_max >= 3):
		return 100 * L # Too complicated
	else:
		lmax = 100 * L
		if zlam_min<=2.57 and zlam_max >= 2.57:
			zlam = 2.57
		elif zlam_min >= 3:
			zlam = zlam_min
		elif zlam_max <=2:
			zlam = 2
		else:
			print("Not implemented yet!!!")
			return -1
		for iloop in range(8):
			Fr2_cr = Fr2_crit_getter(1,zlam,FrXcoefs,Fr2_crt_PolyExtra)*0.95
			wz = get_rise_speed(lmax,zlam*lmax,kt,et,nu,cL,cEta,method=2)
			ulam = sqrt(ulambda_sq(lmax,kt,et,cL,cEta,nu,pope_spec=1.01))
			lmax = sqrt(pi*pi*lmax*ulam*wz/g/Fr2_cr)
		return lmax
####################################################################
def Jent_numerical_New(kt,et,nu,g,rhoc,sig,Table,zlam_min,zlam_max,wmeth,Fr2_crt_PolyExtra,F_tab_NP,sector):
	if (wmeth!=2 and wmeth>0):
		print('Not working in this mode.')
		return -1
	Refitcoefs=Table['Refitcoefs'];	FrXcoefs=Table['FrXcoefs']
	Fr2_lst=Table['flxfr_data']; zcoa_lst=Table['z_a_data']; F_tab=Table['F_lookuptable']
	cL,cEta=findcLceta(kt,et,nu,mode=1)
	# Note: for zp/lam > 6.751203194740859 the B is less than 0, therefore integrating over this point is meanning less.
	zl_min,zl_max=zl_sector(zlam_min,zlam_max,sector)
	zl_max = min (6.751203194740859,zlam_max)
	x1=sqrt(4*sig/rhoc/g); x2=sqrt(200*sig/rhoc/g); x4=max_lambda(kt,et,nu,g,cL,cEta,zl_min,zl_max,FrXcoefs,Fr2_crt_PolyExtra); # Lambda range
	# For speed get a table of ulambda_square
	nlst=400
	lst=logspace(log10(x1),log10(x4),nlst);	ul2_lst=zeros(nlst) #with dimension!
	for i in range(nlst):
		ul2_lst[i]=ulambda_sq(lst[i],kt,et,cL,cEta,nu,pope_spec=1.01)
	def intgrd(u,kt,et,cL,cEta,nu,g,rhoc,sig,Refitcoefs,FrXcoefs,Fr2_lst,zcoa_lst,F_tab,zl_min,zl_max,Fr2_crt_PolyExtra,F_tab_NP,sector):
		return J_lambda(exp(u),lst,ul2_lst,kt,et,cL,cEta,nu,g,rhoc,sig,Refitcoefs,FrXcoefs,Fr2_lst,zcoa_lst,F_tab,zl_min,zl_max,Fr2_crt_PolyExtra,F_tab_NP,sector)*exp(u)
	J=quadrature(intgrd,  log(x1), log(x2),
	             args=(kt,et,cL,cEta,nu,g,rhoc,sig,Refitcoefs,FrXcoefs,Fr2_lst,zcoa_lst,F_tab,zl_min,zl_max,Fr2_crt_PolyExtra,F_tab_NP,sector),
	             vec_func=False,maxiter=51,rtol=1e-3)[0] +\
	quadrature(intgrd,  log(x2), log(x4),
	            args=(kt,et,cL,cEta,nu,g,rhoc,sig,Refitcoefs,FrXcoefs,Fr2_lst,zcoa_lst,F_tab,zl_min,zl_max,Fr2_crt_PolyExtra,F_tab_NP,sector),
	            vec_func=False,maxiter=52,rtol=1e-3)[0]
	return J
####################################################################
# These constans are pre-calculate
#region
# @jit(nopython=True, cache=True, nogil=True)
# def Fr2_crit_lst_getter_helper(FrXcoefs,Fr2_crt_PolyExtra):
# 	if ~Fr2_crt_PolyExtra(): # Pre processing
# 		#left
# 		zcoa_l=(-4-FrXcoefs[7])/FrXcoefs[8]
# 		base_l = FrXcoefs[0]*zcoa_l**6+FrXcoefs[1]*zcoa_l**5+FrXcoefs[2]*zcoa_l**4+\
# 		FrXcoefs[3]*zcoa_l**3+FrXcoefs[4]*zcoa_l**2+FrXcoefs[5]*zcoa_l + FrXcoefs[6]
# 		slope_l = FrXcoefs[0]*zcoa_l**5*6+FrXcoefs[1]*zcoa_l**4*5+FrXcoefs[2]*zcoa_l**3*4+\
# 		FrXcoefs[3]*zcoa_l**2*3+FrXcoefs[4]*zcoa_l*2+FrXcoefs[5]
# 		#right
# 		zcoa_r=(-6-FrXcoefs[7])/FrXcoefs[8]
# 		base_r = FrXcoefs[0]*zcoa_r**6+FrXcoefs[1]*zcoa_r**5+FrXcoefs[2]*zcoa_r**4+\
# 		FrXcoefs[3]*zcoa_r**3+FrXcoefs[4]*zcoa_r**2+FrXcoefs[5]*zcoa_r + FrXcoefs[6]
# 		slope_r = FrXcoefs[0]*zcoa_r**5*6+FrXcoefs[1]*zcoa_r**4*5+FrXcoefs[2]*zcoa_r**3*4+\
# 		FrXcoefs[3]*zcoa_r**2*3+FrXcoefs[4]*zcoa_r*2+FrXcoefs[5]
# 	else:
# 		zcoa_l = 0; base_l = 0; slope_l = 0; zcoa_r = 0; base_r = 0; slope_r = 0
# 	return zcoa_l,base_l,slope_l,zcoa_r,base_r,slope_r
#endregion
def F_func_table_ext(Fr2_lst,Fr2_tgt,zcoa_lst,zcoa_tgt,F_tab,method,sector):
	d1  = zcoa_tgt.size-1;	d2  = Fr2_tgt.size-1
	d1o = zcoa_lst.size-1;	d2o = Fr2_lst.size-1
	F_tab_out=zeros((d1+1,d2+1))
	if method == "Linear Exrapolation":
		# z_a_data goes from -6 to -4; flxfr_data goes from 0 to 4
		# extrapolate zoa dimension first 
		for iz in range(d1+1):
			izcoa=-1; zcoa = zcoa_tgt[iz]
			if zcoa >= zcoa_lst[d1o-1]:
				izcoa=d1o-1; zcoa_lw=(zcoa_lst[d1o]-zcoa)/(zcoa_lst[d1o]-zcoa_lst[d1o-1]); 
			elif zcoa <= zcoa_lst[0]:
				izcoa=0; 	zcoa_lw=(zcoa_lst[1] -zcoa)/(zcoa_lst[1] -zcoa_lst[0]); 
			if izcoa == -1: #within the range
				for i in range(d1o):
					if zcoa>=zcoa_lst[i] and zcoa<zcoa_lst[i+1]:
						izcoa=i; zcoa_lw=(zcoa_lst[i+1]-zcoa)/(zcoa_lst[i+1]-zcoa_lst[i])
			if izcoa == -1:
				print('Sthg is very wrong(zcoa), {}, {}'.format(izcoa, zcoa_lw))
			F_lst=F_tab[izcoa,:]*zcoa_lw+F_tab[izcoa+1,:]*(1-zcoa_lw)
			# F=interp(Fr2,Fr2_lst,F_lst)
			for jf in range(d2+1):
				jfr=-1; Fr2 = Fr2_tgt[jf]
				if Fr2 >= Fr2_lst[d2o-1]:
					jfr=d2o-1; fr2_lw=(Fr2_lst[d2o]-Fr2)/(Fr2_lst[d2o]-Fr2_lst[d2o-1]); 
				elif Fr2 <= Fr2_lst[0]:
					jfr=0; 	fr2_lw=(Fr2_lst[1] -Fr2)/(Fr2_lst[1] -Fr2_lst[0]); 
				if jfr == -1: #within the range
					for i in range(len(Fr2_lst)-1):
						if Fr2>=Fr2_lst[i] and Fr2<Fr2_lst[i+1]:
							jfr=i; fr2_lw=(Fr2_lst[i+1]-Fr2)/(Fr2_lst[i+1]-Fr2_lst[i])
				if jfr == -1:
					print('Sthg is very wrong (Fr2)')
				flg = F_filter(zcoa,Fr2,sector)
				if (flg):
					F_tab_out[iz, jf] = max(0.0, F_lst[jfr]*fr2_lw+F_lst[jfr+1]*(1-fr2_lw))
				else:
					F_tab_out[iz, jf] = 0.0
	elif method == "Nearest Point":
		# RectBivariateSpline use nearest point by default.
		X = zcoa_lst
		Y = Fr2_lst
		# f = RectBivariateSpline(X, Y, F_tab)
		f = interp2d(X, Y, F_tab.T)
		for i in range(d1+1):
			for j in range(d2+1):
				F_tab_out[i,j] = f(zcoa_tgt[i],Fr2_tgt[j])
	else:
		print("ERROR: Wrong method parameter involked!")
	return F_tab_out
