import numpy as np
from scipy.integrate import quad
from scipy.optimize  import fsolve
from scipy.interpolate import RectBivariateSpline, interp2d
from numpy import sin, cos, pi, sqrt, interp, zeros
from numba import jit, float64
########################################################################################################################
def get_ug(d,rhoc,rhod,sig,nu,alphac):
	import numpy as np
	sigma_rhoc = sig/rhoc
	g = 9.81
	maxinit = 100
	rho_quotient = rhod/rhoc
	Eo = g*d**2/sigma_rhoc
	eps = 1.0; ug = 1.0e-4; ii = 1
	while eps > 1.0e-4 :
		Re = ug*d/nu
		Cd = 1/alphac*max(24e0/Re*(1+0.168e0*Re**0.75e0),8/3*Eo/(Eo+4))
		dfdu =3/4*Cd/(d/2)*ug
		fu =3/8*Cd/(d/2)*ug**2-(1-rho_quotient)*g
		un =ug - fu/dfdu
		eps =np.abs(un -ug)
		ii =ii+1
		if ii == maxinit :
			eps =1e-9;
			print('Warning: terminal velocity iteration divergence,change for approximation');
		else :
			ug = un
	return ug
########################################################################################################################
@jit([float64(float64, float64, float64, float64, float64, float64, float64, float64)], nopython=True, cache=True)
def Ek(k,c1,c2,C,L,p0,beta,eta):
	return C*k**(-5.0/3.0)*(k*L/((k*L)**2+c1)**0.5)**(5.0/3.0+p0)*np.exp(-beta*(((k*eta)**4+c2**4)**0.25-c2))
@jit([float64(float64, float64, float64, float64, float64, float64, float64, float64)], nopython=True, cache=True)
def Ekeps(k,c1,c2,C,L,p0,beta,eta):
	return 2*k**2*C*k**(-5.0/3.0)*(k*L/((k*L)**2+c1)**0.5)**(5.0/3.0+p0)*np.exp(-beta*(((k*eta)**4+c2**4)**0.25-c2))
########################################################################################################################
def findcLceta(kt,epsl,nu,mode):
	# =============================================
	# calculate cL and ceta used in Pope's spectrum
	# =============================================
	L=kt**1.5/epsl;  eta=(nu**3/epsl)**0.25
	C=1.5;	p0=2.0;	beta=5.2
	if mode==1 : # Direct integral
		kL=2*pi/L;  keta=2*pi/eta;
		def TKE_eqn(c1,c2):
			p1 = quad(Ek,   0,	       kL/100,		args=(c1,c2,C,L,p0,beta,eta),limit=100)[0]
			p2 = quad(Ek,   kL/100,     kL/50,  	args=(c1,c2,C,L,p0,beta,eta),limit=100)[0]
			p3 = quad(Ek,   kL/50,      kL/25,  	args=(c1,c2,C,L,p0,beta,eta),limit=100)[0]
			p4 = quad(Ek,   kL/25,      kL/5,   	args=(c1,c2,C,L,p0,beta,eta),limit=100)[0]
			p5 = quad(Ek,   kL/5,       kL,     	args=(c1,c2,C,L,p0,beta,eta),limit=100)[0]
			kmid1=(kL**2*keta**1)**(1/3);	kmid2=(kL**1*keta**2)**(1/3)
			p6 = quad(Ek,   kL,          kmid1, 	args=(c1,c2,C,L,p0,beta,eta),limit=100)[0]
			p7 = quad(Ek,   kmid1,       kmid2, 	args=(c1,c2,C,L,p0,beta,eta),limit=100)[0]
			p8 = quad(Ek,   kmid2,       keta,  	args=(c1,c2,C,L,p0,beta,eta),limit=100)[0]
			p9 = quad(Ek,   keta,        1e1*keta, args=(c1,c2,C,L,p0,beta,eta),limit=100)[0]
			p10= quad(Ek,   1e1*keta,    1e2*keta, args=(c1,c2,C,L,p0,beta,eta),limit=100)[0]
			p11= quad(Ek,   1e2*keta,    1e4*keta, args=(c1,c2,C,L,p0,beta,eta),limit=100)[0]
			p12= quad(Ek,   1e4*keta,    1e6*keta, args=(c1,c2,C,L,p0,beta,eta),limit=100)[0]
			p13= quad(Ek,   1e6*keta,    1e8*keta, args=(c1,c2,C,L,p0,beta,eta),limit=100)[0]
			p14= quad(Ek,   1e8*keta,    np.inf, 	args=(c1,c2,C,L,p0,beta,eta),limit=100)[0]

			return p1+p2+p3+p4+p5+p6+p7+p8+p9+p10+p11+p12+p13+p14-L**(2.0/3.0)
		def TKE_Diss_eqn(c1,c2):
			p1 = quad(Ekeps,   0,	       kL/100, 	args=(c1,c2,C,L,p0,beta,eta),limit=100)[0]
			p2 = quad(Ekeps,   kL/100,     kL/50,  	args=(c1,c2,C,L,p0,beta,eta),limit=100)[0]
			p3 = quad(Ekeps,   kL/50,      kL/25,  	args=(c1,c2,C,L,p0,beta,eta),limit=100)[0]
			p4 = quad(Ekeps,   kL/25,      kL/5,   	args=(c1,c2,C,L,p0,beta,eta),limit=100)[0]
			p5 = quad(Ekeps,   kL/5,       kL,     	args=(c1,c2,C,L,p0,beta,eta),limit=100)[0]
			kmid1=(kL**2*keta**1)**(1/3);	kmid2=(kL**1*keta**2)**(1/3)
			p6 = quad(Ekeps,   kL,          kmid1, 	args=(c1,c2,C,L,p0,beta,eta),limit=100)[0]
			p7 = quad(Ekeps,   kmid1,       kmid2, 	args=(c1,c2,C,L,p0,beta,eta),limit=100)[0]
			p8 = quad(Ekeps,   kmid2,       keta,  	args=(c1,c2,C,L,p0,beta,eta),limit=100)[0]
			p9 = quad(Ekeps,   keta,        1e1*keta, args=(c1,c2,C,L,p0,beta,eta),limit=100)[0]
			p10= quad(Ekeps,   1e1*keta,    1e2*keta, args=(c1,c2,C,L,p0,beta,eta),limit=100)[0]
			p11= quad(Ekeps,   1e2*keta,    1e4*keta, args=(c1,c2,C,L,p0,beta,eta),limit=100)[0]
			p12= quad(Ekeps,   1e4*keta,    1e6*keta, args=(c1,c2,C,L,p0,beta,eta),limit=100)[0]
			p13= quad(Ekeps,   1e6*keta,    1e8*keta, args=(c1,c2,C,L,p0,beta,eta),limit=100)[0]
			p14= quad(Ekeps,   1e8*keta,    np.inf, 	args=(c1,c2,C,L,p0,beta,eta),limit=100)[0]
			return p1+p2+p3+p4+p5+p6+p7+p8+p9+p10+p11+p12+p13+p14-eta**(-4.0/3.0)
		def Eqns(c):
			return (TKE_eqn(c[0],c[1]), TKE_Diss_eqn(c[0],c[1]))
		sol=fsolve(Eqns,[6.78, 0.4]);
		cL=sol[0]; ceta=sol[1];
		# ===  VALIDATE  ===
		kt = L**(2.0/3.0);	epsl = eta**(-4.0/3.0)
		kt_out  = TKE_eqn(cL,ceta) 		+ kt;
		eps_out = TKE_Diss_eqn(cL,ceta) 	+ epsl
		kt_err  = np.abs(kt_out	- kt)		/kt;
		eps_err = np.abs(eps_out- epsl)	/epsl;
		if kt_err>5e-2 or eps_err>5e-2:
			# print('Difference(Scaled): kt, eps: {:.3e}%  {:.3e}%'.format(kt_err*100, eps_err*100))
			# print('kt, epsl (org): {:.4e}  {:.4e}' .format(kt, epsl))
			# print('kt, epsl (int): {:.4e}  {:.4e}' .format(kt_out, eps_out))
			# print('Now trying the other method:')
			ReL=(L/eta)**(4.0/3.0)
			Relam=sqrt(20.0/3.0)*ReL**0.5
			cL1  =np.exp(-(4.478+18.362*C)/Relam**(1.075-0.07*C)-1.342+2.024*C)-1.913+2.169*C
			ceta1=np.exp((14.043-4.2222*C)/Relam**(1.986-0.363*C))-(0.089+0.339*C)
			kt_out  = TKE_eqn(cL1,ceta1) 		+ kt
			eps_out = TKE_Diss_eqn(cL1,ceta1)+ epsl
			kt_err1 = np.abs(kt_out	- kt)		/kt
			eps_err1= np.abs(eps_out- epsl)	/epsl
			# print('Difference(Scaled)(Ana): kt, eps: {:.3e}%  {:.3e}%'.format(kt_err1*100, eps_err1*100))
			if kt_err1+eps_err1>kt_err+eps_err:
				return (cL, ceta)
			else:
				return (cL1, ceta1)
		else:
			return (cL, ceta)
	elif mode==2 :# Interpolation
		ReL=(L/eta)**(4.0/3.0)
		Relam=sqrt(20.0/3.0)*ReL**0.5
		cL  =np.exp(-(4.478+18.362*C)/Relam**(1.075-0.07*C)-1.342+2.024*C)-1.913+2.169*C
		ceta=np.exp((14.043-4.2222*C)/Relam**(1.986-0.363*C))-(0.089+0.339*C)
		return (cL, ceta)
########################################################################################################################
@jit([float64(float64, float64, float64, float64, float64, float64, float64)], nopython=True, cache=True)
def u2_intgrand(k,l,cL,cEta,L,eta,epsl):
	kl=l*k
	# return 4.0/3.0*Spectrum(k,cL,cEta,kt,epsl,nu)*(1-3*(np.sin(k*l)/((k*l)**3) - np.cos(k*l)/((k*l)**2)))
	C=1.5;	p0=2.0;   be=5.2
	fl  = (k*L/((k*L)**2+cL)**0.5)**(5.0/3.0+p0)
	feta= np.exp(-be*(((k*eta)**4+cEta**4)**0.25-cEta))
	Ek  = C*epsl**(2.0/3.0)*k**(-5.0/3.0)*fl*feta
	if kl < 5e-4:
		return 0
	elif kl < 10:
		return 4/3*Ek*((1-3*(sin(kl)/((kl)**3) - cos(kl)/((kl)**2))))
	else:
		return  4/3*Ek
########################################################################################################################
# @vectorize([float64(float64, float64, float64, float64, float64, float64)], nopython=True, cache=True)
def ulambda_sq(l,kt,epsl,cL,cEta,nu,pope_spec):
	import sys
	if pope_spec == 1.01 or pope_spec == 1:
		L=kt**1.5/epsl;  eta=(nu**3/epsl)**0.25
		kL=2*pi/L;	keta=2*pi/eta;
		# 0 - kL
		ulam00 = quad(u2_intgrand,   1e-30,      kL/100,	args=(l,cL,cEta,L,eta,epsl), limit=50, epsrel=1e-2,epsabs=1.0e-8)[0]
		ulam01 = quad(u2_intgrand,   kL/100,     kL/50, 	args=(l,cL,cEta,L,eta,epsl), limit=50, epsrel=1e-2,epsabs=1.0e-8)[0]
		ulam02 = quad(u2_intgrand,   kL/50,      kL/25, 	args=(l,cL,cEta,L,eta,epsl), limit=50, epsrel=1e-2,epsabs=1.0e-8)[0]
		ulam03 = quad(u2_intgrand,   kL/25,      kL/5,  	args=(l,cL,cEta,L,eta,epsl), limit=200,epsrel=2e-2,epsabs=1.0e-8)[0]
		ulam04 = quad(u2_intgrand,   kL/5,       kL,    	args=(l,cL,cEta,L,eta,epsl), limit=200,epsrel=2e-2,epsabs=1.0e-8)[0]
		# kL - keta
		kmid1=(kL**2*keta**1)**(1/3)
		kmid2=(kL**1*keta**2)**(1/3)
		ulam11 = quad(u2_intgrand,  kL,          kmid1,  	args=(l,cL,cEta,L,eta,epsl), limit=200, epsrel=1e-2,epsabs=1.0e-8)[0]
		ulam12 = quad(u2_intgrand,  kmid1,       kmid2,  	args=(l,cL,cEta,L,eta,epsl), limit=200, epsrel=1e-2,epsabs=1.0e-8)[0]
		ulam13 = quad(u2_intgrand,  kmid2,       keta,  	args=(l,cL,cEta,L,eta,epsl), limit=200, epsrel=1e-2,epsabs=1.0e-8)[0]
		# keta - 1e10 keta
		ulam2 = quad(u2_intgrand,   keta,        1e1*keta,	args=(l,cL,cEta,L,eta,epsl), limit=50, epsrel=1e-2,epsabs=1.0e-8)[0]
		ulam3 = quad(u2_intgrand,   1e1*keta,    1e2*keta,	args=(l,cL,cEta,L,eta,epsl), limit=50, epsrel=1e-2,epsabs=1.0e-8)[0]
		ulam4 = quad(u2_intgrand,   1e2*keta,    1e4*keta,	args=(l,cL,cEta,L,eta,epsl), limit=50, epsrel=1e-2,epsabs=1.0e-8)[0]
		ulam5 = quad(u2_intgrand,   1e4*keta,    1e6*keta,	args=(l,cL,cEta,L,eta,epsl), limit=50, epsrel=1e-2,epsabs=1.0e-8)[0]
		ulam6 = quad(u2_intgrand,   1e6*keta,    1e8*keta,	args=(l,cL,cEta,L,eta,epsl), limit=50, epsrel=1e-2,epsabs=1.0e-8)[0]
		# current - infinity
		ulam7 = quad(u2_intgrand,   1e8*keta,    np.inf, 	args=(l,cL,cEta,L,eta,epsl), limit=50,epsrel=1e-2,epsabs=1.0e-8)[0]

		usquare = ulam00+ulam01+ulam02+ulam03+ulam04+ulam11+ulam12+ulam13+ulam2+ulam3+ulam4+ulam5+ulam6+ulam7
		return usquare
		if ulam00*ulam01*ulam02*ulam03*ulam04*ulam11*ulam12*ulam13*ulam2*ulam3*ulam4*ulam5*ulam6*ulam7<=0:
			sys.exit('usquare<=0')
	else:
		return 2*(epsl*l)**(2.0/3.0)
########################################################################################################################
def assemble_matrix(A, z, nut, VT, dt):
	Nx = z.size
	i = 0
	A[i, i] = 1.0
	A[i, i+1] = -1.0
	for i in range(1, Nx-1):
		
		A[i, i-1] = 0.0
		A[i, i+1] = 0.0
		A[i, i] = 0.0
		
		hi = (z[i+1]-z[i-1])/2.0
		hp = z[i+1]-z[i]
		hm = z[i]-z[i-1]
		
		nut_m = 0.5*(nut[i-1]+nut[i])
		nut_p = 0.5*(nut[i+1]+nut[i])
		
		#Diffusion term
		A[i, i-1] = -nut_m/hi/hm
		A[i, i] = 1.0/hi*(nut_m/hm+nut_p/hp)
		A[i, i+1] = -nut_p/hi/hp
		
		#Convection term [assumes rising velocity, u<0]
		A[i, i] = A[i, i] + VT/hp
		A[i, i+1] = A[i, i+1] - VT/hp
		
		#Temporal term
		A[i, i] = A[i, i] + 1.0/dt
		
		#RHS
		#b[i] = N0[i]/dt + S[i];
		
		i = Nx-1
		A[i, i] = 1.0
		A[i, i-1] = -1.0  # b(i) = 0.0; %zero derivative
	return A
########################################################################################################################
def assemble_rhs(N0,S,dt):
	Nx= S.size
	b = N0/dt + S
	b[0] = 0.0; #zero derivative at i=0
	b[Nx-1] = 0.0; #zero derivative at i=Nx-1
	return b
########################################################################################################################