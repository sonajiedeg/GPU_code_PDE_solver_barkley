import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------
# --------------- 1D wave equation stability analysis ------------------------------------------------
# ----------------------------------------------------------------------------------------------------
print('1D wave equation stability analysis')
# pde
u_tt, u_xx, c = sp.symbols( 'u_tt,u_xx,c' )
pde = sp.Eq(u_tt, c**2 * u_xx)
# discretization
# ujn, j ~ position, n ~ time
u00, u01, u02, u10, u11, u12, u20, u21, u22 = sp.symbols( 'u00,u01,u02,u10,u11,u12,u20,u21,u22' )
dx, dt = sp.symbols( 'dx,dt', real=True )
pdeD = pde.subs( u_tt, ( u12 - 2*u11 + u10 ) / dt**2 )
pdeD = pdeD.subs( u_xx, ( u21 - 2*u11 + u01 ) / dx**2 )
# substitute one mode of the error
e0, e1, e2, k = sp.symbols( 'e0,e1,e2,k', real=True )
uD_to_error = [(u00,e0*sp.exp(-sp.I*k*dx)), (u01,e1*sp.exp(-sp.I*k*dx)), (u02,e2*sp.exp(-sp.I*k*dx)), (u10,e0), (u11,e1), (u12,e2), (u20,e0*sp.exp(sp.I*k*dx)), (u21,e1*sp.exp(sp.I*k*dx)), (u22,e2*sp.exp(sp.I*k*dx))]
pdeDE = pdeD.subs( uD_to_error )
# check if this is same as the sol. from presentation
s = sp.symbols('s', real=True)
e2c = sp.solve(pdeDE,e2)[0].subs( dt**2*c**2/dx**2,s )
e2ct = e2c.rewrite(sp.exp).simplify().trigsimp()
# e2c = sp.solve(pdeDE,e2).subs( dt**2*c**2/dx**2,s )
result = [a.trigsimp() for a in e2c.rewrite(sp.cos).expand().as_real_imag()]
# char. eq. of difference eq. with char. roots r1 r2
r = sp.symbols( 'r', real=True )
pdeDEC = pdeDE.subs( [ (e2, r**2), (e1, r), (e0,1) ] )
roots = [ x.simplify() for x in sp.solve(pdeDEC, r) ]
roots2 = [ x**2 for x in roots ]
roots2 = [ sp.collect( x, sp.exp(2*sp.I*k*dx) ) for x in roots2 ]
# sp.solve_rational_inequalities( [ [ ( ( roots[0]**2**(1/2), 1), '<=') ] ] )
theta = sp.symbols('theta', real=True)
r_simple = [ x.subs(sp.exp(sp.I*k*dx), sp.exp(sp.I*theta)) for x in roots ]
r_simple = [ sp.simplify(x) for x in r_simple ]

r_simple = [ x.subs(sp.exp(sp.I*theta) + sp.exp(-sp.I*theta), 2*sp.cos(theta)) for x in r_simple ]

exp_trig_subs = { sp.exp(sp.I*theta):sp.cos(theta)+sp.I*sp.sin(theta), sp.exp(2*sp.I*theta): (sp.cos(theta)+sp.I*sp.sin(theta))**2 }

r_trig = [ x.subs(exp_trig_subs) for x in r_simple ]
# r_trig_s = [ sp.trigsimp(x) for x in r_trig ]
# r_trig_s_a_2 = [ sp.trigsimp(sp.re(x)**2 + sp.im(x)**2) for x in r_trig ]

print('\tpde:', pde)
print('\tpde discr.:', pdeD)
print('\tpde discr. with error(von Neumann/Fourier):', pdeDE)
print('\te2c = ', e2c )
print('\te2ct = ', e2ct )
print('\tresult = ', result)
print('\tchar eq:', pdeDEC)
print('\tchar. roots of difference eq:', roots  )
print('\tchar. roots simplified:', r_simple)
print('\tchar. roots in trig form:', r_trig)
# print('\tchar. roots in trig form simplify:', r_trig_s)
# print('\tchar. abs roots in trig form simplify ^2:', r_trig_s_a_2)

tmp = 3 + 3*sp.I
print('test',sp.Abs( sp.re(tmp)**2 + sp.im(tmp)**k ))
