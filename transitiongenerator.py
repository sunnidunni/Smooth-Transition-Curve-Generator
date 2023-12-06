import numpy as np
import math
import sympy as sym
import numpy as np

pi = math.pi

t = sym.Symbol('t')


fx1 = 2*sym.cos(t)
fy1 = 0.5*t
fz1 = 2*sym.sin(t)

fx2 = 1/10* t*sym.cos(t)
fy2 = 1/15* t*sym.sin(t)+8
fz2 = 0.5*t-7

t1 = 4*pi

t2 = 4.5*pi

fx1d1 = sym.diff(fx1)    
fx1d2 = sym.diff(fx1d1)
fx1d3 = sym.diff(fx1d2)

fy1d1 = sym.diff(fy1)
fy1d2 = sym.diff(fy1d1)
fy1d3 = sym.diff(fy1d2)

fz1d1 = sym.diff(fz1)
fz1d2 = sym.diff(fz1d1)
fz1d3 = sym.diff(fz1d2)

fx2d1 = sym.diff(fx2)
fx2d2 = sym.diff(fx2d1)
fx2d3 = sym.diff(fx2d2)

fy2d1 = sym.diff(fy2)
fy2d2 = sym.diff(fy2d1)
fy2d3 = sym.diff(fy2d2)

fz2d1 = sym.diff(fz2)
fz2d2 = sym.diff(fz2d1)
fz2d3 = sym.diff(fz2d2)

#transfer curve t1 = -pi, t2 = pi. Equations calculated based on these 2 ts.

A = np.array([
    [1,-pi,pi**2,-pi**3,pi**4,-pi**5,pi**6,-pi**7],
    [0,1,-2*pi,3*pi**2,-4*pi**3,5*pi**4,-6*pi**5,7*pi**6],
    [0,0,2,-6*pi,12*pi**2,-20*pi**3,30*pi**4,-42*pi**5],
    [0,0,0,6,-24*pi,60*pi**2,-120*pi**3,210*pi**4],
    [1,pi,pi**2,pi**3,pi**4,pi**5,pi**6,pi**7],
    [0,1,2*pi,3*pi**2,4*pi**3,5*pi**4,6*pi**5,7*pi**6],
    [0,0,2,6*pi,12*pi**2,20*pi**3,30*pi**4,42*pi**5],
    [0,0,0,6,24*pi,60*pi**2,120*pi**3,210*pi**4]
              ])

x11 = float(fx1.subs(t,t1))
x22 = float(fx1d1.subs(t,t1))
x33 = float(fx1d2.subs(t,t1))
x44 = float(fx1d3.subs(t,t1))
x55 = float(fx2.subs(t,t2))
x66 = float(fx2d1.subs(t,t2))
x77 = float(fx2d2.subs(t,t2))
x88 = float(fx2d3.subs(t,t2))

y11 = float(fy1.subs(t,t1))
y22 = float(fy1d1.subs(t,t1))
y33 = float(fy1d2.subs(t,t1))
y44 = float(fy1d3.subs(t,t1))
y55 = float(fy2.subs(t,t2))
y66 = float(fy2d1.subs(t,t2))
y77 = float(fy2d2.subs(t,t2))
y88 = float(fy2d3.subs(t,t2))

z11 = float(fz1.subs(t,t1))
z22 = float(fz1d1.subs(t,t1))
z33 = float(fz1d2.subs(t,t1))
z44 = float(fz1d3.subs(t,t1))
z55 = float(fz2.subs(t,t2))
z66 = float(fz2d1.subs(t,t2))
z77 = float(fz2d2.subs(t,t2))
z88 = float(fz2d3.subs(t,t2))

#solving for x

#[x2(t1),x2'(t1),x2''(t1),x2'''(t1),x2(t2),x2'(t2),x2''(t2),x2'''(t2)]

y1 = np.array([x11,x22,x33,x44,x55,x66,x77,x88])

x = np.linalg.solve(A, y1)

#solving for y

#[y2(t1),y2'(t1),y2''(t1),y2'''(t1),y2(t2),y2'(t2),y2''(t2),y2'''(t2)]

y2 = np.array([y11,y22,y33,y44,y55,y66,y77,y88])

y = np.linalg.solve(A, y2)

#solving for z

#[z2(t1),z2'(t1),z2''(t1),z2'''(t1),z2(t2),z2'(t2),z2''(t2),z2'''(t2)]

y3 = np.array([z11,z22,z33,z44,z55,z66,z77,z88])

z = np.linalg.solve(A, y3)


#formats the transfer equation
def makeout(x):
    x = x.tolist()
    out = ""
    for i in range(8):
        temp = str(x[i])
        if i == 0:
            out += temp
        else:
            if i == 1:
                out+= temp+"t"
            else:
                out+= temp+"t^"+str(i)
        if i != 7:
            if str(x[i+1])[0] != "-":
                out += "+"
    return out

xout = makeout(x)
yout = makeout(y)
zout = makeout(z)

print("(("+xout+","+yout+","+zout+"),t,-pi,pi)")


    


