import numpy as np
import math
import sympy as sym

pi = math.pi
t = sym.Symbol('t')

#curve 1
fx1=  sym.sin(t)
fy1=  sym.cos(t)
fz1=  -t+10

#curve 2
fx2= sym.cos(t)
fy2= 1/2*t
fz2= sym.sin(t)

time1 = 2*pi

time2 = 4*pi


t1 = time1
t2 = time2

#calculating derivatives for first 2 curves
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

#substituting time
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

def round(x):
    for i in range(len(x)):
        if -1*10**(-10) < x[i] < 1*10**(-10):
            x[i] = 0
    return x

#prints out formatted transfer curve equation
xout = makeout(x)
yout = makeout(y)
zout = makeout(z)

print("(("+xout+","+yout+","+zout+"),t,-pi,pi)")
print("\n")

#getting transition curve equations
resfx1 = x[0] + x[1]*t + x[2]*t**2 + x[3]*t**3 + x[4]*t**4 + x[5]*t**5 + x[6]*t**6 + x[7]*t**7
resfy1 = y[0] + y[1]*t + y[2]*t**2 + y[3]*t**3 + y[4]*t**4 + y[5]*t**5 + y[6]*t**6 + y[7]*t**7
resfz1 = z[0] + z[1]*t + z[2]*t**2 + z[3]*t**3 + z[4]*t**4 + z[5]*t**5 + z[6]*t**6 + z[7]*t**7

#gets derivatives of transfer curve
resfx1d1 = sym.diff(resfx1)    
resfx1d2 = sym.diff(resfx1d1)
resfx1d3 = sym.diff(resfx1d2)

resfy1d1 = sym.diff(resfy1)
resfy1d2 = sym.diff(resfy1d1)
resfy1d3 = sym.diff(resfy1d2)

resfz1d1 = sym.diff(resfz1)
resfz1d2 = sym.diff(resfz1d1)
resfz1d3 = sym.diff(resfz1d2)

#substitutes ts to transfer curve
t1 = -pi
t2 = pi

resx11 = float(resfx1.subs(t,t1))
resx22 = float(resfx1d1.subs(t,t1))
resx33 = float(resfx1d2.subs(t,t1))
resx44 = float(resfx1d3.subs(t,t1))

resy11 = float(resfy1.subs(t,t1))
resy22 = float(resfy1d1.subs(t,t1))
resy33 = float(resfy1d2.subs(t,t1))
resy44 = float(resfy1d3.subs(t,t1))

resz11 = float(resfz1.subs(t,t1))
resz22 = float(resfz1d1.subs(t,t1))
resz33 = float(resfz1d2.subs(t,t1))
resz44 = float(resfz1d3.subs(t,t1))

dresx11 = float(resfx1.subs(t,t2))
dresx22 = float(resfx1d1.subs(t,t2))
dresx33 = float(resfx1d2.subs(t,t2))
dresx44 = float(resfx1d3.subs(t,t2))

dresy11 = float(resfy1.subs(t,t2))
dresy22 = float(resfy1d1.subs(t,t2))
dresy33 = float(resfy1d2.subs(t,t2))
dresy44 = float(resfy1d3.subs(t,t2))

dresz11 = float(resfz1.subs(t,t2))
dresz22 = float(resfz1d1.subs(t,t2))
dresz33 = float(resfz1d2.subs(t,t2))
dresz44 = float(resfz1d3.subs(t,t2))

print([fx1,fy1,fz1])
print("t1 = " + str(time1))
print([fx2,fy2,fz2])
print("t2 = " + str(time2))
print("")

curvature1 = np.linalg.norm(np.cross((x22,y22,z22),(x33,y33,z33))) / np.linalg.norm([x22,y22,z22])**3
print("curve 1 curvature at t1: " + str(curvature1))

rescurvature1 = np.linalg.norm(np.cross((resx22,resy22,resz22),(resx33,resy33,resz33))) / np.linalg.norm([resx22,resy22,resz22])**3

print("transition curve curvature at t1: " + str(rescurvature1))

curvature2 = np.linalg.norm(np.cross((x66,y66,z66),(x77,y77,z77))) / np.linalg.norm([x66,y66,z66])**3
print("curve 2 curvature at t2: " + str(curvature2))

rescurvature2 = np.linalg.norm(np.cross((dresx22,dresy22,dresz22),(dresx33,dresy33,dresz33))) / np.linalg.norm([dresx22,dresy22,dresz22])**3
print("transition curve curvature at t2: " + str(rescurvature2))

torsion1 = np.dot(np.cross([x22,y22,z22],[x33,y33,z33]),[x44,y44,z44])/ ((np.linalg.norm(np.cross([x22,y22,z22],[x33,y33,z33])))**2)
print("curve 1 torsion at t1: " + str(torsion1))

restorsion1 = np.dot(np.cross([resx22,resy22,resz22],[resx33,resy33,resz33]),[resx44,resy44,resz44])/ ((np.linalg.norm(np.cross([resx22,resy22,resz22],[resx33,resy33,resz33])))**2)
print("transition curve torsion at t1: " + str(restorsion1))

torsion2 = np.dot(np.cross([x66,y66,z66],[x77,y77,z77]),[x88,y88,z88])/ ((np.linalg.norm(np.cross([x66,y66,z66],[x77,y77,z77])))**2)
print("curve 2 torsion at t2: " + str(torsion2))

restorsion2 = np.dot(np.cross([dresx22,dresy22,dresz22],[dresx33,dresy33,dresz33]),[dresx44,dresy44,dresz44])/ ((np.linalg.norm(np.cross([dresx22,dresy22,dresz22],[dresx33,dresy33,dresz33])))**2)
print("transition curve torsion at t2: " + str(restorsion2))

print("")

print("Frenet Frames: \n")

#unit tangent
T1 = round([x22,y22,z22] / np.linalg.norm([x22,y22,z22]))
resT1 = round([resx22,resy22,resz22] / np.linalg.norm([resx22,resy22,resz22]))
T2 = round([x66,y66,z66]/ np.linalg.norm([x66,y66,z66]))
resT2 = round([dresx22,dresy22,dresz22] / np.linalg.norm([dresx22,dresy22,dresz22]))
print("curve 1 unit tangent at t1: " + str(T1))
print("transition curve unit tangent at t1: " + str(resT1))
print("curve 2 unit tangent at t2: " + str(T2))
print("transition curve unit tangent at t2: " + str(resT2))

#unit normal
'''
N1 = round([x33,y33,z33]/ np.linalg.norm([x33,y33,z33]))
resN1 = round([resx33,resy33,resz33] / np.linalg.norm([resx33,resy33,resz33]))
N2 = round([x77,y77,z77]/np.linalg.norm([x77,y77,z77]))
resN2 = round([dresx33,dresy33,dresz33]/np.linalg.norm([dresx33,dresy33,dresz33]))

print("curve 1 unit normal at t1: " + str(N1))
print("transition curve unit normal at t1: " + str(resN1))
print("curve 2 unit normal at t2: " + str(N2))
print("transition curve unit normal at t2: " + str(resN2))
'''
N11 = round(np.cross([x22,y22,z22],np.cross([x33,y33,z33],[x22,y22,z22])) / (np.linalg.norm([x22,y22,z22])* np.linalg.norm(np.cross([x33,y33,z33],[x22,y22,z22]))))
resN11 = round(np.cross([resx22,resy22,resz22],np.cross([resx33,resy33,resz33],[resx22,resy22,resz22])) / (np.linalg.norm([resx22,resy22,resz22])* np.linalg.norm(np.cross([resx33,resy33,resz33],[resx22,resy22,resz22]))))
N22 = round(np.cross([x66,y66,z66],np.cross([x77,y77,z77],[x66,y66,z66])) / (np.linalg.norm([x66,y66,z66])* np.linalg.norm(np.cross([x77,y77,z77],[x66,y66,z66]))))
resN22 = round(np.cross([dresx22,dresy22,dresz22],np.cross([dresx33,dresy33,dresz33],[dresx22,dresy22,dresz22])) / (np.linalg.norm([dresx22,dresy22,dresz22])* np.linalg.norm(np.cross([dresx33,dresy33,dresz33],[dresx22,dresy22,dresz22]))))

print("curve 1 unit normal at t1: " + str(N11))
print("transition curve unit normal at t1: " + str(resN11))
print("curve 2 unit normal at t2: " + str(N22))
print("transition curve unit normal at t2: " + str(resN22))

B1 = round(np.cross(T1,N11))
resB1 = round(np.cross(resT1,resN11))
B2 = round(np.cross(T2,N22))
resB2 = round(np.cross(resT2,resN22))
print("curve 1 unit binormal at t1: " + str(B1))
print("transition curve unit binormal at t1: " + str(resB1))
print("curve 2 unit binormal at t2: " + str(B2))
print("transition curve unit binormal at t2: " + str(resB2))
                    
            


    


