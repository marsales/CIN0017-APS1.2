'''
CL1-2: According to Archimedes principle, the buoyancy force is
equal to the weight of fluid displaced by the submerged portion of an
object. For the depicted sphere, use bisection to determine the height
h of the portion that is above water. Employ the following values for
your computation: r = 1.25 m, ρs = density of sphere = 250 kg/m3,
and ρw = density of water = 1000 kg/m3. Note that the volume of the
above-water portion of the sphere can be computed with
V = ( (pi . h**2) / 3 ) * (3*r - h)
'''
import pandas as pd

r = 1.25

def v(h):
  return 3*r**3 - 3*r*h**2 + h**3

rows = []

def bisection(f, a, b, tol=1e-6, max_iter=1000):

  fa = f(a)
  fb = f(b)

  if (fa*fb >= 0):
    raise ValueError("f(a) and f(b) must have opposite signs.")


  for i in range(1, max_iter+1):

    c = (a+b)/2
    fc = f(c)

    rows.append({
        "iteration":i,
        "a":a,
        "b":b,
        "mid":c,
        "f(mid)":fc,
        "interval":b-a
    })

    if (fc == 0 or (b-a)/2 < tol):
      break

    if (fa*fc < 0):
      b = c
      fb = fc
    else:
      a = c
      fa = fc

  table = pd.DataFrame(rows)
  return c, table


a = 0
b = 2*r

root, table = bisection(v, a, b)

print("Approximate root:", root)
print("f(root) =", v(root))

table
