import glob
import pandas as pd
import numpy as np
df = pd.read_csv('richFaces_focus.csv')
data = pd.DataFrame.to_numpy(df)
index = data[:, 1]
d = (data[:, 2:])
x = d[:, ::2]
y = d[:, 1::2]
add_0 = np.zeros(x.shape[0])
x = np.c_[add_0.T, x]
y = np.c_[add_0.T, y]
print(x, y)
print(x.shape, y.shape)

# forehead
print(x[0][76], x[0][75], y[0][76], y[0][75])
w = []
h = []

for i in range(x.shape[0]):
    w.append(int(abs(x[i][76] - x[i][75])))
    h.append(int(abs(y[i][73] - y[i][21])))
w = np.array(w)
h = np.array(h)
a = w * h
print(a, a.shape)
forehead_area = np.c_[index.T, a.T]
print(forehead_area)
df = pd.DataFrame(forehead_area)
df.to_csv("./face_chars/forehead_area.csv")


# eyebrow
l = []
r = []

for i in range(x.shape[0]):
    l.append((abs((y[i][18] - y[i][22])/(x[i][18] - x[i][22]))))
    r.append((abs((y[i][23] - y[i][27])/(x[i][23] - x[i][27]))))
l = np.array(l)
r = np.array(r)
s = (l+r)/2
print(s, s.shape)
eyebrow_slope = np.c_[index.T, s.T]
print(eyebrow_slope)
df = pd.DataFrame(eyebrow_slope)
df.to_csv("./face_chars/eyebrow_slope.csv")

# cheek
la = []
ra = []

for i in range(x.shape[0]):
    la.append((abs((y[i][41] - y[i][49])*(x[i][3] - x[i][32]))))
    ra.append((abs((y[i][48] - y[i][55])*(x[i][36] - x[i][15]))))
la = np.array(la)
ra = np.array(ra)
a = (la+ra)/2
print(a, a.shape)
cheek_area = np.c_[index.T, a.T]
print(cheek_area)
df = pd.DataFrame(cheek_area)
df.to_csv("./face_chars/cheek_area.csv")

# chin
w = []
h = []

for i in range(x.shape[0]):
    w.append(int(abs(x[i][8] - x[i][10])))
    h.append(int(abs(y[i][58] - y[i][9])))
w = np.array(w)
h = np.array(h)
a = w * h
print(a, a.shape)
chin_area = np.c_[index.T, a.T]
print(chin_area)
df = pd.DataFrame(chin_area)
df.to_csv("./face_chars/chin_area.csv")


# nose

h = []

for i in range(x.shape[0]):
    h.append(int(abs(y[i][58] - y[i][9])))
h = np.array(h)
print(h, h.shape)
nose_l = np.c_[index.T, h.T]
print(nose_l)
df = pd.DataFrame(nose_l)
df.to_csv("./face_chars/nose_l.csv")
