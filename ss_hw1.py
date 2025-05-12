import numpy as np
import matplotlib.pyplot as plt

student_num = "18332629011"
O1 = int(student_num[-1])
O2 = int(student_num[-2])
O3 = int(student_num[-3])
O4 = int(student_num[-4])

n_x = np.arange(O3, O4 + 12 + 1)
x = 2.0 ** (-n_x)
n_h = np.arange(O3, O2 + 12 + 1)
h = np.ones_like(n_h, dtype=float)

y = np.convolve(x, h)
n_y = np.arange(n_x[0] + n_h[0], n_x[-1] + n_h[-1] + 1)

plt.figure()
markerline, stemlines, baseline = plt.stem(n_y, y)
baseline.set_visible(False)
plt.xlabel("n")
plt.ylabel("y[n]")
plt.title("Sistem Çıkışı y[n]")
plt.show()
