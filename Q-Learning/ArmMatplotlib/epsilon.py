
import matplotlib.pyplot as plt

epsilon = 0.99
EPSILON_DECAY = 0.00005
MIN_EPSILON = 0.05

x = [epsilon]

for timestep in range(1, 5000):
    if epsilon > MIN_EPSILON:
        epsilon = epsilon * 1.0/(1.0 + EPSILON_DECAY*timestep)
    x.append(epsilon)
plt.plot(x, 'b', linewidth=3)


epsilon = 0.99
EPSILON_DECAY = 0.000001
MIN_EPSILON = 0.05

x = [epsilon]

for timestep in range(1, 5000):
    if epsilon > MIN_EPSILON:
        epsilon = epsilon * 1.0/(1.0 + EPSILON_DECAY*timestep)
    x.append(epsilon)
plt.plot(x, 'r', linewidth=3)

plt.xlabel('# of episode', size=15)
plt.ylabel('epsilon', size=15)
plt.legend(['previous exploration', 'current exploration'])
plt.show()