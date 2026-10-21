import numpy as np
import matplotlib.pyplot as plt


# ==========================
# Parameters
# ==========================

K = 50          # Strike price
T = 1.0         # Time to maturity
r = 0.05        # Risk-free rate
sigma = 0.20    # Volatility

S_max = 100     # Maximum stock price
M = 100         # Space steps
N = 1000        # Time steps

S0 = 50         # Current stock price


# ==========================
# Grid
# ==========================

dS = S_max / M
dt = T / N

S = np.linspace(0, S_max, M + 1)


# ==========================
# Initial condition
# European Call payoff
# ==========================

V = np.maximum(S - K, 0)


# ==========================
# Explicit coefficients
# ==========================

i = np.arange(1, M)

a = 0.5 * dt * (sigma**2 * i**2 - r*i)
b = 1 - dt * (sigma**2 * i**2 + r)
c = 0.5 * dt * (sigma**2 * i**2 + r*i)


# ==========================
# Time stepping
# ==========================

for n in range(N):

    V_old = V.copy()

    for j in range(1, M):

        V[j] = (
            a[j-1] * V_old[j-1]
            + b[j-1] * V_old[j]
            + c[j-1] * V_old[j+1]
        )


    # Boundary conditions

    V[0] = 0

    V[-1] = S_max - K*np.exp(-r*(T-(n+1)*dt))


# ==========================
# Option price today
# ==========================

price = np.interp(S0, S, V)

print("European Call Price =", price)


# ==========================
# Plot
# ==========================

plt.plot(S, V)
plt.xlabel("Stock Price S")
plt.ylabel("Option Value V")
plt.title("Explicit Finite Difference Method")
plt.grid()
plt.show()
