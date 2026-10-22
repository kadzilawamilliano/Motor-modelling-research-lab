import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


# ==================================
# Explicit Finite Difference Function
# ==================================

def explicit_fd(K, S0, T, r, sigma, S_max, M, N):

    dS = S_max / M
    dt = T / N

    # Stock price grid
    S = np.linspace(0, S_max, M + 1)

    # Initial payoff at maturity
    V = np.maximum(S - K, 0)

    # Coefficients
    i = np.arange(1, M)

    a = 0.5 * dt * (sigma**2 * i**2 - r*i)
    b = 1 - dt * (sigma**2 * i**2 + r)
    c = 0.5 * dt * (sigma**2 * i**2 + r*i)


    # Time stepping
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


    # Interpolate value at S0
    price = np.interp(S0, S, V)

    return price, S, V



# ==================================
# Streamlit Interface
# ==================================

st.title("European Call Option Pricing")
st.write(
    "Explicit Finite Difference Method (FTCS)"
)


# User inputs

K = st.number_input(
    "Strike Price (K)",
    value=50.0
)

S0 = st.number_input(
    "Current Stock Price (S0)",
    value=50.0
)

T = st.number_input(
    "Time to Maturity (Years)",
    value=1.0
)

r = st.number_input(
    "Risk-free Rate",
    value=0.05
)

sigma = st.number_input(
    "Volatility",
    value=0.20
)


# Numerical parameters

M = st.slider(
    "Space Steps (M)",
    50,
    300,
    100
)

N = st.slider(
    "Time Steps (N)",
    500,
    3000,
    1000
)


if st.button("Calculate Option Price"):

    S_max = 2*K

    price, S, V = explicit_fd(
        K,
        S0,
        T,
        r,
        sigma,
        S_max,
        M,
        N
    )


    st.success(
        f"European Call Price = {price:.4f}"
    )


    # Plot

    fig, ax = plt.subplots()

    ax.plot(S, V)

    ax.set_xlabel(
        "Stock Price"
    )

    ax.set_ylabel(
        "Option Value"
    )

    ax.set_title(
        "Option Price using Explicit Finite Difference"
    )

    ax.grid()


    st.pyplot(fig)
