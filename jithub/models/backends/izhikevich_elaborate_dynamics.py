from numba import jit
import numpy as np
@jit(nopython=True)
def get_vm_four(
    C=89.7960714285714,
    a=0.01,
    b=15,
    c=-60,
    d=10,
    k=1.6,
    vPeak=(86.364525297619 - 65.2261863636364),
    vr=-65.2261863636364,
    vt=-50,
    I=[],
    dt = 0.25
):
    tau = dt
    N = len(I)

    v = vr * np.ones(N)
    u = np.zeros(N)
    v[0] = vr
    for i in range(N):
        # forward Euler method
        v[i + 1] = v[i] + tau * (k * (v[i] - vr) * (v[i] - vt) - u[i] + I[i]) / C
        u[i + 1] = u[i] + tau * a * (b * (v[i] - vr) - u[i])
        # Calculate recovery variable
        if v[i + 1] > (vPeak - 0.1 * u[i + 1]):
            v[i] = vPeak - 0.1 * u[i + 1]
            v[i + 1] = c + 0.04 * u[i + 1]
            # Reset voltage
            if (u[i] + d) < 670:
                u[i + 1] = u[i + 1] + d
                # Reset recovery variable
            else:
                u[i + 1] = 670

    return v


@jit(nopython=True)
def get_vm_five(
    C=89.7960714285714,
    a=0.01,
    b=15,
    c=-60,
    d=10,
    k=1.6,
    vPeak=(86.364525297619 - 65.2261863636364),
    vr=-65.2261863636364,
    vt=-50,
    I=[],
    dt = 0.25
):
    N = len(I)

    tau = dt
    # dt
    v = vr * np.ones(N)
    u = np.zeros(N)
    v[0] = vr
    for i in range(N):
        # forward Euler method
        v[i + 1] = v[i] + tau * (k * (v[i] - vr) * (v[i] - vt) - u[i] + I[i]) / C

        if v[i + 1] < d:
            u[i + 1] = u[i] + tau * a * (0 - u[i])
        else:
            u[i + 1] = u[i] + tau * a * ((0.025 * (v[i] - d) ** 3) - u[i])

        if v[i + 1] >= vPeak:
            v[i] = vPeak
            v[i + 1] = c

    return v


@jit(nopython=True)
def get_vm_six(
    C=89.7960714285714,
    a=0.01,
    b=15,
    c=-60,
    d=10,
    k=1.6,
    vPeak=(86.364525297619 - 65.2261863636364),
    vr=-65.2261863636364,
    vt=-50,
    I=[],
    dt = 0.25
):
    tau = dt
    # dt
    N = len(I)

    v = vr * np.ones(N)
    u = np.zeros(N)
    v[0] = vr
    for i in range(N):
        # forward Euler method
        v[i + 1] = v[i] + tau * (k * (v[i] - vr) * (v[i] - vt) - u[i] + I[i]) / C

        if v[i + 1] > -65:
            b = 0
        else:
            b = 15
        u[i + 1] = u[i] + tau * a * (b * (v[i] - vr) - u[i])

        if v[i + 1] > (vPeak + 0.1 * u[i + 1]):
            v[i] = vPeak + 0.1 * u[i + 1]
            v[i + 1] = c - 0.1 * u[i + 1]
            # Reset voltage
            u[i + 1] = u[i + 1] + d

    return v


@jit(nopython=True)
def get_vm_seven(
    C=89.7960714285714,
    a=0.01,
    b=15,
    c=-60,
    d=10,
    k=1.6,
    vPeak=(86.364525297619 - 65.2261863636364),
    vr=-65.2261863636364,
    vt=-50,
    I=[],
    dt = 0.25
):
    tau = dt
    # dt
    N = len(I)
    v = vr * np.ones(N)
    u = np.zeros(N)
    v[0] = vr
    for i in range(N):

        # forward Euler method
        v[i + 1] = v[i] + tau * (k * (v[i] - vr) * (v[i] - vt) - u[i] + I[i]) / C

        if v[i + 1] > -65:
            b = 2
        else:
            b = 10

        u[i + 1] = u[i] + tau * a * (b * (v[i] - vr) - u[i])
        if v[i + 1] >= vPeak:
            v[i] = vPeak
            v[i + 1] = c
            u[i + 1] = u[i + 1] + d
            # reset u, except for FS cells

    return v
@jit(nopython=True)
def get_2003_vm(I, times, a=0.01, b=15, c=-60, d=10, vr=-70):
    u = b * vr
    V = vr
    tau = dt = 0.25
    N = len(I)
    vv = np.zeros(N)
    UU = np.zeros(N)

    for i in range(N):
        V = V + tau * (0.04 * V ** 2 + 5 * V + 140 - u + I[i])
        u = u + tau * a * (b * V - u)
        if V > 30:
            vv[i] = 30
            V = c
            u = u + d
        else:
            vv[i] = V
        UU[i] = u
    return vv
