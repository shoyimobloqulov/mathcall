# tasks.py
from celery import shared_task
import numpy as np
from scipy.special import gamma

@shared_task
def perform_calculation(C0, vm, Dm, n, tau, h, tmax, w, tetim, tetm, gam, alpha, bet):
    A = 1 + (w * gamma(2 - alpha) * tau**alpha) / (gam * tetim)

    # Boshlang`ich shartlar
    Cm = np.zeros((tmax + 1, n + 1))
    Cim = np.zeros((tmax + 1, n + 1))

    # Chegaraviy shartlar
    for k in range(tmax + 1):
        Cm[k, 0] = C0

    # Asosiy qism
    for k in range(tmax):
        for i in range(n + 1):
            s1 = 0
            for l1 in range(k):
                if k > 0:
                    s1 += (Cim[l1 + 1, i] - Cim[l1, i]) * ((k - l1 + 1)**(1 - alpha) - (k - l1)**(1 - alpha))
            Cim[k + 1, i] = (Cim[k, i] - s1 + (gamma(2 - alpha) * tau**alpha * w) * Cm[k, i] / (gam * tetim)) / A

        for i in range(1, n):
            if bet == 2:
                s01 = Cm[k, i + 1] - 2 * Cm[k, i] + Cm[k, i - 1]
            else:
                s01 = 0
                for l in range(i):
                    s01 += ((l + 1)**(2 - bet) - (l)**(2 - bet)) * (Cm[k, i + 1 - l] - 2 * Cm[k, i - l] + Cm[k, i - 1 - l])
            Cm[k + 1, i] = (tau * Dm * s01) / (gamma(3 - bet) * (h**bet)) - tau * vm * (Cm[k, i] - Cm[k, i - 1]) / h - gam * tau * tetim * s1 / (tetm * gamma(2 - alpha) * tau**alpha) * (Cim[k + 1, i] - Cim[k, i]) + Cm[k, i]
        Cm[k + 1, n] = Cm[k + 1, n - 1]

    return Cm, Cim
