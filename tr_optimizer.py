
# Author @ Ioannis Tsingalis

import numpy as np
import warnings
import time
from numpy.linalg import cholesky, LinAlgError  # use NumPy version
from scipy.optimize import lsq_linear, minimize_scalar


# ======================
# Trust-Region machinery
# ======================
def Hfc(fc: float, T: float, N: int) -> np.ndarray:
    k = np.arange(N, dtype=float)
    alpha = np.cos(2 * np.pi * T * fc * k)
    beta = np.sin(2 * np.pi * T * fc * k)
    return np.vstack([alpha, beta]).T


def dH_df(fc: float, T: float, N: int) -> np.ndarray:
    k = np.arange(N, dtype=float)
    w = 2 * np.pi * T
    dalpha = -w * k * np.sin(w * fc * k)
    dbeta = w * k * np.cos(w * fc * k)
    return np.vstack([dalpha, dbeta]).T


def d2H_d2f_x(fc: float, x: np.ndarray, T: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    N = x.size
    k = np.arange(N, dtype=float)
    w = 2 * np.pi * T
    wk = w * fc * k
    s1 = np.sum(k ** 2 * np.cos(wk) * x)
    s2 = np.sum(k ** 2 * np.sin(wk) * x)
    return -(w ** 2) * np.array([s1, -s2], dtype=float)


def model_value(step_s: np.ndarray, B: np.ndarray, g: np.ndarray, J0: float) -> float:
    return float(J0 + g @ step_s + 0.5 * (step_s @ (B @ step_s)))


def build_problem(x: np.ndarray, T: float):
    x = np.asarray(x, dtype=float)
    N = x.size

    def J(mu: np.ndarray) -> float:
        fc, t1, t2 = mu
        theta = np.array([t1, t2], dtype=float)
        H = Hfc(fc, T, N)
        r = x - H @ theta
        return float(r @ r)

    def gradJ(mu: np.ndarray) -> np.ndarray:
        fc, t1, t2 = mu
        theta = np.array([t1, t2], dtype=float)
        H  = Hfc(fc, T, N)
        dH = dH_df(fc, T, N)

        grad_fc    = -2.0 * float(x @ (dH @ theta))
        grad_theta = -2.0 * (H.T @ x) + 2.0 * theta
        return np.hstack([grad_fc, grad_theta]).astype(float)

    def hessJ(mu: np.ndarray) -> np.ndarray:
        fc, t1, t2 = mu
        theta = np.array([t1, t2], dtype=float)

        a = float(theta @ d2H_d2f_x(fc, x, T))  # scalar
        b = dH_df(fc, T, N).T @ x              # shape (2,)

        HESS = np.zeros((3, 3), dtype=float)
        HESS[0, 0]   = -2.0 * a
        HESS[0, 1:]  = -2.0 * b
        HESS[1:, 0]  = -2.0 * b
        HESS[1:, 1:] =  2.0 * np.eye(2)
        return HESS

    return J, gradJ, hessJ


def solve_trust_region_step(B_lam, g, mu_curr, tol=1e-12):
    A = np.asarray(B_lam, float)
    b = np.asarray(-g, float)
    lower = -np.asarray(mu_curr, float)
    upper = np.full(len(g), np.inf, dtype=float)
    try:
        res = lsq_linear(A, b, bounds=(lower, upper), method="trf")
        s = np.asarray(res.x, float)
        s = np.maximum(s, lower - 1e-12)
        return s
    except Exception as e:
        warnings.warn(f"lsq_linear failed ({e}); using diagonal fallback.")
        diag = np.diag(A).copy()
        diag[np.abs(diag) < 1e-12] = 1e-12
        s = -b / diag
        s = np.maximum(s, lower)
        return s


def solve_alpha(s, u3, Delta_sq, F, mu_curr, tol=1e-12):
    L, U = -np.inf, np.inf
    base = mu_curr + s
    for i in range(len(base)):
        ui = u3[i]
        if abs(ui) < tol:
            if base[i] < -tol: return 0.0
            continue
        bound = -base[i] / ui
        if ui > 0:
            L = max(L, bound)
        else:
            U = min(U, bound)
    if L > U: return 0.0

    a = float(u3 @ u3)
    b = 2.0 * float(s @ u3)
    c = float(s @ s) - float(Delta_sq)
    disc = b * b - 4 * a * c
    if disc < 0:
        if (s @ s) <= Delta_sq:
            L_tr, U_tr = -np.inf, np.inf
        else:
            return 0.0
    else:
        sqrt_disc = np.sqrt(disc)
        r1 = (-b - sqrt_disc) / (2 * a)
        r2 = (-b + sqrt_disc) / (2 * a)
        L_tr, U_tr = (r1, r2) if r1 <= r2 else (r2, r1)

    L = max(L, L_tr);
    U = min(U, U_tr)
    if L > U: return 0.0

    try:
        res = minimize_scalar(lambda alpha: F(s + alpha * u3), bounds=(L, U), method="bounded")
        return float(res.x)
    except Exception:
        return float((L + U) / 2.0)


def model_minimizer(B, g, Delta_sq, kappa_easy, F, mu_curr, max_iterations=100):
    try:
        eigvals, eigvecs = np.linalg.eigh(B)
    except LinAlgError:
        B = B + 1e-8 * np.eye(B.shape[0])
        eigvals, eigvecs = np.linalg.eigh(B)

    idx_min = int(np.argmin(eigvals))
    lam_min = float(eigvals[idx_min])
    u3 = eigvecs[:, idx_min]
    lam = 0.0 if np.all(eigvals > 0.0) else (-lam_min + 1e-10)

    for _ in range(max_iterations):
        B_lam = B + lam * np.eye(B.shape[0])
        s = solve_trust_region_step(B_lam, g, mu_curr)
        norm_sq = float(s @ s)

        if norm_sq <= Delta_sq * (1 + 1e-12):
            if np.all(eigvals > 0.0) or abs(norm_sq - Delta_sq) <= kappa_easy * Delta_sq:
                return s
            alpha = solve_alpha(s, u3, Delta_sq, lambda z: model_value(z, B, g, 0.0), mu_curr)
            s_new = s + alpha * u3
            if np.all(mu_curr + s_new >= -1e-10): return s_new
            return np.maximum(s_new, -mu_curr + 1e-12)

        try:
            L = cholesky(B_lam)  # ✅ NumPy version
            w = np.linalg.solve(L, s)
            denom = Delta_sq * float(w @ w)
            if abs(denom) > 1e-18:
                lam += (norm_sq * (norm_sq - Delta_sq)) / denom
                lam = max(lam, 1e-12)
            else:
                lam = max(lam * 1.5, 1e-12)
        except LinAlgError:
            lam = max(lam * 2.0, 1e-10)

        if abs(norm_sq - Delta_sq) <= kappa_easy * Delta_sq:
            return s

    warnings.warn("Model minimizer reached max iterations; returning last step.")
    return s

def trust_region(J, gradJ, hessJ, mu0, max_iters=100, tol_grad=1e-6, tol_step=1e-8):
    Delta_sq = 1.0
    kappa_easy = 0.1
    alpha1, alpha2 = 4.5, 0.25
    eta1, eta2 = 0.05, 0.75
    eps_m = 1e-6

    mu_k = np.asarray(mu0, dtype=float).copy()
    t_start = time.time()
    iters, accepted, rejected = 0, 0, 0
    grad_norms, step_norms, J_values, rhos = [], [], [], []

    for _ in range(max_iters):
        try:
            B = hessJ(mu_k)
            g = gradJ(mu_k)
            J_current = J(mu_k)

            iters += 1
            J_values.append(J_current)
            ng = float(np.linalg.norm(g));
            grad_norms.append(ng)
            if ng < tol_grad: break

            F = lambda s: model_value(s, B, g, J_current)
            s = model_minimizer(B, g, Delta_sq, kappa_easy, F, mu_k)

            mu_new = mu_k + s
            if np.any(mu_new < -1e-12):
                s = np.maximum(s, -mu_k + 1e-12)
                mu_new = mu_k + s

            J_new = J(mu_new)
            ared = J_current - J_new
            pred = J_current - F(s)
            rho_k = (ared / pred) if abs(pred) > 1e-15 else 0.0
            rhos.append(float(rho_k))

            if rho_k >= eta1:
                mu_k, accepted = mu_new, accepted + 1
            else:
                rejected += 1

            sn = float(np.linalg.norm(s))
            step_norms.append(sn)
            if rho_k >= eta2:
                Delta_sq = max(alpha1 * sn * sn, Delta_sq)
            elif rho_k < eta1:
                Delta_sq = max(alpha2 * sn * sn, eps_m)

            if sn < tol_step: break
        except Exception as e:
            warnings.warn(f"Trust-region error: {e}")
            break

    info = {
        "iters": iters, "accepted": accepted, "rejected": rejected,
        "grad_norm_init": grad_norms[0] if grad_norms else np.nan,
        "grad_norm_final": grad_norms[-1] if grad_norms else np.nan,
        "step_norm_final": step_norms[-1] if step_norms else np.nan,
        "J_init": J_values[0] if J_values else np.nan,
        "J_final": J_values[-1] if J_values else np.nan,
        "J_decrease": (J_values[0] - J_values[-1]) if len(J_values) >= 2 else np.nan,
        "rho_mean": float(np.nanmean(rhos)) if rhos else np.nan,
        "rho_std": float(np.nanstd(rhos, ddof=1)) if len(rhos) > 1 else 0.0,
        "time_sec": time.time() - t_start,
    }
    return mu_k, info