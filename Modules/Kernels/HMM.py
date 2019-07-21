# Mathematical packages
import numpy as np


def lnp_u_conditionning_to_q(u_t, q_t, pi_u):
    """Return the value of p(u_t|q_t) for the given u_t and the given q_t."""

    # Extract the value for u_t
    ln_pi_u_t = np.log(pi_u[q_t, u_t])

    return ln_pi_u_t


def lnalpha_t_plus_1(lnalpha_t, u_t_plus_1, a, pi_u, K=2):
    """Compute alpha_t_plus_1."""

    # Initialize alpha_t_plus_1
    lnalpha_t_plus_1 = np.ones(K)

    # Compute each state of alpha_t_plus_1
    for q_t_plus_1 in range(K):

        # Compute p(u_t+1 | q_t_+1)
        lnp_u_cond_q = lnp_u_conditionning_to_q(u_t_plus_1, q_t_plus_1,
                                                pi_u)

        # Update alpha_t_plus_1
        vec_proba = np.log(a[:, q_t_plus_1]) + lnalpha_t
        maxi = np.max(vec_proba)

        argexp = lnp_u_cond_q + \
                 np.log(np.sum(np.exp(vec_proba - maxi),
                                      dtype=np.longdouble),
                        dtype=np.longdouble) + maxi
        lnalpha_t_plus_1[q_t_plus_1] = argexp

    return lnalpha_t_plus_1


def lnbeta_t(lnbeta_t_plus_1, u_t_plus_1, a, pi_u, K=2):
    """Compute beta_t_plus_1."""

    # Initialize beta_t
    lnbeta_t = np.ones(K)

    # Compute each state of beta_t
    for q_t in range(K):

        vec_proba = np.zeros(K)

        for q_t_plus_1 in range(K):

            # Compute the log of the product of the probabilities
            vec_proba[q_t_plus_1] = lnp_u_conditionning_to_q(u_t_plus_1,
                                                             q_t_plus_1, pi_u)
            vec_proba[q_t_plus_1] += np.log(a[q_t, q_t_plus_1])
            vec_proba[q_t_plus_1] += lnbeta_t_plus_1[q_t_plus_1]

        # Update alpha_t_plus_1
        maxi = np.max(vec_proba)
        argexp = np.log(np.sum(np.exp(vec_proba - maxi,
                                      dtype=np.longdouble)),
                        dtype=np.longdouble) + maxi
        lnbeta_t[q_t] = argexp

    return lnbeta_t


def estimation(U, pi, a, pi_u, K=2):
    """Compute alpha_t and beta_t."""

    # Parameters
    N, T, d = np.shape(U)

    # Initialise alpha and beta
    lnalpha = np.ones((N, T, K))
    lnbeta = np.ones((N, T, K))

    for j in range(N):

        # Initialisation of alpha_0 and beta_T
        for q_0_k in range(K):
            lnalpha[j, 0, q_0_k] = np.log(pi[q_0_k])  # + lnp_u_conditionning_to_q(U[0,:], q_0_k, mu, cov)
        lnbeta[j, T - 1, :] = np.zeros(K)

        # Computation of alpha
        for t in range(1, T):

            # Update of alpha
            lnalpha[j, t, :] = lnalpha_t_plus_1(lnalpha[j, t - 1, :],
                                                U[j, t, :], a, pi_u, K=K)

        # Computation of beta
        for i in range(2, T + 1):

            # t
            t = T - i

            # Update of beta
            lnbeta[j, t, :] = lnbeta_t(lnbeta[j, t + 1, :], U[j, t + 1, :],
                                       a, pi_u, K=K)

    return lnalpha, lnbeta


def compute_p_q_u(lnalpha, lnbeta):
    """Return log(p(q_t, u_0, ..., u_T))."""

    return lnalpha + lnbeta


def compute_p_u(lnalpha, lnbeta):
    """Return log(p(u_0, ..., u_T))."""

    # Parameters
    N, T, K = np.shape(lnalpha)

    # Initialisation of result
    res = np.zeros(N)

    for i in range(N):
        # Extract max
        arg = (lnalpha + lnbeta)[i, 0, :]
        maxi = np.max(arg)

        # Compute the result
        res[i] = np.log(np.sum(np.exp(arg - maxi))) + maxi

    return res.reshape((-1, 1, 1))


def compute_q(lnalpha, lnbeta):
    """Return log(p(q_t| u_0, ..., u_T))."""

    # Parameters
    N, T, K = np.shape(lnalpha)

    # Compute p(z_t, y_0, ..., y_T)
    lnp_q_u = compute_p_q_u(lnalpha, lnbeta)

    # Compute p(y_0, ..., y_T)
    lnp_u = compute_p_u(lnalpha, lnbeta)

    return lnp_q_u - np.tile(lnp_u, (1, T, K))


def compute_q_q(lnalpha, lnbeta, U, a, pi_u):
    """Return log(p(q_t, q_t+1 | u_0, ..., u_T))."""

    # Parameters
    N, T, K = np.shape(lnalpha)

    # Compute p(y_0, ..., y_T)
    lnp_u = compute_p_u(lnalpha, lnbeta)

    # Initialisation of p(z_t, z_t+1 |y)
    lnp_q_q = np.ones((N, T-1, K, K))

    # Compute each value
    for i in range(N):

        for t in range(T-1):

            for k in range(K):
                for l in range(K):

                    # p_u_cond_q
                    lnp_u_cond_q = lnp_u_conditionning_to_q(U[i, t+1, :], l, pi_u)

                    # Update p_q_q
                    lnp_q_q[i, t, k, l] = -lnp_u[i, :, :] + lnalpha[i, t, k] +\
                                          lnbeta[i, t+1, l] +\
                                          np.log(a[k, l]) + lnp_u_cond_q

    return lnp_q_q


def Mstep(U, pi, a, pi_u, p_q, p_q_q):
    """This function executes the M step of the EM algorithm."""

    # Parameters
    K = len(pi)
    N, T, d = np.shape(U)

    # Estimator of pi_u
    _, K_u = np.shape(pi_u)
    new_pi_u = np.zeros((K, K_u))
    divisor = np.zeros(K)

    for i in range(N):
        for k in range(K):
            for t in range(T):

                # Compute p(u_t)
                p_u_t = np.dot(p_q[i, t, :].reshape((1, -1)), pi_u)

                # Update new_pi_u
                new_pi_u[k, :] += p_q[i, t, k] * p_u_t.reshape(-1)

                # Update divisor
                divisor[k] += p_q[i, t, k]

    for k in range(K):
        new_pi_u[k, :] /= divisor[k]

    # Renormalisation
    new_pi_u = new_pi_u / np.sum(new_pi_u, axis=1).reshape((-1, 1))

    # Estimator of a_k
    a = np.zeros((K, K))
    divisor = np.zeros(K)

    for i in range(N):
        for k in range(K):
            for l in range(K):

                a[k, l] += np.sum(p_q_q[i, :, k, l])

            divisor[k] += np.sum(p_q_q[i, :, k, :])

    for k in range(K):
        a[k, :] = a[k, :] / divisor[k]

    # Estimator of pi_k
    pi = np.mean(p_q[:, 0, :], axis=0)

    # Return results
    return pi, a, new_pi_u


def objectif(U, pi, a, pi_u, p_q, p_q_q):
    """This function computes the complete-log-likelihood for the given parameters."""

    # Parameters
    K = len(pi)
    N, T, d = np.shape(U)

    # Initialisation of the objectif
    objectif = 0

    # Loop over each data point
    for i in range(N):

        # Add the zero term
        objectif += np.sum(pi * np.log(p_q[i, 0, :]))

        for t in range(T):
            for k in range(K):

                # Add the term with a
                if t > 0:
                    objectif += np.sum(a[k, :] * np.log(p_q_q[i, t-1, k, :]))

                # Add the term of the gaussian
                p_u_t = np.dot(p_q[i, t, :].reshape(-1), pi_u[:, U[i, t, 0]].reshape(-1))
                objectif += p_q[i, t, k] * p_u_t * np.log(pi_u[k, U[i, t, 0]])

    return objectif


def EM(U, pi_u, K=4):
    """This function executes the EM algorithm."""

    # Parameters
    N, T, d = np.shape(U)

    # Initialisation of a
    a = np.random.rand(K, K)  # np.zeros((K, K)) + 1
    for k in range(K):
        a[k, :] /= np.sum(a[k, :])

    # Initialisation of the parameters for the E step
    pi = np.random.rand(K)  # np.ones(K)
    pi /= np.sum(pi)

    # Initialisation of the latent probabilities
    p_q_q = np.random.rand(N, T, K, K)
    for t in range(T):
        p_q_q[:, t, :, :] /= np.sum(p_q_q[:, t, :, :])

    p_q = np.sum(p_q_q, axis=3)

    # Initialisation of the objectif
    objectif_new = objectif(U, pi, a, pi_u, p_q, p_q_q)
    objectif_old = 2 * objectif_new

    # Count the number of iteration
    ite = 0

    while ite < 3000 or abs((objectif_old - objectif_new) / objectif_old) > 10e-6: #10

        # E step
        lnalpha, lnbeta = estimation(U, pi, a, pi_u, K=K)
        p_q = compute_q(lnalpha, lnbeta)
        p_q = np.exp(p_q)
        p_q_q = compute_q_q(lnalpha, lnbeta, U, a, pi_u)
        p_q_q = np.exp(p_q_q)

        # M step
        pi, a, pi_u = Mstep(U, pi, a, pi_u, p_q, p_q_q)

        # Update of the objectif
        objectif_old = objectif_new
        objectif_new = objectif(U, pi, a, pi_u, p_q, p_q_q)

        # Increase the counter
        ite += 1
        print("Value of the objective: ", objectif_new)

    # Display the numbner of iterations done
    # print("Number of iterations done:", ite)

    return pi, a, pi_u, p_q, p_q_q
