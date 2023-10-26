import numpy as np
from hype import MANIFOLDS
import torch as th
from scipy.special import logsumexp

class ManifoldGMM():
    def __init__(self) -> None:
        pass

    def multivariate_normal(x, mu, sigma=None, log=True):
        """Multivariate normal distribution

        Args:
            x (array): data                                                     (nb_samples x nb_dim)
            mu (array): mean                                                    (nb_dim)
            sigma (array, optional): covariance matrix. Defaults to None.       (nb_dim x nb_dim)
            log (bool, optional): _description_. Defaults to True.
        """

        dx = x - mu
        if sigma.ndim == 1:
            sigma = sigma[:, None]
            dx = dx[:, None]
            inv_sigma = np.linalg.inv(sigma)
            log_lik = -0.5 * np.sum(np.dot(dx, inv_sigma)*dx, axis=1) - 0.5 * np.log(np.linalg.det(2 * np.pi * sigma))
        else:
            inv_sigma = np.linalg.inv(sigma)
            _, logdet = np.linalg.slogdet(2 * np.pi * sigma)    # sign must be 1
            log_lik = -0.5 * np.einsum('...j,...j', dx, np.einsum('...jk,...j->...k', inv_sigma, dx)) - 0.5 * logdet

        return log_lik if log else np.exp(log_lik)
    
    
    def compute_weighted_frechet_mean(manifold: MANIFOLDS, data, weights, nb_iter_max: int=10, convergence_threshold: float=1e-6) -> np.array:
        """_summary_

        Args:
            manifold (MANIFOLDS): Riemannian manifold
            data (_type_): data points                                                  (nb_data x nb_dim)
            weights (_type_): _description_                                             (nb_data)
            nb_iter_max (int, optional): _description_. Defaults to 10.
            convergence_threshold (float, optional): _description_. Defaults to 1e-6.

        Returns:
            np.array: weighted frechet mean
        """
        nb_data = data.shape[0]
        nb_dim = data.shape[1]

        mean = data[0]
        mean = th.Tensor(mean)
        weights = th.Tensor(weights)
        data = th.Tensor(data)
        # data_on_tangent_space = np.zeros_like(data)
        distance_previous_mean = np.inf
        data_on_tangent_space = th.zeros_like(data)
        nb_iter = 0

        while distance_previous_mean > convergence_threshold and nb_iter < nb_iter_max:
            previous_mean = mean

            # Compute the weighted mean of the data in the tangent space of the previous mean
            for i in range(nb_data):
                data_on_tangent_space[i] = manifold.logm(previous_mean.reshape(1, nb_dim), data[i].reshape(1, nb_dim))

            weighted_data_on_tangent_space = weights[:, None] * data_on_tangent_space
            # weighted_mean_on_tangent_space = np.sum(weighted_data_on_tangent_space, axis=0)
            weighted_mean_on_tangent_space = th.sum(weighted_data_on_tangent_space, dim=0)

            mean = manifold.expm(previous_mean, weighted_mean_on_tangent_space)

            # Check convergence
            distance_previous_mean = th.linalg.norm(manifold.logm(mean.reshape(1, nb_dim), previous_mean.reshape(1, nb_dim)))
            nb_iter += 1
        
        if distance_previous_mean > convergence_threshold:
            print("frechet mean is not converged.")


        return mean.detach().cpu().numpy()
    

    def manifold_gmm_em(manifold: MANIFOLDS, data, nb_states, initial_means=None, initial_covariances=None, initial_priors=None,
                        nb_iter_max=100, max_diff_ll=1e-5, regularization_factor=1e-10):
        """EM algorithm for a GMM on a Riemannian manifold

        Args:
            manifold (MANIFOLDS): Riemannian manifold on which the outputs are living
            data (array): data points                                                           (nb_data x data_dimensions)
            nb_states (_type_): number of clusters                                              
            initial_means (array, optional): initial GMM means. Defaults to None.               (nb_states x data_dimension)
            initial_covariance (array, optional): initial GMM covariance. Defaults to None.     (nb_states x data_dimension x data_dimension)
            initial_priors (array, optional): initial GMM priors. Defaults to None.             (nb_states)
            nb_iter_max (int, optional): max number of iterations for EM. Defaults to 100.
            max_diff_ll (_type_, optional): threshold of log likelihood change for convergence. Defaults to 1e-5.
            regularization_factor (_type_, optional): regularization for the covariance matrix. Defaults to 1e-10.
        """
        
        nb_min_steps = 5    # min number of iterations

        nb_data = data.shape[0]
        nb_dim = data.shape[1]
        means = np.copy(initial_means)
        covariances = np.copy(initial_covariances)
        priors = np.copy(initial_priors)

        data = th.Tensor(data)
        means = th.from_numpy(means)
        covariances = th.from_numpy(covariances)
        priors = th.from_numpy(initial_priors)
        LL = th.zeros(nb_iter_max)

        xts = th.zeros((nb_states, nb_data, nb_dim))
        for k in range(nb_states):
            for n in range(nb_data):
                xts[k, n, :] = manifold.logm(means[k].reshape(1, nb_dim), data[n].reshape(1, nb_dim))

        for it in range(nb_iter_max):
            # E-step
            L = th.zeros((nb_data, nb_states))
            L_log = th.zeros((nb_data, nb_states))

            for k in range(nb_states):
                L_log[:, k] = th.log(priors[k]) + ManifoldGMM.multivariate_normal(x=xts[k], mu=th.zeros_like(means[k]),
                                                                                  sigma=covariances[k], log=True)
                
            L = th.exp(L_log)

            # compute responsibilities r_nk
            # responsibilities = L / (th.sum(L, axis=1)[:, None] + regularization_factor)
            responsibilities = th.exp(L_log - logsumexp(L_log, axis=1)[:, None])
            # compute r_nk / N_k
            weighted_responsibilities = responsibilities / (th.sum(responsibilities, axis=0) + 1e-10)
            
            # M-step
            for k in range(nb_states):
                # update means
                means[k] = ManifoldGMM.compute_weighted_frechet_mean(manifold, data, weighted_responsibilities[:, k])

                for n in range(nb_data):
                    xts[k, n, :] = manifold.logm(means[k].reshape(1, nb_dim), data[n].reshape(1, nb_dim))

                # update covariance
                covariances[k] = th.dot(xts[k].T, th.dot(th.diag(weighted_responsibilities[:, k]), xts[k])) + regularization_factor * th.eye(nb_dim)

            # update priors
            priors = th.mean(responsibilities, axis=0)

            # Log-likelihood
            LL[it] = th.mean(th.log(th.sum(L, axis=1)))

            # Check for convergence
            if it > nb_min_steps:
                if LL[it] - LL[it - 1] < max_diff_ll:
                    print('The EM algorithm converged after {%d} iterations.' %it)

                    return means.detach().cpu().numpy(), covariances.detach().cpu().numpy(), priors.detach().cpu().numpy(), responsibilities.detach().cpu().numpy()
                
        print("GMM did not converge before reaching the maximum number of iterations.")
        return means.detach().cpu().numpy(), covariances.detach().cpu().numpy(), priors.detach().cpu().numpy(), responsibilities.detach().cpu().numpy()
    
    def compute_gmm_density(manifold: MANIFOLDS, data, nb_states, means, covariances, priors, log=False) -> np.array:
        """_summary_

        Args:
            manifold (MANIFOLDS): _description_
            data (_type_): _description_                                        (nb_data x data_dimensions)
            nb_states (int): _description_
            means (_type_): _description_                                       (nb_states x data_dimension)
            covariances (_type_): _description_                                 (nb_states x data_dimension x data_dimension)
            priors (_type_): _description_                                      (nb_states)
            log (bool, optional): _description_. Defaults to False.

        Returns:
            np.array: likelihood or log-likelihood                              (nb_data)
        """
        nb_data = data.shape[0]
        nb_dim = data.shape[1]

        xts = np.zeros((nb_states, nb_data, nb_dim))
        for k in range(nb_states):
            for n in range(nb_data):
                xts[k, n, :] = manifold.logm(means[k], data[n])

        states_likelihood = np.zeros((nb_data, nb_states))
        for k in range(nb_states):
            states_likelihood[:, k] = priors[k] + ManifoldGMM.multivariate_normal(xts[k], np.zeros_like(means[k]), covariances[k], log=False)

        likelihood = np.sum(states_likelihood, axis=1)

        if log:
            return np.log(likelihood)
        else:
            return likelihood
        
        

                