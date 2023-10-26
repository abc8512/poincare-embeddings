import geoopt
import torch
import numpy as np
from scipy.special import logsumexp


class ManifoldGMM():
    def __init__(self, manifold: geoopt.Manifold) -> None:
        self.manifold = manifold
        if manifold.name == 'PoincareBall':
            self.compute_mean = manifold.weighted_midpoint
        elif manifold.name == 'Lorentz':
            self.compute_mean = self.compute_weighted_frechet_mean
        else:
            NotImplementedError("Check your manifold. The manifold is not supported")
        pass

    def multivariate_normal(x, mu, sigma=None, log=True):

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
    
    def compute_weighted_frechet_mean(self, data, weights, nb_iter_max: int=10, convergence_threshold: float=1e-6):
        nb_data = data.shape[0]
        nb_dim = data.shape[1]

        mean = data[0]
        mean = torch.Tensor(mean)
        weights = torch.Tensor(weights)
        data = torch.Tensor(data)
        distance_previous_mean = torch.inf
        data_on_tangent_space = torch.zeros_like(data)
        nb_iter = 0

        while distance_previous_mean > convergence_threshold and nb_iter < nb_iter_max:
            previous_mean = mean
            
            # Compute the weighted mean of the data in the tangent space of the previous mean
            for i in range(nb_data):
                data_on_tangent_space[i] = self.manifold.logmap(previous_mean, data[i])

            weighted_data_on_tangent_space = weights[:, None] * data_on_tangent_space
            weighted_mean_on_tangent_space = torch.sum(weighted_data_on_tangent_space, dim=0)

            mean = self.manifold.expmap(previous_mean, weighted_mean_on_tangent_space)
            # mean = manifold.weighted_midpoint(data, weights=weights)

            # Check convergence
            distance_previous_mean = torch.linalg.norm(self.manifold.logmap(mean.reshape(1, nb_dim), previous_mean.reshape(1, nb_dim)))
            nb_iter += 1

        if distance_previous_mean > convergence_threshold:
            print("frechet mean is not converged. distance: ", distance_previous_mean, ", nb_iter: ", nb_iter)
        # else:
        #     print("frechet mean converged: ", nb_iter)

        return mean
    
    def manifold_gmm_em(self, data, nb_states, initial_means=None, initial_covariances=None, initial_priors=None,
                        nb_iter_max=100, max_diff_ll=1e-5, regularization_factor=1e-10):
        
        nb_min_steps = 5
        nb_data = data.shape[0]
        nb_dim = data.shape[1]
        data = torch.Tensor(data)
        means = torch.Tensor(initial_means)
        covariances = torch.Tensor(initial_covariances)
        priors = torch.Tensor(initial_priors)
        LL = torch.zeros(nb_iter_max)

        xts = torch.zeros((nb_states, nb_data, nb_dim))
        for k in range(nb_states):
            xts[k, :, :] = self.manifold.logmap(means[k], data)

        for it in range(nb_iter_max):
            # E-step
            # L = torch.zeros((nb_data, nb_states))
            L_log = torch.zeros((nb_data, nb_states))

            for k in range(nb_states):
                L_log[:, k] = torch.log(priors[k]) + ManifoldGMM.multivariate_normal(x=xts[k], mu=torch.zeros_like(means[k]),
                                                                                     sigma=covariances[k], log=True)
            # compute responsibilities r_nk
            responsibilities = torch.exp(L_log - logsumexp(L_log, axis=1)[:, None])
            # compute r_nk / N_k
            weighted_responsibilities = responsibilities / (torch.sum(responsibilities, dim=0))

            # M-step
            for k in range(nb_states):
                # update means
                # means[k] = manifold.weighted_midpoint(data, weighted_responsibilities[:, k])
                means[k] = self.compute_mean(data, weighted_responsibilities[:, k])

                xts[k, :, :] = self.manifold.logmap(means[k], data)

                # update covariances
                # covariances[k] = torch.dot(xts[k].T, torch.dot(torch.diag(weighted_responsibilities[:, k]), xts[k])) + regularization_factor * torch.eye(nb_dim)
                covariances[k] = xts[k].T @ torch.diag(weighted_responsibilities[:, k]) @ xts[k] + regularization_factor * torch.eye(nb_dim)

            # update priors
            priors = torch.mean(responsibilities, dim=0)

            # log-likelihood
            LL[it] = torch.mean(torch.log(torch.sum(torch.exp(L_log), dim=1)))

            # Check for convergence
            if it > nb_min_steps:
                if LL[it] - LL[it - 1] < max_diff_ll:
                    print('The EM algorithm converged after {%d} iterations.' %it)

                    return means, covariances, priors, responsibilities
                
        print("GMM did not converge before reaching the maximum number of iterations.")

        return means, covariances, priors, responsibilities




