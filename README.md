# MultivariateNormalCRP

This non-parametric generative model performs something akin to kernel density estimation (KDE) in a space of arbitrary dimension. It models a cloud of sample points using a mixture of multivariate normal distributions with an arbitrary number of components.

It differs from traditional KDE in that it is a clustering algorithm where sample points are assigned to mixture components and each mixture component has its own covariance matrix which can be thought of as a generalization of the bandwidth in KDE.

 One salient feature of the model is that, as mentionned above, the number of mixture components is arbitrary and itself a distribution infered from the data and so is the covariance matrix for each of those components independently.

The model uses a non-parametric Chinese Restaurant Process (CRP) prior (Aldous 1983) over the number of mixture components and assignments of sample points to those components. For the base distribution of the CRP we use the Normal-Inverse-Wishart (NIW) distribution given that it is the conjugate prior to our data likelihood a multivariate normal distribution with unknown mean and covariance. Conjugacy is desirable because it greatly simplifies the accompanying algorithm.

The generative model is sampled using MCMC. Our MCMC algorithm uses three kinds of sampling strategies. For sampling the CRP we use traditional Gibbs sampling (Neal 2000, algorithm 3) together with restricted split-merge moves (Jain & Neal 2004). Hyperparameters of the model for both the CRP and the NIW base distributions have their own hyperpriors, some improper and some proper, which we sample using traditional Metropolis-Hastings moves.

Of note is the method we use to sample the precision matrix hyperparameter Psi of the NIW with a flat hyperprior. Since the precision matrix is positive definite, we use the Cholesky decomposition Psi = L * L' and perform uniform symmetric moves on the diagonal and lower triangular elements of L. To transform those symmetric moves over L into uniform moves over Psi, we use the absolute value of the determinant of the Jacobian

$$\left\vert\frac{\partial\Psi}{\partial L}\right\vert = 2^d \vert L_{11}\vert^d \vert L_{22}\vert^{d - 1}\ldots \vert L_{dd}\vert$$

to construct the Hastings factor. In other words this gives us a scheme to uniformly explore the space of positive-definite matrices. As far as we are aware there is no reference to this approach in the litterature.
