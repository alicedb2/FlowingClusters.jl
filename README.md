# MultivariateNormalCRP

This non-parametric generative model performs density estimation in arbitrary dimension using mixtures of Dirichlet processes (Escobar & West 1994, MacEarchern 1994). It models a cloud of sample points using a mixture of multivariate normal distributions with an arbitrary number of components.

One can think of this approach as as alternative to kernel density estimation (KDE).It differs from KDE in that it is a clustering algorithm where sample points are assigned to mixture components and each mixture component has its own covariance matrix. This latter property can be thought of as a more flexible generalization of the bandwidth in KDE.

One salient feature of the model is that, as mentionned above, the number of mixture components is arbitrary and itself a distribution infered from the data and so is the covariance matrix of each component.

The model uses a non-parametric Chinese Restaurant Process (CRP) prior (Aldous 1983) over partitions of the sample points, i.e. over both the number of mixture components and over the assignments of sample points to those components. For the base distribution of the CRP we use the Normal-Inverse-Wishart (NIW) distribution given that it is the conjugate prior to our data likelihood the multivariate normal distribution with unknown mean and covariance. Conjugacy is desirable because it greatly simplifies the accompanying algorithm.

The generative model is sampled using MCMC. Our MCMC algorithm uses three kinds of sampling strategies. For sampling the CRP we use both traditional Gibbs sampling (Neal 2000, algorithm 3) and the more ambitious restricted split-merge moves (Jain & Neal 2004). Hyperparameters of the model for both the CRP and the NIW base distributions have their own hyperpriors, some improper and some proper, which we sample using traditional Metropolis-Hastings moves.

Of note is the method we use to sample the precision matrix hyperparameter Psi of the NIW over a flat hyperprior. Indeed it is not immediately obvious what a flat prior means and looks like over the space of positive definite matrices. To do so we use the Cholesky decomposition Psi = L * L<sup>T</sup>. Since the entries of the matrix L are real numbers we can use symmetric moves in each of its lower triangular elements. To translate those symmetric uniform moves over L into uniform moves over Psi we use the absolute value of the determinant of the Jacobian

$$\left\vert\frac{\partial\Psi_{ij}}{\partial L_{kl}}\right\vert_{i\ge j, k\ge l} = 2^d \vert L_{11}\vert^d \vert L_{22}\vert^{d - 1}\ldots \vert L_{dd}\vert$$

to construct a Hastings factor. In other words symmetric moves in the elements of L plus the Hastings factor give us a scheme to uniformly explore the space of positive-definite matrices. This is the matrix equivalent of how in the scalar case one can translate uniform symmetric moves over x on the real axis into uniform moves over y=x<sup>2</sup> on the positive real axis. As far as we are aware there is no reference to this simple approach in the literature.
