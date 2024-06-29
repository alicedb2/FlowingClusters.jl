# FlowingClusters.jl

FlowingClusters.jl performs unsupervised clustering of arbitrary real data that have first been deformed into a base space under a FFJORD normalizing flow ([Grathwohl et al. 2018](https://arxiv.org/abs/1810.01367)).

The generative model in the base space consists of a non-parametric Chinese Restaurant Process (CRP) prior ([Pitman 1995](https://doi.org/10.1007%2FBF01213386), [Aldous 1985](https://doi.org/10.1007%2FBFb0099421), [Frigyik et al. 2010 Tutorial](https://web.archive.org/web/20190327085650/https://pdfs.semanticscholar.org/775e/5727f5df0cb9bf834af2ea2548a696c27a38.pdf)). The base distribution of the CRP is given by a [normal-inverse-Wishart distribution](https://en.wikipedia.org/wiki/Normal-inverse-Wishart_distribution) and the data likelihood by multivariate normal. For the hyperprior for the parameters of the normal-inverse-Wishart distribution and the CRP prior we use an independence Jeffreys prior. The CRP part of the generative model is therefore given by
$$
\begin{matrix}
\pi   \alpha \quad\sim\quad CRP(\alpha),\\
\mu_\omega, \Sigma_\omega   \pi,\mu_0, \lambda_0, \Psi_0, \nu_0 \quad\sim\quad normal-inverse-Wishart(\mu_0, \lambda_0, \Psi_0, \nu_0), \omega\in\pi,\\
z_j   \omega, \mu_\omega, \Sigma_\omega \quad\sim\quad MvNormal(\mu_\omega, \Sigma_\omega), j\in\omega.
\end{matrix}
$$