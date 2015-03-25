## PYTHON IBP (PyIBP)

David Andrzejewski (andrzeje@cs.wisc.edu) 
Department of Computer Sciences 
University of Wisconsin-Madison, USA


### DESCRIPTION

This code uses NumPy and SciPy to efficiently implement "accelerated"
Gibbs sampling [1] for the linear-Gaussian infinite latent feature
model (aka Indian Buffet Process or IBP) [2].

This code also allows the use real-valued latent features [3], using a
novel slice sampler for compatibility with the accelerated Gibbs
scheme [5].

New features are sampled using a Metropolis-Hastings scheme [4].


### EXAMPLE USAGE

See ./test/example.py for example usage on a simple synthetic dataset
consisting of latent factors with nice structure.  This dataset is
derived from the one packaged with Finale Doshi-Velez's accelerated
IBP code (http://people.csail.mit.edu/finale), but my understanding is
that it originally appeared in earlier IBP work [2].


### ACKNOWLEDGEMENTS

Thanks to Finale Doshi-Velez for making MATLAB code available and for
answering detailed questions about inference.  The modified slice
sampler for real-valued latent features was the result of discussions
with David Knowles.


### LICENSE

This software is open-source, released under the terms of the GNU
General Public License version 3, or any later version of the GPL (see
COPYING).


### REFERENCES

[1] 
Accelerated Gibbs sampling for the Indian buffet process
Finale Doshi and Zoubin Ghahramani, ICML 2009

[2]
Infinite latent feature models and the Indian buffet process
Tom Griffiths and Zoubin Ghahramani, NIPS 2006

[3]
Infinite Sparse Factor Analysis and Infinite Independent Components Analysis
David Knowles and Zoubin Ghahramani, ICA 2007

[4]
Modeling Dyadic Data with Binary Latent Factors
Edward Meeds, Zoubin Ghahramani, Radford Neal, and Sam Roweis, NIPS 2006

[5]
[Accelerated Gibbs Sampling for Infinite Sparse Factor Analysis](http://www.david-andrzejewski.com/publications/llnl-accelerated-gibbs.pdf)
David Andrzejewski, LLNL Technical Report (LLNL-TR-499647)

