# Derivations

## Computation of the posterior $p\left(r_{t}\mid x_{1:t+h}\right)$

Starting with $h=1$, the joint distribution $p(r_{t},x_{1:(t+1)})$ is:

$$
\begin{align}p(r_{t},x_{1:(t+1)}) & =\sum_{r_{t+1}}p(r_{t},r_{t+1},x_{1:(t+1)})\\
 & =\sum_{r_{t+1}}p(r_{t},x_{1:t})p(r_{t+1},x_{t+1}\vert r_{t},x_{1:t})\\
 & =p(r_{t},x_{1:t})\sum_{r_{t+1}}p(x_{t+1}\vert r_{t},r_{t+1},x_{1:t})p(r_{t+1}\vert r_{t},\cancel{x_{1:t}})\\
 & =p(r_{t},x_{1:t})\sum_{r_{t+1}}p(x_{t+1}\vert r_{t},x_{(t-r_{t+1}):t})p(r_{t+1}\vert r_{t})
\end{align}
$$

Explanations:

* $p(x_{t+1}\vert r_{t},r_{t+1},x_{1:t})=p(x_{t+1}\vert r_{t},x_{(t-r_{t+1}):t})$, because knowing $r_{t+1}$ just selects the previous observations w.r.t. which we condition on.
* The transition probability $p(r_{t+1}\vert r_{t},x_{1:t})$ from the run length at time $t$ to the run length at time $t+1$ does not depend on the history of observations $x_{1:t}$. This is a model assumption in BOCD.

For $h=2$, the joint distribution $p(r_{t},x_{1:t+2})$ is:

$$
\begin{align}p(r_{t},x_{1:t+2}) & =\sum_{r_{t+1},r_{t+2}}p(r_{t},r_{t+1},r_{t+2},x_{1:(t+1)})\\
 & =\sum_{r_{t+1},r_{t+2}}p(r_{t},r_{t+1},x_{1:(t+1)})p(r_{t+2},x_{t+2}\vert r_{t},r_{t+1},x_{1:(t+1)})\\
 & =\sum_{r_{t+1},r_{t+2}}p(r_{t},r_{t+1},x_{1:(t+1)})p(x_{t+2}\vert\cancel{r_{t}},r_{t+1},r_{t+2},x_{1:(t+1)})p(r_{t+2}\vert\cancel{r_{t}},r_{t+1},\cancel{x_{1:(t+1)}})\\
 & =\sum_{r_{t+1},r_{t+2}}p(r_{t},r_{t+1},x_{1:(t+1)})p(x_{t+2}\vert r_{t+1},x_{\left(t+1-r_{t+2}\right):\left(t+1\right)})p(r_{t+2}\vert r_{t+1})\\
 & =p(r_{t},x_{1:t})\sum_{r_{t+1},r_{t+2}}p(x_{t+1}\vert r_{t},x_{(t-r_{t+1}):t})p(r_{t+1}\vert r_{t})p(x_{t+2}\vert r_{t+1},x_{\left(t+1-r_{t+2}\right):\left(t+1\right)})p(r_{t+2}\vert r_{t+1})
\end{align}
$$

Explanations:

* The same reasoning as before applies to both, the predictive distribution as well as the transition probability.
* In the last step, the previous result was used.

By following this pattern, for arbitrary $h\geq0$ the joint distribution $p\left(r_{t},x_{1:t+h}\right)$ becomes

$$
p(r_{t},x_{1:t+h})=p(r_{t},x_{1:t})\underbrace{\sum_{r_{t+1},...,r_{t+h}}\prod_{m=1}^{h}\left[p(x_{t+m}\vert r_{t+m-1},x_{(t+m-r_{t+m}):\left(t+m-1\right)})p(r_{t+m}\vert r_{t+m-1})\right]}_{=:M}
$$

The factor $M$ is understood to be one, if $h=0$.

The joint distribution $p\left(r_{t},x_{1:t+h}\right)$ can therefore be computed by the product of the joint distribution $p(r_{t},x_{1:t})$, which BOCD computes anyway, and a factor  $M$, which depends on the predictive and transition probabilities at future time steps. These objects, however, are components of the BOCD algorithm which are anyway computed in every time step and once computed can be re-used internally in the implementation.

The run length posterior can be computed from the joint distribution:

$$p(r_{t}\vert x_{1:t+h})=\frac{p(r_{t},x_{1:t+h})}{p(x_{1:t+h})}=\frac{p(r_{t},x_{1:t+h})}{\sum_{r_{t}}p(r_{t},x_{1:t+h})}$$

The factor $M$ can be best understood as being the product of $h$ matrices of the form $M_{t+m}$. Let's, e.g., assume that $t=1$. The first product for $m=1$, can be written as

$$
M_{2}=\underbrace{\left[\begin{array}{ccc}
p(x_{2}) & p(x_{2}\vert x_{1}) & 0\\
p(x_{2}) & 0 & p(x_{2}\vert x_{1})
\end{array}\right]}_{p(x_{2}\vert x_{(2-r_{2}):1})}\circ\underbrace{\left[\begin{array}{ccc}
p(r_{2}=0\vert r_{1}=0) & p(r_{2}=1\vert r_{1}=0) & 0\\
p(r_{2}=0\vert r_{1}=1) & 0 & p(r_{2}=2\vert r_{1}=1)
\end{array}\right]}_{p(r_{2}\vert r_{1})}
$$

where $\circ$ denotes element-wise multiplication. Similarly, for $m=2$ we have:

$$
M_{3}=\underbrace{\left[\begin{array}{cccc}
p(x_{3}) & p(x_{3}\vert x_{1}) & 0 & 0\\
p(x_{3}) & 0 & p(x_{3}\vert x_{1:2}) & 0\\
p(x_{3}) & 0 & 0 & p(x_{3}\vert x_{1:2})
\end{array}\right]}_{p(x_{3}\vert x_{(3-r_{3}):2})}\circ\underbrace{\left[\begin{array}{cccc}
p(r_{3}=0\vert r_{2}=0) & p(r_{3}=1\vert r_{2}=0) & 0 & 0\\
p(r_{3}=0\vert r_{2}=1) & 0 & p(r_{3}=2\vert r_{2}=1) & 0\\
p(r_{3}=0\vert r_{2}=2) & 0 & 0 & p(r_{3}=3\vert r_{2}=2)
\end{array}\right]}_{p(r_{3}\vert r_{2})}
$$

For all matrices $M_{t+m}$, $r_{t+m-1}$ increases along the rows and $r_{t+m}$ along the columns.

If $h$ was just two, then $M=M_{2}M_{3}$ in this example. The joint distribution can be written as a vector:
$$
\left[\begin{array}{c}
p(r_{1}=0,x_{1:3})\\
p(r_{1}=1,x_{1:3})
\end{array}\right]=\left[\begin{array}{c}
p(r_{1}=0,x_{1})\\
p(r_{1}=1,x_{1})
\end{array}\right]\circ M
$$

It is possible to use the particular structure of each matrix $M_{m}$ (which has only elements in the first column and first upper diagonal) to implement this matrix multiplication efficiently (see `bocd._log_matmul_fast()`).