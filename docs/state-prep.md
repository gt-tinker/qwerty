Arbitrary State Prep
====================

This document will describe the trick for arbitrary state prepration from
Section IV of [Shende et al][1]. (This is necessary to implement in the Qwerty
compiler to support superposition literals.)

Surprisingly, the overall idea of the algorithm is based on _un-computing_ the
desired state, qubit-by-qubit. It starts with uncomputing the rightmost qubit
back to $\ket{0}$, then uncomputes the next-to-rightmost qubit back to
$\ket{0}$, and so on until you have $\ket{0}^{\otimes n}$. Then you can run the
circuit that accomplishes this uncomputation backwards to compute the desired
state! I describe the algorithm in more detail below.

I. Algorithm
------------

**Input:** A vector $\ket{\psi}$ of dimension $2^n$, where $n \ge 1$

**Output:** A circuit (list of gates) which performs a $2^n{\times}2^n$ unitary
$U$ such that $U\ket{0}^{\otimes n} = \ket{\psi}$.

**Steps:**

1. Run the Undo subroutine on $\ket{\psi}$
2. Reverse the order of gates output by Undo, take the adjoint of each one, and
   return them

### A. Undo Subroutine

**Input:** A vector $\ket{\psi}$ of dimension $2^n$, where $n \ge 1$

**Output:** A circuit (list of gates) which un-computes $\ket{\psi}$
back to zero. Specifically, the product of the gates it returns is a
$2^n{\times}2^n$ unitary $U'$ such that $U'\ket{\psi} = \ket{0}^{\otimes n}$.

**Steps:**

1. If $n=1$, then use the procedure described above in Section 2B to calculate
   $\phi$ and $\theta$. Emit the gate $W^\dagger(\theta,\phi)$ (defined in
   Section 2C).
2. If $n>1$:
   1. For each $j \in {0,1,\ldots,2^{n-1}-1}$,
      1. Compute $\theta_j$, $\phi_j$, $r_j$, and $\gamma_j$ as described in Section 2D
      2. Emit $(n-1,j)$-controlled $W^{\dagger}(\theta_j,\phi_j)$. (See Section 2A
         for what I mean by $(n,j)$-controlled.)
   2. Create a new $2^{n-1}$-dimensional vector
      $\ket{\psi'} = \sum_{j=0}^{2^{n-1}-1} r_j e^{i \gamma_j}\ket{j}$.
      Recurse on $\ket{\psi'}$.

II. Notation and Lemmas
-----------------------

Otherwise, gates are defined as in Section 4.2 of Nielsen and Chuang ("Mike and
Ike"), which also matches the QCirc dialect definitions.

### A. Controlling on a Bitstring

You are probably familiar with the notation $CU$, which denotes a version of
$U$ that is controlled on the left. For example, $CX$ performs the following
operation:

$$
\begin{aligned}
    \ket{00} &\mapsto \ket{00} \\
    \ket{01} &\mapsto \ket{01} \\
    \ket{10} &\mapsto \ket{11} \\
    \ket{11} &\mapsto \ket{10}
\end{aligned}
$$

There is similar notation $cU$ which indicates being zero-controlled instead of
one-controlled. For example, $cX$ is the following transformation:

$$
\begin{aligned}
    \ket{00} &\mapsto \ket{01} \\
    \ket{01} &\mapsto \ket{00} \\
    \ket{10} &\mapsto \ket{10} \\
    \ket{11} &\mapsto \ket{11}
\end{aligned}
$$

You could also have an operation like $cCcX$, which flips the rightmost bit
only when the first three bits are $010$. I will refer to a generalized case of
this as $(n,j)$-controlled $U$. This represents performing $U$ on the
rightmost qubits if and only if the leftmost $n$ qubits match the $n$-bit
binary form of $j$. For example, $(3,2)$-controlled $X$ is exactly $cCcX$
because $2$ is written $010$ in three-bit binary.

In this document, I may write $(n,j)$-controlled $U$ as ${C^{(n,j)}}U$ to save
some space.

As an additional note, to accomplish $cCcX$ in an IR that supports only $CCCX$
(such as the QCirc MLIR dialect), you can perform
$(X\otimes I\otimes X \otimes I)CCCX(X\otimes I\otimes X \otimes I)$.

### B. Parameterization of an Arbitrary Single-Qubit State

Recall a qubit state $\ket{\psi}$ is defined as $\alpha\ket{0} + \beta\ket{1}$
where the complex amplitudes $\vert\alpha\vert^2 + \vert\beta\vert^2 = 1$.
One parameterization of this state can be written as
$\ket{\psi} = e^{i\gamma}(e^{-i\phi/2} \cos(\theta/2)\ket{0} + e^{i\phi/2} \sin(\theta/2)\ket{1})$, where
the parameters $\gamma$, $\theta$, and $\phi$ are real. $\gamma$ is the global
phase and $\phi$ is the relative phase.

Computing these three parameters given $\alpha$ and $\beta$ isn't too bad. You
can find $\gamma$ with $(\text{arg}(\alpha) + \text{arg}(\beta))/2$. Then if
$\alpha \ne 0$, one can compute $\phi = 2(\gamma - \text{arg}(\alpha))$. Or if
$\beta \ne 0$, then $\phi = 2(\text{arg}(\alpha) - \gamma)$ will work instead.
Finally, you can calculate $\theta = 2\arccos(\vert \alpha \vert)$.

### C. Preparation of an Arbitrary Single-Qubit State

If you know the $\phi$ and $\theta$ in the parameterization in Section 2B, you
can prepare $\ket{\psi}$ with the following tiny circuit:
$e^{i\gamma}R_z(\phi)R_y(\theta)\ket{0} = \ket{\psi}$. This is easy to verify.
Henceforth, I will define $W(\theta,\phi) = R_z(\phi)R_y(\theta)$. Note that
$W^\dagger(\theta,\phi)\ket{\psi} = e^{i\gamma}\ket{0}$, i.e., it can be used
to uncompute too.

### D. Crazy Factoring of a Multi-Qubit State

Imagine $\ket{\psi}$ is an $n$-qubit state where $n > 1$. Then you can do the
following insane linear algebra:

$$
\begin{aligned}
\ket{\psi} &= \sum_{i=0}^{2^n-1} \alpha_i \ket{i} \\
           &= \sum_{j=0}^{2^{n-1}-1} \alpha_{2j} \ket{2j} + \alpha_{2j+1} \ket{2j+1} \\
           &= \sum_{j=0}^{2^{n-1}-1} \alpha_{2j} \ket{j}\ket{0} + \alpha_{2j+1} \ket{j}\ket{1} \\
           &= \sum_{j=0}^{2^{n-1}-1} r_j e^{i \gamma_j}(e^{-i\phi_j/2} \cos(\theta_j/2) \ket{j}\ket{0} + e^{i\phi_j/2} \sin(\theta_j/2) \ket{j}\ket{1}) \\
           &= \sum_{j=0}^{2^{n-1}-1} r_j e^{i \gamma_j}\ket{j} \otimes (e^{-i\phi_j/2} \cos(\theta_j/2) \ket{0} + e^{i\phi_j/2} \sin(\theta_j/2) \ket{1})
\end{aligned}
$$

Hopefully you can see that each term in the sum) has its own qubit-like space.
Given two adjacent amplitudes $\alpha_{2j}$ and $\alpha_{2j+1}$ in the
statevector, you can compute $r_j$ with
$r_j = \sqrt{\vert \alpha_{2j} \vert^2 + \vert \alpha_{2j+1} \vert^2}$. If
you set $\alpha_{2j}' = \alpha_{2j}/r_j$ and $\alpha_{2j+1}' = \alpha_{2j+1}/r_j$,
then you can calculate the other parameters ($\gamma_j$, $\theta_j$, and
$\phi_j$) as described in Section 2B if you treat $\alpha_{2j}'$ as $\alpha$
and $\alpha_{2j+1}'$ as $\beta$.

### E. Multiplexed Ry and Rz Decomposition

This subsection describes how to decompose an $R_y$ that is multiplexed on a
single qubit. The definition of a multiplexed $R_y$ is equivalent to any of the
following (take your pick!):

$$
\begin{aligned}
cR_y(\alpha) CR_y(\beta) &= \ket{0}\bra{0} \otimes R_y(\alpha) + \ket{1}\bra{1} \otimes R_y(\beta) \\
                         &=
                         \begin{bmatrix}
                         R_y(\alpha) & 0 \\
                         0 & R_y(\beta)
                         \end{bmatrix}
\end{aligned}
$$

Theorem 4 of [Shende et al.][1] says that the following circuit is equivalent
to a multiplexed $R_y$:

$$
CX \quad (I \otimes R_y((\alpha - \beta)/2)) \quad CX \quad (I \otimes R_y((\alpha + \beta)/2))
$$

You can multiply out these matrices to verify if you want. **This also holds
for** $R_z$ **instead of** $R_y$.

III. Optimizations
------------------

This is based on Theorem 8 of [Shende et al][1].

### A. Quantum Multiplexers

For each call to the Undo subroutine (Section 1A), at Step 2(i)(b) in Section
1A we will generate the following circuit:

$$
C^{(n-1,2^{n-1}-1)}W^\dagger(\theta_{2^{n-1}-1},\phi_{2^{n-1}-1})\quad
\cdots
\quad C^{(n-1,1)}W^\dagger(\theta_1,\phi_1)
\quad C^{(n-1,0)}W^\dagger(\theta_0,\phi_0)
$$

The main algorithm will take the adjoint of this subcircuit, yielding the
following:

$$
C^{(n-1,0)}W(\theta_0,\phi_0)
\quad C^{(n-1,1)}W(\theta_1,\phi_1)
\quad\cdots
\quad C^{(n-1,2^{n-1}-1)}W(\theta_{2^{n-1}-1},\phi_{2^{n-1}-1})
$$

Expanding out the definition of $W$, we get

$$
C^{(n-1,0)}R_z(\phi_0) C^{(n-1,0)}R_y(\theta_0)
\quad
C^{(n-1,1)}R_z(\phi_1) C^{(n-1,1)}R_y(\theta_1)
\quad\cdots
\quad
C^{(n-1,2^{n-1}-1)}R_z(\phi_{2^{n-1}-1}) C^{(n-1,2^{n-1}-1)}R_y(\theta_{2^{n-1}-1})
$$

Now, we can observe that each of these pairs of z-rotation and phase gates are
happening in orthogonal subspaces, so these pairs commute with each other. In
fact, as long as we make sure the relative phase gates happen after the
z-rotation gates, we can rearrange this as the following:

$$
\left(
C^{(n-1,0)}R_z(\phi_0)
\quad
C^{(n-1,1)}R_z(\phi_1)
\quad\cdots
\quad
C^{(n-1,2^{n-1}-1)}R_z(\phi_{2^{n-1}-1})
\right)
\left(
C^{(n-1,0)}R_y(\theta_0)
\quad
C^{(n-1,1)}R_y(\theta_1)
\quad\cdots
\quad
C^{(n-1,2^{n-1}-1)}R_y(\theta_{2^{n-1}-1})
\right)
$$

Why is this interesting? Well, it turns out that each of the parenthesized
products are block diagonal, so I can rewrite the last expression as follows:

$$
\begin{bmatrix}
R_z(\phi_0) &           &        & 0 \\
          & R_z(\phi_1) &        &   \\
          &           & \ddots &   \\
0         &           &        & R_z(\phi_{2^{n-1}-1})
\end{bmatrix}
\begin{bmatrix}
R_y(\theta_0) &               &        & 0 \\
              & R_y(\theta_1) &        &   \\
              &               & \ddots &   \\
0             &               &        & R_y(\theta_{2^{n-1}-1})
\end{bmatrix}
$$

Shende et al. call each of these matrices _quantum multiplexers_. I found this
name counterintuitive at first, but regardless, they describe an amazing
optimization for lowering these so-called multiplexers to CNOTs and $R_y$
gates.

### B. Efficient Multiplexed Ry Decomposition

Let's focus on the multiplexed $R_y$ for a moment, first rewriting it with
Dirac notation (below, $k0$ means the bits of $k$ with a $0$ appended on the end):

$$
\begin{aligned}
\sum_{j=0}^{2^{n-1}-1} \ket{j}\bra{j} \otimes R_y(\theta_j)
&= \sum_{k=0}^{2^{n-2}-1} \ket{k0}\bra{k0} \otimes R_y(\theta_{2k})
                          + \ket{k1}\bra{k1} \otimes R_y(\theta_{2k+1}) \\
&= \sum_{k=0}^{2^{n-2}-1} \ket{k}\bra{k} \otimes \left(
                          \ket{0}\bra{0} \otimes R_y(\theta_{2k})
                          + \ket{1}\bra{1} \otimes R_y(\theta_{2k+1}) \right)
\end{aligned}
$$

Now we can apply the result in Section 2E to get the following:

$$
\begin{aligned}
&\sum_{k=0}^{2^{n-2}-1} \ket{k}\bra{k} \otimes \left(
                       \ket{0}\bra{0} \otimes R_y(\theta_{2k})
                       + \ket{1}\bra{1} \otimes R_y(\theta_{2k+1}) \right) \\
&{}= \sum_{k=0}^{2^{n-2}-1} \ket{k}\bra{k} \otimes \left(
                          CX (I \otimes R_y((\theta_{2k} - \theta_{2k+1})/2)) CX (I \otimes R_y((\theta_{2k} + \theta_{2k+1})/2))\right)
\end{aligned}
$$

We can make a crucial observation here: both CNOTs seen in the sum above are
running in all $2^{n-2}$ subspaces. So why bother controlling them? If a more
matrix-based argument helps your intuition, hopefully this equality is clear:

$$
\begin{bmatrix}
CX &    &        & 0 \\
   & CX &        &   \\
   &    & \ddots &   \\
0  &    &        & CX
\end{bmatrix} =
I_{2^{n-2}} \otimes CX
$$

Thus, we can split the sum above into

$$
\begin{aligned}
&\sum_{k=0}^{2^{n-2}-1} \ket{k}\bra{k} \otimes \left(CX (I \otimes R_y((\theta_{2k} - \theta_{2k+1})/2)) CX (I \otimes R_y((\theta_{2k} + \theta_{2k+1})/2))\right) \\
&{}= (I_{2^{n-2}}\otimes CX)
     \quad \left( \sum_{k=0}^{2^{n-2}-1} \ket{k}\bra{k} \otimes (I \otimes R_y((\theta_{2k} - \theta_{2k+1})/2)) \right)
     \quad (I_{2^{n-2}}\otimes CX)
     \quad \left( \sum_{k=0}^{2^{n-2}-1} \ket{k}\bra{k} \otimes (I \otimes R_y((\theta_{2k} + \theta_{2k+1})/2)) \right)
\end{aligned}
$$

Those two remaining sum terms are themselves $R_y$ multiplexors, except with
one fewer control qubit! Thus, this math forms the basis for a recursive
algorithm. The base case is simply an $R_y$ gate.

#### Algorithm

**Input:** A positive number of qubits $n$, a list of $n$ qubit indices to
operate on, and a list of rotation angles for a multiplexed $R_y$:
$\theta_0,\theta_1,\ldots,\theta_{2^{n-1}-1}$

**Output:** A circuit (list of gates) which performs a multiplexed $R_y$

**Steps:**

1. If $n=1$, emit $R_y(\theta_0)$.
2. Else:
   1. Calculate a new list of $2^{n-2}$ rotation angles. For
      $k=0,1,\ldots,2^{n-2}-1$, set $\theta_k' = (\theta_{2k} + \theta_{2k+1})/2$.
      Recurse with number of qubits $n-1$, the list of qubits excluding the
      **second-to-last** qubit, and
      $\theta_0',\theta_1',\ldots,\theta_{2^{n-2}-1}'$ as the rotation angles.
   2. Emit a CNOT with control $n-2$ and target $n-1$
   3. Calculate a new list of $2^{n-2}$ rotation angles. For
      $k=0,1,\ldots,2^{n-2}-1$, set $\theta_k'' = (\theta_{2k} - \theta_{2k+1})/2$.
      Recurse with number of qubits $n-1$, the list of qubits excluding the
      **second-to-last** qubit, and
      $\theta_0'',\theta_1'',\ldots,\theta_{2^{n-2}-1}''$ as the rotation angles.
   4. Emit another CNOT with control $n-2$ and target $n-1$

This works equivalently if you replace every $\theta_j$ with $\phi_j$ and every
$R_y$ with $R_z$.

### D. Overall Changes to Original Algorithm

That was a lot. So it's time to summarize.

The main change this optimization makes to the state preparation algorithm is
how the Undo subroutine (Section 1A) behaves. Instead of emitting
${C^{(n-1,j)}} {W^\dagger} ({\theta_j},{\phi_j})$ gates directly, the undo subroutine
should return two lists of angles:
$\theta_0,\theta_1,\ldots,\theta_{2^{n-1}-1}$ and
$\phi_0,\phi_1,\ldots,\phi_{2^{n-1}-1}$. Call these $\vec{\theta_n}$ and
$\vec{\phi_n}$ respectively. We can then rewrite Section 1 as follows:

#### Revised Overall Algorithm

**Input:** A vector $\ket{\psi}$ of dimension $2^n$, where $n \ge 1$

**Output:** A circuit (list of gates) which performs a $2^n{\times}2^n$ unitary
$U$ such that $U\ket{0}^{\otimes n} = \ket{\psi}$.

**Steps:**

1. Run the Undo subroutine on $\ket{\psi}$
2. For each $i \in {1,2,\ldots,n}$:
   1. Run the multiplexed $R_y$ synthesis algorithm in Section 3B with number
      of qubits $i$, list of qubits $0,1,\ldots,i-1$, and angles $\vec{\theta_i}$
   2. Run the multiplexed $R_z$ synthesis algorithm in Section 3B with number
      of qubits $i$, list of qubits $0,1,\ldots,i-1$, and angles $\vec{\phi_i}$

#### Revised Undo Subroutine

**Input:** A vector $\ket{\psi}$ of dimension $2^n$, where $n \ge 1$

**Output:** Stores lists of angles for multiplexed rotations in
$\vec{\theta_n}$ and $\vec{\phi_n}$

**Steps:**

1. If $n=1$, then use the procedure described above in Section 2B to calculate
   $\phi$ and $\theta$. Set $\vec{\theta_1} = \theta$ and $\vec{\phi_1} = \phi$.
2. If $n>1$:
   1. For each $j \in {0,1,\ldots,2^{n-1}-1}$,
      1. Compute $\theta_j$, $\phi_j$, $r_j$, and $\gamma_j$ as described in Section 2D
   2. Set $\vec{\theta_n} = \theta_0,\theta_1,\ldots,\theta_{2^{n-1}-1}$
      and $\vec{\phi_n} = \phi_0,\phi_1,\ldots,\phi_{2^{n-1}-1}$
   2. Create a new $2^{n-1}$-dimensional vector
      $\ket{\psi'} = \sum_{j=0}^{2^{n-1}-1} r_j e^{i \gamma_j}\ket{j}$.
      Recurse on $\ket{\psi'}$.

[1]: https://doi.org/10.1109/TCAD.2005.855930
[2]: https://en.wikipedia.org/wiki/Argument_(complex_analysis)#Computing_from_the_real_and_imaginary_part
