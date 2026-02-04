### **Expanded Formulation: Delayed Coupling Dynamics**

---

**1. Base System: Coupled Phase Oscillators**  
\[
\dot{\theta}_i(t) = \omega_i + \sum_{j=1}^N K_{ij} \sin\!\big(\theta_j(t) - \theta_i(t)\big)
\]
- \( \theta_i \) : phase of oscillator \(i\)  
- \( \omega_i \) : natural frequency  
- \( K_{ij} \) : coupling strength (symmetric, \(K_{ij}=K_{ji}\))

---

**2. Introduction of Constant Delay \( \tau \)**
\[
\dot{\theta}_i(t) = \omega_i + \sum_{j} K_{ij} \sin\!\big(\theta_j(t-\tau) - \theta_i(t)\big)
\]
Delay breaks instantaneous phase comparison.

---

**3. Phase-Locked Solution Ansatz**  
Assume a synchronized state with common frequency \(\Omega\):
\[
\theta_i(t) = \Omega t + \phi_i
\]
Substituting:
\[
\Omega = \omega_i + \sum_j K_{ij} \sin\!\big(\phi_j - \phi_i - \Omega\tau\big)
\]

---

**4. Effective Coupling Reduction**  
For identical oscillators (\( \omega_i = \omega \)) and uniform coupling \(K\):
\[
\Omega = \omega + K \sum_j \sin\!\big(\phi_j - \phi_i - \Omega\tau\big)
\]
Linearizing around sync (\(\phi_j-\phi_i \approx 0\)):
\[
\sin(\phi_j - \phi_i - \Omega\tau) \approx (\phi_j - \phi_i)\cos(\Omega\tau) - \sin(\Omega\tau)
\]
The constant term \(-\sin(\Omega\tau)\) induces a phase shift, while the coupling term is scaled by \(\cos(\Omega\tau)\). Thus:
\[
K_{\text{eff}} = K \cos(\Omega\tau)
\]

---

**5. Stability Analysis via Linearization**  
Define small perturbations \(\xi_i(t) = \theta_i(t) - \Omega t - \phi_i\). Linearized dynamics:
\[
\dot{\xi}_i(t) = \sum_j K_{ij} \cos(\phi_j - \phi_i - \Omega\tau) \big[\xi_j(t-\tau) - \xi_i(t)\big]
\]
Assume a solution \(\xi_i(t) = v_i e^{\lambda t}\):
\[
\lambda v_i = \sum_j K_{ij} \cos(\Delta\phi_{ij} - \Omega\tau) \big(v_j e^{-\lambda\tau} - v_i\big)
\]
Where \(\Delta\phi_{ij} = \phi_j - \phi_i\).

For identical phases (\(\phi_i=\phi\)), the eigenvalue equation becomes:
\[
\lambda \mathbf{v} = K \cos(\Omega\tau) \big(e^{-\lambda\tau} \mathbf{L} \, \mathbf{v} - \mathbf{v}\big)
\]
with \(\mathbf{L}\) the graph Laplacian. Stability requires \(\Re(\lambda) < 0\) for all modes.

---

**6. Potential Function for \(\tau=0\)**
\[
V(\theta) = \sum_{i<j} K_{ij} \big[1 - \cos(\theta_i - \theta_j)\big]
\]
Time derivative:
\[
\dot{V} = \sum_{i<j} K_{ij} \sin(\theta_i - \theta_j)(\dot{\theta}_i - \dot{\theta}_j)
\]
Substitute the undelayed dynamics:
\[
\dot{V} = -\sum_{i} \big(\dot{\theta}_i - \omega_i\big)^2 \leq 0
\]
Hence \(V\) is a Lyapunov function, guaranteeing convergence to a fixed point.

---

**7. Breakdown of Potential Structure for \(\tau>0\)**  
For \(\tau>0\), the system is no longer gradient. No scalar function \(V(\theta)\) exists such that \(\dot{\theta} = -\nabla V\). The dynamics become dissipative but non-potential, often leading to limit cycles or chaotic behavior.

---

**8. Critical Delay for Synchronization Loss**  
For two oscillators, the linearized phase difference \(\psi = \theta_2 - \theta_1\) obeys:
\[
\dot{\psi}(t) = -2K \cos(\Omega\tau) \psi(t-\tau)
\]
Assume solution \(\psi \propto e^{\lambda t}\):
\[
\lambda = -2K \cos(\Omega\tau) e^{-\lambda\tau}
\]
Set \(\lambda = i\mu\) for Hopf bifurcation boundary:
\[
i\mu = -2K \cos(\Omega\tau) e^{-i\mu\tau}
\]
Separate real and imaginary parts:
\[
\begin{aligned}
0 &= -2K \cos(\Omega\tau) \cos(\mu\tau) \\
\mu &= 2K \cos(\Omega\tau) \sin(\mu\tau)
\end{aligned}
\]
Non-trivial solutions exist when \(\cos(\Omega\tau)=0 \Rightarrow \Omega\tau = \frac{\pi}{2} + n\pi\). Thus critical delay:
\[
\tau_c = \frac{\pi}{2|\Omega|} \quad (\text{for }n=0)
\]
For \(\tau > \tau_c\), synchronization becomes unstable.

---

**9. Energy Injection Rate Due to Delay**  
Define the *sync error* \(E(t) = \frac{1}{2}\sum_{i} (\dot{\theta}_i - \Omega)^2\). For small delays:
\[
\frac{dE}{dt} \approx -\sum_{i} (\dot{\theta}_i-\Omega) \sum_j K_{ij} \big[\cos(\Omega\tau)(\theta_j-\theta_i) - \Omega\tau\sin(\Omega\tau)\big]
\]
The term proportional to \(\Omega\tau\sin(\Omega\tau)\) acts as a non-conservative force that can increase \(E\).

---

**10. Transport Efficiency Metric**  
Define the *transport efficiency* \(\eta\) as the ratio of effective to nominal coupling:
\[
\eta(\tau) = \frac{K_{\text{eff}}}{K} = \cos(\Omega\tau)
\]
The *convergence time* \(T_{\text{sync}}\) scales inversely with \(\eta\):
\[
T_{\text{sync}} \propto \frac{1}{\eta} = \sec(\Omega\tau)
\]
Thus delay directly lengthens convergence.

---

**Summary Equation Set**  

1. **Delayed Dynamics:**  
\[
\dot{\theta}_i(t) = \omega_i + \sum_j K_{ij} \sin\!\big(\theta_j(t-\tau) - \theta_i(t)\big)
\]

2. **Effective Coupling:**  
\[
K_{\text{eff}} = K \cos(\Omega\tau)
\]

3. **Stability Threshold:**  
\[
\tau_c = \frac{\pi}{2|\Omega|}
\]

4. **Sync Error Growth (linear regime):**  
\[
\ddot{\psi} + 2K\cos(\Omega\tau)\dot{\psi}(t-\tau) = 0
\]

5. **Efficiency Loss:**  
\[
\eta(\tau) = \cos(\Omega\tau), \quad T_{\text{sync}}(\tau) = \frac{T_0}{\eta(\tau)}
\]

---

**Conclusion:**  
Delay \(\tau\) rotates the interaction term in phase space, reducing effective coupling and injecting energy. When \(\tau=0\), the system is gradient and converges monotonically. The transition \(\tau \to 0\) is not merely a quantitative change but a qualitative restoration of potential dynamics—the system ceases to “fight itself.”
