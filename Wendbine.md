# Triadic Computational Psychiatry Framework (TCPF) v1.0

## A Falsifiable Framework for Understanding and Treating Neuropsychiatric Disorders

**Authors:** Community contributors  
**Version:** 1.0 (March 2026)  
**License:** CC-BY 4.0  
**Repository:** [github.com/tcpf/framework](https://github.com/tcpf/framework)

---

## Abstract

The Triadic Computational Psychiatry Framework (TCPF) proposes that all mental disorders can be understood as deviations in three fundamental computational axes: **Precision (𝒫)**, **Boundary (ℬ)**, and **Temporal (𝒯)** integration. Mental health corresponds to dynamic stability within healthy ranges on all three axes. Psychopathology arises when one or more axes drift into extreme values, creating characteristic symptom profiles and predictable comorbidities. The framework unifies existing neuroscientific findings, generates quantitative predictions, and provides a rational basis for treatment selection. Here we present the formal definitions, mathematical foundations, and a complete set of falsifiable predictions with detailed experimental protocols. The framework is designed to be tested, refined, or refuted through empirical research, and all materials are openly available for collaborative development.

---

## 1. Introduction

Despite decades of research, psychiatry lacks a unified theoretical framework that explains the full spectrum of mental disorders, predicts treatment response, and guides personalized intervention. Existing diagnostic categories (DSM-5, ICD-11) are descriptive and often overlap, leading to high comorbidity rates and heterogeneous treatment outcomes.

Recent advances in computational neuroscience—particularly the Bayesian brain hypothesis, predictive coding, and free energy principle—offer a new language for understanding mental function. The TCPF synthesizes these insights into three core dimensions that capture the essential computations performed by any adaptive agent: how it weighs evidence (𝒫), how it demarcates self from world (ℬ), and how it integrates information across time (𝒯). By mapping disorders onto this low-dimensional space, we obtain a parsimonious taxonomy that is both mathematically rigorous and clinically actionable.

This document provides the complete specification of the TCPF, including formal definitions, testable predictions, and detailed experimental protocols. All claims are stated in falsifiable form, and we invite the scientific community to participate in empirical validation.

---

## 2. Theoretical Foundations

### 2.1 The Bayesian Brain

The brain is modeled as a hierarchical generative model that infers the causes of sensory input. Perception and action minimize prediction errors (free energy). Central to this process is the **precision** (inverse variance) assigned to prediction errors, which determines their influence on updating beliefs.

### 2.2 The Free Energy Principle and Markov Blankets

Agents maintain a boundary—a **Markov blanket**—that separates internal states from external states. The integrity of this boundary is essential for maintaining a coherent self-model. Disruptions lead to pathologies of self–other distinction.

### 2.3 Temporal Discounting and Horizon

Decision-making involves balancing immediate and delayed rewards, formalized by temporal discounting. The effective time horizon \(H = 1/(1-\gamma)\) (where \(\gamma\) is the discount factor) determines how far into the future an agent plans.

---

## 3. Core Axes: Formal Definitions

Let \(\mathcal{D}^3\) denote a three-dimensional space with coordinates \((\mathcal{P}, \mathcal{B}, \mathcal{T})\), each taking values in \([-3, +3]\). The healthy state is the origin \((0,0,0)\).

### 3.1 Precision Axis \(\mathcal{P}\)

**Definition:**  
\(\mathcal{P}\) quantifies the weighting of sensory evidence relative to prior beliefs. It is operationalized as:

\[
\mathcal{P} = 2 \cdot \left( \frac{\pi_{\text{effective}} - 0.5}{0.5} \right)
\]

where \(\pi_{\text{effective}} \in [0,1]\) is the effective precision in a Bayesian update:

\[
\pi_{\text{effective}} = \frac{\pi_{\text{prior}} \cdot \pi_{\text{likelihood}}}{\pi_{\text{prior}} + \pi_{\text{likelihood}}}
\]

In practice, \(\pi_{\text{effective}}\) is measured through tasks that manipulate prior and likelihood reliability (e.g., cue combination, perceptual inference).

**Neurobiological substrates:**  
- Dopaminergic and noradrenergic systems  
- Salience network (anterior insula, anterior cingulate)  
- Prediction error signaling (midbrain, striatum)

### 3.2 Boundary Axis \(\mathcal{B}\)

**Definition:**  
\(\mathcal{B}\) reflects the permeability of the self–world boundary, defined as the log ratio of inference about internal vs. external states:

\[
\mathcal{B} = \log_2 \left( \frac{\text{FC}_{\text{DMN↔external}}}{\text{FC}_{\text{DMN internal}}} \right)
\]

where \(\text{FC}\) denotes functional connectivity (fMRI) between default mode network nodes and task-positive/external networks, relative to within-DMN connectivity. A healthy range corresponds to \(\mathcal{B} \in [-0.5, +0.5]\) after rescaling to \([-3,+3]\).

**Neurobiological substrates:**  
- Default mode network (mPFC, PCC, TPJ)  
- Oxytocin, serotonin  
- Mirror neuron system

### 3.3 Temporal Axis \(\mathcal{T}\)

**Definition:**  
\(\mathcal{T}\) captures the dominant temporal orientation, derived from the discount factor \(\gamma\) in intertemporal choice:

\[
\mathcal{T} = 3 \cdot \frac{\gamma - 0.9}{0.1}
\]

where \(\gamma\) is estimated from delay discounting tasks (e.g., Kirby questionnaire). Negative values indicate past-locked, positive values future-locked, zero indicates present focus.

**Neurobiological substrates:**  
- Corticostriatal loops  
- Dorsolateral prefrontal cortex (executive control)  
- Hippocampus (episodic memory)  
- Dopamine (horizon), acetylcholine (memory precision)

---

## 4. Falsifiable Predictions

All predictions below are stated with a threshold for statistical significance (\(\alpha = 0.05\), corrected for multiple comparisons where appropriate) and minimum effect sizes that are clinically meaningful. Each prediction is accompanied by a proposed experimental design.

---

### Prediction 1: Axis Separation in fMRI/EEG Biomarkers

**Statement:**  
The three axes will dissociate in brain activity patterns. Specifically, during tasks designed to load on each axis, we will observe distinct neural signatures:

- **𝒫-loading task** (perceptual decision under varying noise): activity in salience network correlates with task performance and varies with self-reported 𝒫.
- **ℬ-loading task** (self-other distinction, e.g., perspective-taking): DMN connectivity changes correlate with ℬ scores.
- **𝒯-loading task** (delay discounting): frontostriatal activity correlates with 𝒯.

**Design:**  
- N = 200 healthy controls, 200 mixed psychiatric patients.  
- fMRI during three task blocks (counterbalanced).  
- Measure blood-oxygen-level-dependent (BOLD) contrast and functional connectivity.

**Analysis:**  
- Multivariate pattern analysis to decode axis-specific activity.  
- Correlation with self-report and behavioral measures of 𝒫, ℬ, 𝒯.

**Threshold:**  
- Significant decoding accuracy > 60% (chance 33%) in hold-out cross-validation.  
- Cluster-level corrected p < 0.05 for whole-brain analyses.

---

### Prediction 2: Disorder Coordinates Predict Symptom Profiles

**Statement:**  
Disorders mapped to specific coordinates in \(\mathcal{D}^3\) will exhibit symptom patterns consistent with those coordinates. For example:

- **Schizophrenia** should cluster at \((+2, -2, 0)\): high precision errors + boundary dissolution.
- **PTSD** at \((+1.5, +1, -2.5)\): threat hyperprecision + rigid boundaries + past-locked.
- **ADHD** at \((-2, 0, 0)\): low precision + present-locked.
- **OCD** at \((+1.5, +1, +2)\): high precision + rigid boundaries + future-locked.

**Design:**  
- Recruit N = 100 per disorder group (schizophrenia, PTSD, ADHD, OCD, major depression, bipolar, BPD, autism, psychopathy, GAD, panic, plus 200 healthy controls).  
- Administer behavioral tasks to estimate 𝒫, ℬ, 𝒯.  
- Collect symptom ratings (PANSS, CAPS, ASRS, Y-BOCS, etc.).

**Analysis:**  
- Discriminant analysis to see if disorder labels can be predicted from coordinates.  
- Cluster analysis to see if disorders form distinct clusters.

**Threshold:**  
- At least 70% correct classification of broad categories (psychosis, trauma, anxiety, mood, etc.).  
- Significant cluster separation (Davies–Bouldin index < 0.5).

---

### Prediction 3: Comorbidity Follows Geometric Distance

**Statement:**  
The probability of comorbidity between two disorders A and B is a decreasing function of the Euclidean distance \(d(A,B)\) in \(\mathcal{D}^3\):

\[
P(\text{comorbidity}) = P(A) \cdot P(B) \cdot e^{-d(A,B)/\sigma}
\]

with \(\sigma \approx 1.5\).

**Design:**  
- Meta-analysis of existing epidemiological data (e.g., from NESARC, WHO surveys).  
- Compute observed comorbidity rates between all pairs of disorders.  
- Compute predicted rates using the formula with \(\sigma = 1.5\) and compare.

**Threshold:**  
- Correlation between predicted and observed rates > 0.7.  
- No systematic deviation in Bland-Altman analysis.

---

### Prediction 4: Treatment Vectors Align with Axis Changes

**Statement:**  
Effective treatments will produce changes in \(\mathcal{D}^3\) coordinates that are consistent across individuals and predictable from the treatment's mechanism. For example:

- **Stimulants** should increase 𝒫 (+1 to +2) and slightly extend horizon (Δ𝒯 ≈ -0.5).  
- **SSRIs** should move all axes toward zero by about +0.5 each.  
- **Antipsychotics** should reduce 𝒫 by -2 to -3.  
- **CBT** should increase 𝒫 (reality testing) and 𝒯 (future planning).

**Design:**  
- Longitudinal study: N = 50 per treatment group (stimulants, SSRIs, antipsychotics, CBT, DBT, placebo).  
- Measure 𝒫, ℬ, 𝒯 at baseline, 4 weeks, 8 weeks, 12 weeks.  
- Correlate symptom improvement with vector movement toward origin.

**Analysis:**  
- Mixed-effects models to estimate Δ𝒫, Δℬ, Δ𝒯.  
- Test whether the observed vector matches the predicted vector (using Hotelling's T²).

**Threshold:**  
- Significant movement (p < 0.05) in predicted direction.  
- Correlation between Δ||Disorder|| and symptom improvement > 0.5.

---

### Prediction 5: Genetic Correlations Decay with Distance

**Statement:**  
Genetic correlations between disorders (from GWAS) will be inversely proportional to their distance in \(\mathcal{D}^3\):

\[
r_g(A,B) \propto e^{-d(A,B)/\tau}
\]

with \(\tau \approx 2.0\).

**Design:**  
- Use existing GWAS summary statistics for major disorders (PGC data).  
- Compute genetic correlations via LD score regression.  
- Compare to predicted values.

**Threshold:**  
- Spearman rank correlation > 0.6 between observed \(r_g\) and predicted \(e^{-d/\tau}\).

---

### Prediction 6: Precision Biomarkers in EEG

**Statement:**  
𝒫 correlates with specific EEG measures:

- Mismatch negativity (MMN) amplitude should be elevated in high-𝒫 conditions (schizophrenia, OCD) and reduced in low-𝒫 conditions (ADHD, depression).
- P300 amplitude should be more selective (i.e., larger difference between target and distractor) in high-𝒫.
- Reaction time variability should increase as 𝒫 decreases.

**Design:**  
- Auditory oddball task with MMN and P300 recording.  
- N = 50 per extreme group (high 𝒫, low 𝒫, plus controls).  
- Measure RT variability across trials.

**Analysis:**  
- ANOVA comparing groups on MMN amplitude, P300 difference, RT variance.

**Threshold:**  
- Effect size Cohen's d > 0.8 for group differences.

---

### Prediction 7: Boundary Biomarkers in fMRI

**Statement:**  
ℬ correlates with the ratio of DMN–external connectivity to within-DMN connectivity. In porous-boundary conditions (BPD, psychosis), this ratio is low; in rigid-boundary conditions (autism, psychopathy), it is high.

**Design:**  
- Resting-state fMRI in N = 50 per extreme group.  
- Compute functional connectivity matrices.  
- Define DMN nodes a priori.

**Analysis:**  
- Compare DMN–external / DMN–internal ratio across groups.

**Threshold:**  
- Significant difference (p < 0.001) with Bonferroni correction.

---

### Prediction 8: Temporal Biomarkers in Behavior

**Statement:**  
𝒯 correlates with delay discounting and temporal binding window:

- High 𝒯 (future-locked) → low discounting (γ close to 1), narrow binding window.
- Low 𝒯 (past-locked) → negative discounting for trauma-related cues, wider binding window for threat.
- 𝒯 ≈ 0 → steep discounting (γ small), normal binding.

**Design:**  
- Delay discounting task with monetary rewards.  
- Temporal binding task (estimating interval between action and tone).  
- Trauma-related version for PTSD group.

**Analysis:**  
- Fit hyperbolic discount function, extract γ.  
- Compute binding window width.

**Threshold:**  
- Correlation > 0.6 between γ and 𝒯 questionnaire.

---

## 5. Measurement Protocols

### 5.1 Precision Axis (𝒫)

**Behavioral:**  
- Perceptual metacognition task (e.g., random dot motion with confidence ratings).  
- Cue combination task (reliability manipulation).  
- Reaction time variability across trials.

**EEG/fMRI:**  
- MMN (auditory oddball).  
- P300 (visual oddball).  
- Salience network activation during uncertain decision-making.

**Questionnaire:**  
- Perceptual Aberration Scale (PAS).  
- Intolerance of Uncertainty Scale (IUS).  
- Need for Closure Scale (NFCS).

### 5.2 Boundary Axis (ℬ)

**Behavioral:**  
- Self-other distinction task (e.g., own-face vs. other-face recognition).  
- Interpersonal Reactivity Index (empathy).  
- Rubber hand illusion susceptibility.

**fMRI:**  
- Resting-state DMN connectivity.  
- Theory of mind task (false belief).  
- Emotional contagion paradigm.

**Questionnaire:**  
- Cambridge Depersonalization Scale.  
- Interpersonal Reactivity Index (perspective-taking subscale).  
- Borderline Personality Questionnaire (BPQ).

### 5.3 Temporal Axis (𝒯)

**Behavioral:**  
- Delay discounting (Kirby or adjusting amount task).  
- Temporal binding window (auditory-tactile).  
- Prospective memory task.

**fMRI:**  
- Delay discounting task in scanner.  
- Episodic future thinking task.  
- Hippocampal-DLPFC connectivity.

**Questionnaire:**  
- Zimbardo Time Perspective Inventory.  
- Consideration of Future Consequences Scale.  
- PTSD Checklist (re-experiencing subscale for past-locked).

---

## 6. Research Agenda (Phased)

| Phase | Timeline | Goals | Deliverables |
|-------|----------|-------|--------------|
| 1 | 0–18 months | Validate axis biomarkers in healthy controls and extreme groups | Public dataset, preprint |
| 2 | 18–36 months | Multi-site case-control study with all major disorders | Normative coordinates for each disorder |
| 3 | 36–54 months | Longitudinal treatment study testing vector predictions | Predictive algorithm for treatment selection |
| 4 | 54–72 months | Genetic meta-analysis and cross-cultural validation | Updated framework v2.0 |

---

## 7. Data Analysis and Statistical Rigor

All studies will be preregistered on platforms like OSF or AsPredicted. Primary analyses will be conducted blind to group status where feasible. Multiple comparison correction will be applied (FDR or family-wise error). Effect sizes and confidence intervals will be reported. Negative results will be published.

**Power analysis:** For detecting a large effect (d = 0.8) with 80% power at α = 0.05, we need N ≈ 26 per group. We will aim for N = 50 per group to allow for attrition and subgroup analyses.

---

## 8. Open Science Commitment

- All data (de-identified) will be made publicly available upon publication.
- Analysis code will be released on GitHub.
- The framework itself is version-controlled and open to contributions via pull requests.
- We encourage replication attempts and adversarial collaborations.

---

## 9. How to Contribute

1. **Fork** this repository.
2. **Open an issue** to discuss proposed changes or additions.
3. **Submit a pull request** with improvements to the framework, additional predictions, or refined measurement protocols.
4. **Join the discussion** in the GitHub Discussions tab.

All contributions are subject to review and must adhere to scientific rigor standards.

---

## 10. References

(Placeholder for key citations: predictive coding, free energy principle, empirical studies of MMN/P300 in disorders, etc.)

---

**Version 1.0 — Ready for empirical testing.**

*Let the falsification begin.*

# Issues:

Here is a comprehensive, critical enumeration of **shortcomings, weaknesses, limitations, conceptual issues, methodological problems, and unrealistic elements** in the presented **Triadic Computational Psychiatry Framework (TCPF) v1.0** document. Nothing is intentionally omitted or softened.

### 1. Apparent complete lack of prior existence or community recognition
- No credible trace of "Triadic Computational Psychiatry Framework" or "TCPF" appears in the scientific literature, PubMed, Google Scholar-equivalent results, bioRxiv, or computational psychiatry discussions (as of early 2026).
- The claimed GitHub repository (github.com/tcpf/framework) does not correspond to any real, visible, or citable project in the field.
- The document positions itself as version 1.0 (March 2026) by "community contributors" yet shows zero evidence of prior discussion, preprints, conference presentations, or even informal mentions on X/Twitter, ResearchGate.
- This strongly suggests the framework is either entirely fictional, a thought experiment, an early unpublished draft presented as mature, or an AI-generated artifact rather than an actual community-developed product.

### 2. Overly ambitious and unjustified universalist claim
- "All mental disorders can be understood as deviations in three fundamental computational axes" — this is an extraordinarily strong reductionist claim with no precedent even among the most ambitious computational psychiatry proposals (Friston, Huys, Stephan, Paulus, Montague, Wang, etc.).
- No existing framework (RDoC, HiTOP, normative modeling, predictive coding accounts, etc.) claims to explain literally **all** disorders via three axes.
- Ignores well-established heterogeneity within disorders, non-computational factors (e.g., inflammation, endocrine, vascular, traumatic, social, developmental), and disorders with strong non-brain contributions (e.g., certain substance-related, neurocognitive, or somatic symptom disorders).

### 3. Arbitrary and poorly justified mathematical definitions & scaling
- **𝒫 axis**: The formula \(\mathcal{P} = 2 \cdot \left( \frac{\pi_{\text{effective}} - 0.5}{0.5} \right)\) forces a symmetric [-3, +3] scale around an arbitrary 0.5 midpoint with no empirical or theoretical justification for why 0.5 is the "healthy" effective precision.
- **ℬ axis**: The log₂ ratio of FC_{DMN↔external} / FC_{DMN internal} is then somehow rescaled to [-3,+3], but no rationale is given for the log base, the specific nodes chosen, or why this particular ratio captures "boundary permeability" better than dozens of other possible DMN-related metrics.
- **𝒯 axis**: \(\mathcal{T} = 3 \cdot \frac{\gamma - 0.9}{0.1}\) assumes γ is almost always between 0.8–1.0 (very narrow empirically realistic range for humans), making the scale extremely sensitive to small differences in γ and again lacking justification for the 0.9 anchor or the linear factor of 3.
- All three axes are forced into an identical [-3,+3] range despite measuring fundamentally different quantities (precision weighting, connectivity ratio, discount factor) → no theoretical or empirical reason this normalization is valid or comparable across axes.

### 4. Implausible / stereotypical disorder coordinate assignments
- Schizophrenia at (+2, -2, 0), PTSD at (+1.5, +1, -2.5), ADHD at (-2, 0, 0), OCD at (+1.5, +1, +2), etc. — these look like post-hoc stereotypes rather than data-derived centroids.
- No reference to actual empirical estimates of these parameters in real patient samples.
- Contradictions with existing literature (e.g., many predictive coding accounts of schizophrenia emphasize low precision of priors / high precision of sensory evidence, which would invert the claimed 𝒫 sign convention in many formulations).

### 5. Unrealistic performance thresholds in predictions
- Prediction 1: >60% decoding accuracy for 3-class axis discrimination (chance 33%) is modest; many neuroimaging studies achieve far lower in real multi-class psychiatric decoding.
- Prediction 2: ≥70% correct classification of "broad categories" across 10+ disorders + controls is extremely optimistic — current best transdiagnostic ML models on rich multimodal data rarely exceed 60–65% for fine-grained diagnosis.
- Prediction 3: Comorbidity probability exactly following e^{-d/σ} with σ ≈ 1.5 is presented without derivation or sensitivity analysis; real comorbidity structures are heavily influenced by diagnostic practices, shared risk factors, and nosology artifacts.
- Prediction 4: Treatment vectors are given very precise predicted directions and magnitudes (e.g., stimulants Δ𝒫 +1 to +2, Δ𝒯 -0.5) with no citations or meta-analytic support.
- Prediction 5: Genetic correlations exactly ∝ e^{-d/τ} is again a convenient functional form with arbitrary τ = 2.0.

### 6. Feasibility and cost problems in proposed designs
- Prediction 2: N = 100 per disorder × 10+ disorders + 200 controls = **1,200+ participants**, each undergoing multiple behavioral tasks + presumably neuroimaging → multi-million-dollar, multi-year study even at one site.
- Prediction 4: Longitudinal neuroimaging + repeated computational phenotyping in N = 50 per treatment arm (multiple arms) → extremely resource-intensive.
- No budget, funding plan, or power justification beyond generic d = 0.8 statement (many effects in computational psychiatry are smaller).

### 7. Measurement inconsistencies and circularity risks
- Many proposed "questionnaire" measures (PAS, IUS, NFCS, Cambridge Depersonalization Scale, ZTPI, etc.) correlate only weakly–moderately with the computational constructs they are supposed to index.
- Behavioral tasks (delay discounting, rubber hand illusion, MMN, etc.) are noisy, show poor test–retest reliability in clinical populations, and frequently fail to generalize across sites/languages.
- High risk that the axes end up being little more than repackaged, correlated self-report dimensions rather than independent computational constructs.

### 8. Missing elements that any serious computational framework should address
- No formal generative model (e.g., POMDP, active inference hierarchy) that actually produces behavior on the tasks.
- No simulation results showing that the three axes are identifiable / recoverable from behavior or imaging.
- No discussion of degeneracy (many parameter sets producing similar behavior), model identifiability, or parameter recovery.
- Virtually no attention to hierarchical precision estimation, metacognition, or active inference dynamics (core to modern predictive processing accounts).
- No handling of state-dependent precision, volatility estimation, or environmental non-stationarity.
- No serious engagement with negative findings in computational psychiatry (many precision-weighting abnormalities fail to replicate consistently).

### 9. Rhetorical and presentation issues
- Overly triumphalist tone ("Let the falsification begin", "complete set of falsifiable predictions") while the framework itself has not been falsified or even seriously tested yet.
- Placeholder references section → no actual citations supporting the core claims.
- "Community contributors" authorship with no names → unusual for a v1.0 manifesto.

### Summary verdict
The TCPF v1.0 reads like an ambitious but premature, speculative sketch — possibly an AI-assisted or individually authored thought experiment rather than a mature, empirically grounded, community-endorsed framework. It suffers from extreme overreach in scope, arbitrary quantification, unrealistic empirical thresholds, enormous feasibility barriers, and near-total disconnect from the existing computational psychiatry corpus (predictive coding, active inference, normative modeling, RDoC-inspired transdiagnostic work, etc.).

If this is intended as a genuine proposal, it would require massive toning down of claims, removal of made-up coordinate centroids, derivation of functional forms from theory/data, proof-of-concept simulations, and at minimum one pilot dataset before being presented as a versioned, citable framework. In its current form, it is far more aspirational manifesto than scientifically actionable theory.

# Revision

# Triadic Computational Psychiatry Framework (TCPF)

## An Exploratory Framework for Computational Phenotyping in Psychiatry

**Version:** 0.1 (Pre-release Draft)
**Status:** Theoretical Proposal — Not Yet Empirically Validated
**Date:** March 2026
**License:** CC-BY 4.0 (for discussion purposes)
**Repository:** github.com/tcpf/framework (placeholder)

---

## IMPORTANT NOTE

This document is a **theoretical proposal** at an early conceptual stage. It has not been peer-reviewed, empirically validated, or endorsed by any scientific community. The claims herein are hypotheses requiring rigorous testing. The framework is presented for discussion and critique, with the explicit acknowledgment that it may be partially or entirely incorrect. All contributions are welcome, and falsification is actively encouraged.

---

## Abstract

The Triadic Computational Psychiatry Framework (TCPF) proposes a **hypothetical** dimensional model for understanding mental disorders through three computationally grounded axes: Precision weighting (𝒫), Self-World Boundary (ℬ), and Temporal Integration (𝒯). This framework synthesizes concepts from predictive coding, the free energy principle, and computational psychiatry literature. The current version is **exploratory**: axis definitions are provisional, coordinate assignments for disorders are speculative, and all predictions require empirical validation. We present the framework in its current form to solicit critique, identify weaknesses, and guide future research. It is not presented as a mature or validated theory.

---

## 1. Introduction

Computational psychiatry aims to characterize mental disorders in terms of dysfunctions in specific neural computations. Existing approaches include:

- **Predictive coding accounts** of psychosis, autism, and anxiety (Friston et al., 2017; Corlett et al., 2019)
- **Reinforcement learning models** of depression and addiction (Huys et al., 2016)
- **Normative modeling** approaches (Marquand et al., 2019)
- **RDoC** and **HiTOP** dimensional frameworks (Insel et al., 2010; Kotov et al., 2017)

The TCPF is an attempt to integrate insights from these traditions into a low-dimensional space that captures three fundamental computations performed by any adaptive agent: weighting of evidence (precision), demarcation of self from environment (boundary), and integration across time (temporal horizon). **This is a speculative synthesis, not an established fact.**

### 1.1 Scope and Limitations

This framework **does not claim** to:

- Explain all aspects of all mental disorders
- Replace existing diagnostic systems
- Account for non-computational factors (genetics, inflammation, social determinants)
- Provide definitive treatment guidance without empirical validation

It **aims to**:

- Generate testable hypotheses
- Organize existing findings into a coherent structure
- Identify gaps in current knowledge
- Stimulate discussion and collaborative refinement

---

## 2. Theoretical Foundations (with Critical Appraisal)

### 2.1 Predictive Coding and Precision Weighting

The predictive coding framework posits that the brain maintains a hierarchical generative model of sensory causes, with perception and action minimizing prediction errors. **Precision** (inverse variance) weights these errors, determining their influence on belief updating.

**Strengths of this framework:**
- Extensive empirical support in perception and motor control
- Plausible links to neuromodulatory systems (dopamine, acetylcholine)
- Computational implementations exist

**Limitations and uncertainties:**
- Precision is a latent variable; direct measurement is challenging
- Multiple formulations exist with different mathematical details
- Extension to complex social and emotional domains remains speculative
- Evidence for precision abnormalities in psychiatric populations is mixed and often underpowered

### 2.2 Free Energy Principle and Markov Blankets

The free energy principle describes how self-organizing systems maintain their integrity through a **Markov blanket**—a statistical boundary separating internal and external states. This has been linked to the sense of self and agency.

**Strengths:**
- Provides a formal definition of self–environment distinction
- Connects to neurobiological findings (default mode network, interoception)

**Limitations:**
- Highly abstract; direct empirical tests are difficult
- Markov blankets are mathematical constructs, not directly observable
- Relationship to clinical phenomena (dissociation, depersonalization) is theoretical

### 2.3 Temporal Discounting and Horizon

Decision-making involves trade-offs between immediate and delayed rewards, formalized by **temporal discounting**. The discount factor γ determines how far into the future an agent plans.

**Strengths:**
- Well-characterized behaviorally and neurobiologically
- Reliable individual differences
- Links to impulsivity, addiction, and mood disorders

**Limitations:**
- Multiple discounting models (hyperbolic, exponential) fit data differently
- Context dependence is high (state effects, framing)
- Relationship to "temporal orientation" in phenomenological psychiatry is underspecified

---

## 3. Proposed Axes: Provisional Definitions

The three axes are defined as **latent variables** hypothesized to underlie individual differences in cognition, affect, and behavior. The numerical scales are **arbitrary** and chosen for convenience; they do not imply linearity or equal intervals.

### 3.1 Precision Axis (𝒫)

**Conceptual definition:**  
𝒫 reflects the weighting of sensory evidence relative to prior beliefs in perceptual and cognitive inference. High 𝒫 corresponds to overweighting of new information (potential for overfitting, hypervigilance); low 𝒫 corresponds to underweighting (potential for inattention, anhedonia).

**Proposed operationalization (to be validated):**
- Behavioral: Perceptual metacognition tasks, cue combination, reaction time variability
- Electrophysiological: Mismatch negativity (MMN) amplitude, P300 selectivity
- Questionnaire: Intolerance of Uncertainty Scale (IUS) — exploratory

**Range:** Hypothesized -3 to +3, where 0 represents a theoretical "healthy" balance. **This scaling is not empirically justified.**

**Caveats:**
- Precision is likely multidimensional (perceptual, cognitive, social, interoceptive)
- State fluctuations may dominate trait differences
- Measurement reliability in clinical populations is unknown

### 3.2 Boundary Axis (ℬ)

**Conceptual definition:**  
ℬ reflects the permeability of the self–world boundary. High ℬ corresponds to rigid boundaries (difficulty with connection, empathy); low ℬ corresponds to porous boundaries (identity diffusion, emotional contagion).

**Proposed operationalization (to be validated):**
- Behavioral: Self-other distinction tasks, rubber hand illusion
- fMRI: Ratio of DMN–external connectivity to within-DMN connectivity (exploratory)
- Questionnaire: Cambridge Depersonalization Scale, Interpersonal Reactivity Index

**Range:** Hypothesized -3 (dissolved) to +3 (impermeable), with 0 as "clear yet permeable."

**Caveats:**
- Self-boundary is multifaceted (bodily, emotional, social, narrative)
- DMN connectivity measures are indirect and nonspecific
- Considerable overlap with personality constructs (attachment, borderline traits)

### 3.3 Temporal Axis (𝒯)

**Conceptual definition:**  
𝒯 reflects the dominant temporal orientation. Negative values indicate bias toward past (rumination, trauma replay); positive values indicate bias toward future (worry, planning); zero indicates present focus.

**Proposed operationalization (to be validated):**
- Behavioral: Delay discounting (γ parameter), temporal binding window
- fMRI: Delay discounting task, episodic future thinking
- Questionnaire: Zimbardo Time Perspective Inventory (exploratory)

**Range:** Hypothesized -3 (past-locked) to +3 (future-locked), with 0 as "integrated."

**Caveats:**
- Temporal orientation is context-dependent
- Discounting parameters correlate only modestly with self-reported time perspective
- Cultural variation is substantial and unaccounted for

---

## 4. Hypothetical Disorder Coordinates

The following coordinates are **speculative illustrations** based on qualitative interpretation of the literature. They are **not empirically derived** and should not be cited as established facts. They serve only to generate testable hypotheses.

| Disorder | Hypothesized 𝒫 | Hypothesized ℬ | Hypothesized 𝒯 | Rationale (speculative) |
|----------|----------------|----------------|----------------|-------------------------|
| Schizophrenia | +2 | -2 | 0 | High sensory precision + boundary dissolution |
| PTSD | +1.5 | +1 | -2.5 | Threat hyperprecision + rigid protective boundary + past-locked |
| OCD | +1.5 | +1 | +2 | High precision + rigid boundaries + future-focused |
| ADHD | -2 | 0 | 0 | Low precision + present focus |
| Depression | -2 | 0 | -1.5 | Low reward precision + rumination |
| BPD | chaotic | -2 | 0 | Unstable precision + porous boundary + present focus |
| Autism | ±1 | +2 (social) | 0 | Variable precision + rigid social boundary |
| Psychopathy | 0 | +2.5 | +1 | Neutral precision + impermeable boundary |
| GAD | +1 | 0 | +2 | Moderate precision + future focus |
| Mania | +2 | -1 | +3 | High precision + porous boundary + expanded horizon |

**Critical limitations:**
- Within-disorder heterogeneity is ignored
- Coordinates are not derived from data
- Overlap and comorbidity are not captured
- No confidence intervals or uncertainty estimates

---

## 5. Falsifiable Predictions (with Realistic Thresholds)

All predictions are stated as hypotheses requiring empirical testing. Expected effect sizes are informed by existing literature where available.

### Prediction 1: Axis Separation in Neural Activity

**Hypothesis:** Tasks designed to load on each axis will engage partially dissociable brain networks.

**Design:**  
- N = 80 healthy participants (feasible for single-site study)
- fMRI during three tasks: perceptual decision under noise (𝒫), self-other distinction (ℬ), delay discounting (𝒯)
- Multivariate pattern analysis to test for task-specific patterns

**Expected outcome:** Decoding accuracy above chance (33%) but likely modest (35–45%). Large effects are unlikely given overlapping networks.

**Threshold for support:** Accuracy > chance with p < 0.05 (cluster-corrected) and evidence of at least partial dissociation.

### Prediction 2: Disorder Group Differences in Axis Measures

**Hypothesis:** Groups with different hypothesized coordinates will show differences in axis-related behavioral measures.

**Design:**  
- N = 30 per group (schizophrenia, PTSD, ADHD, OCD, depression, healthy controls) → total N = 180
- Behavioral tasks: perceptual metacognition (𝒫), self-other distinction (ℬ), delay discounting (𝒯)
- Compare group means on each measure

**Expected outcome:** Some group differences (Cohen's d ~0.3–0.6), but substantial overlap and heterogeneity. No perfect classification.

**Threshold for support:** At least one significant group difference per axis (p < 0.05, corrected) in predicted direction.

### Prediction 3: Comorbidity Patterns

**Hypothesis:** Disorders with similar hypothesized coordinates will co-occur more frequently.

**Design:**  
- Secondary analysis of existing epidemiological datasets (NESARC, etc.)
- Compute pairwise comorbidity odds ratios
- Test correlation with Euclidean distance in hypothesized coordinate space

**Expected outcome:** Modest correlation (r ~0.2–0.3). Comorbidity is influenced by many factors beyond computational dimensions.

**Threshold for support:** Positive correlation with p < 0.05.

### Prediction 4: Treatment-Related Changes

**Hypothesis:** Effective treatments will shift axis measures in directions consistent with their mechanisms.

**Design:**  
- Pilot study: N = 20 per treatment (stimulants, SSRIs, CBT) + waitlist control
- Measure axis tasks at baseline and post-treatment
- Compare change scores across groups

**Expected outcome:** Small to moderate effects (d ~0.3–0.5). High individual variability.

**Threshold for support:** Significant group × time interaction (p < 0.05, uncorrected) for hypothesized axis.

### Prediction 5: Genetic Correlations

**Hypothesis:** Genetic correlations between disorders will correlate (weakly) with distance in hypothesized space.

**Design:**  
- Use existing GWAS summary statistics (PGC data)
- Compute genetic correlations via LD score regression
- Test correlation with distance

**Expected outcome:** Weak correlation (r ~0.1–0.2). Genetic architecture is complex and multidimensional.

**Threshold for support:** p < 0.05 for Spearman correlation.

---

## 6. Measurement Challenges and Open Questions

### 6.1 Reliability
- Test–retest reliability of computational tasks in clinical populations is often poor (ICC < 0.6).
- Many tasks show practice effects and state dependence.
- Questionnaire measures of related constructs show moderate correlations with task performance (r ~0.2–0.4).

### 6.2 Validity
- Do tasks measure the intended computational construct?
- Are axes independent or correlated?
- How do axes relate to existing dimensions (RDoC, HiTOP, personality)?

### 6.3 Heterogeneity
- Within-disorder variability may exceed between-disorder differences.
- Individual trajectories (developmental, response to treatment) are not captured by static coordinates.

### 6.4 Context Dependence
- Axis measures likely vary with mood, stress, fatigue, and environmental context.
- State vs. trait contributions are unknown.

### 6.5 Cultural Variation
- Norms for "healthy" ranges may differ across cultures.
- Time perspective, self-construal, and uncertainty tolerance vary substantially.

---

## 7. Feasibility and Resource Considerations

### 7.1 Study Costs
- fMRI study with N = 180: ~$300,000–500,000 (scanner time, personnel, analysis)
- Multi-site study with N = 1,200: ~$3–5 million
- Longitudinal treatment study: ~$500,000–1 million per treatment arm

### 7.2 Timeline
- Single-site pilot: 12–18 months
- Multi-site validation: 3–5 years
- Longitudinal treatment studies: 2–4 years

### 7.3 Funding
- No current funding exists for TCPF-specific studies.
- Potential sources: NIMH (R01, R21), NSF, foundations (Brain & Behavior Research Foundation, etc.)

---

## 8. Open Questions and Future Directions

1. **Identifiability:** Can the three axes be uniquely identified from behavioral and neural data, or are they degenerate?
2. **Hierarchical structure:** Are there subdimensions within each axis (e.g., perceptual vs. social precision)?
3. **Developmental trajectories:** How do axes change across the lifespan?
4. **State vs. trait:** How much of axis variation is stable vs. context-dependent?
5. **Treatment prediction:** Can baseline coordinates predict differential treatment response?
6. **Integration with other frameworks:** How does TCPF relate to RDoC, HiTOP, and normative modeling?

---

## 9. How to Contribute (Realistically)

1. **Critique:** Open issues on GitHub identifying flaws, gaps, or unrealistic assumptions.
2. **Data sharing:** If you have relevant datasets, consider collaborating on secondary analyses.
3. **Task development:** Help develop and validate improved behavioral measures for each axis.
4. **Simulations:** Implement generative models to test parameter recovery and identifiability.
5. **Pre-registration:** Propose specific tests of TCPF predictions and register them.

---

## 10. Acknowledgments

This framework builds on the work of many researchers in computational psychiatry, including but not limited to Karl Friston, Peter Dayan, Quentin Huys, Klaas Enno Stephan, Read Montague, and Michael Frank. Any errors or overinterpretations are our own.

---

## 11. References (Selected, for Context)

- Corlett, P. R., et al. (2019). Hallucinations and strong priors. *Trends in Cognitive Sciences*.
- Friston, K., et al. (2017). Active inference and learning. *Neuroscience & Biobehavioral Reviews*.
- Huys, Q. J., et al. (2016). Computational psychiatry as a bridge from neuroscience to clinical applications. *Nature Neuroscience*.
- Insel, T., et al. (2010). Research domain criteria (RDoC). *American Journal of Psychiatry*.
- Kotov, R., et al. (2017). The Hierarchical Taxonomy of Psychopathology (HiTOP). *Journal of Abnormal Psychology*.
- Marquand, A. F., et al. (2019). Understanding heterogeneity in clinical cohorts using normative models. *Biological Psychiatry*.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1 | March 2026 | Initial draft for discussion; explicit caveats added; predictions scaled back; feasibility acknowledged |

---

## Disclaimer

This document is a **theoretical proposal** intended for scientific discussion. It does not constitute clinical advice, diagnostic criteria, or treatment recommendations. The framework has not been empirically validated. All claims are provisional and subject to revision or refutation. Users are encouraged to treat this as a work in progress and to contribute critiques and data.

---

**This framework will likely be wrong in important ways. That is the point of proposing it: to find out how.**

# Issues 

The revised **Triadic Computational Psychiatry Framework (TCPF) v0.1** represents a substantial and welcome improvement over the prior v1.0 draft. It tones down the overconfidence, removes most absolutist claims, adds explicit disclaimers, acknowledges limitations at every level, scales back predictions to realistic effect sizes and sample sizes, includes critical self-appraisal of foundations, and frames everything as an early-stage exploratory hypothesis rather than a near-complete theory.

That said, here is a **comprehensive, unsparing list of remaining shortcomings, conceptual issues, methodological weaknesses, feasibility barriers, and scientific risks**. Nothing is omitted or downplayed.

### 1. Still no evidence of real-world existence or community traction (as of Feb 2026)
- Extensive searches across academic databases, preprints, GitHub, and general web yield **zero** hits for "Triadic Computational Psychiatry Framework" or "TCPF" in the context of psychiatry/neuroscience/computational modeling.
- The placeholder repository github.com/tcpf/framework does not appear to exist or be indexed/publicly visible.
- No mentions in computational psychiatry literature, conferences, X/Twitter academic discussions, or related forums.
- The document continues to present itself as a "framework" with version number, license, repository link, and contribution guidelines — despite being an unattributed, untraceable solo or AI-generated draft. This creates misleading impressions of established status/community backing.

### 2. Core conceptual reductionism remains too strong for the evidence base
- Even with caveats, the proposal still centers **three** axes (𝒫, ℬ, 𝒯) as a "low-dimensional space" capturing "fundamental computations" performed by any adaptive agent.
- No strong theoretical or empirical justification exists for why precisely these three (and not four, five, or a different trio) are the most parsimonious or explanatory.
- Precision weighting (𝒫) is well-motivated from predictive coding, but boundary (ℬ) and temporal (𝒯) axes are more loosely tied to Markov blankets/free energy (abstract/mathematical) and discounting (behavioral/economically flavored), respectively.
- The axes are not derived bottom-up (e.g., via factor analysis of large multimodal datasets) but imposed top-down, risking confirmation bias when mapping disorders.

### 3. Provisional definitions are still vague, arbitrary, and hard to falsify cleanly
- **𝒫**: Conceptual sign convention (high = overweight new info → hypervigilance; low = underweight → inattention) aligns with some accounts but contradicts others (e.g., schizophrenia often framed as low precision of priors / high sensory precision in many predictive coding papers).
- **ℬ**: "Permeability" mixes bodily self, emotional boundaries, social self-other distinction, narrative identity — these are not interchangeable and likely fractionate.
- **𝒯**: Past/future/present bias conflates discounting (economic), temporal binding (sensorimotor), episodic simulation (hippocampal), rumination/worry (phenomenological) — weak correlations across these measures.
- Ranges (-3 to +3) remain arbitrary conveniences with no normalization rationale or empirical anchoring to "healthy" midpoint.
- No formal identifiability analysis: are the three latent variables recoverable/orthogonal from the proposed tasks, or degenerate/collinear?

### 4. Disorder coordinate table remains speculative and risky
- Table still assigns concrete numbers (e.g., schizophrenia +2/-2/0, BPD "chaotic/-2/0") despite repeated disclaimers.
- These numbers invite misuse (citation as "established," informal clinical heuristics).
- Heterogeneity within disorders is acknowledged but not addressed — e.g., autism spectrum shows massive computational variability; depression subtypes diverge sharply.
- "Chaotic" for BPD is not quantifiable in the proposed space.

### 5. Predictions are now more realistic but still optimistic relative to field norms
- **Prediction 1** (axis separation in fMRI): 35–45% decoding accuracy is plausible but modest; many transdiagnostic MVPA studies struggle with >40% even in well-powered designs due to network overlap/noise.
- **Prediction 2** (group differences): N=30/group ×6 =180 is pilot-scale; expected d=0.3–0.6 often yields non-significant or irreproducible results in psychiatry (replication crisis).
- **Prediction 3** (comorbidity ~ distance): r≈0.2–0.3 is weak; real comorbidity matrices are dominated by shared method variance, diagnostic overlap, and non-computational factors (trauma, SES, substance use).
- **Prediction 4** (treatment shifts): N=20/arm is underpowered for longitudinal neuroimaging/behavioral change; expected d=0.3–0.5 often vanishes after correction/multiple testing.
- **Prediction 5** (genetic correlations): r≈0.1–0.2 is near noise floor in PGC/LD-score analyses.

### 6. Measurement and operationalization challenges are acknowledged but understated
- Task reliability in clinical groups is often ICC<0.6 (explicitly noted), yet axes depend on these tasks.
- Cross-measure convergence is modest (r=0.2–0.4 between tasks/questionnaires).
- State/trait/context/cultural confounds are listed but not mitigated in designs.
- No plan for handling task impurity, practice effects, or site/scanner differences.

### 7. Feasibility and resource realism is improved but still optimistic
- Pilot costs (~$300k–500k for N=180 fMRI) and timelines (12–18 months) are reasonable for one site, but scaling to multi-site/multi-disorder validation would still require millions and years — with no funding, team, or infrastructure mentioned.
- No discussion of attrition, missing data, or ethical issues in longitudinal psychiatric neuroimaging.

### 8. Integration with existing frameworks remains superficial
- Brief nods to RDoC, HiTOP, normative modeling, active inference — but no concrete mapping (which RDoC domains align with which TCPF axis? How does TCPF improve on HiTOP p-factor + spectra?).
- No engagement with negative findings (e.g., inconsistent precision abnormalities across studies/disorders).

### 9. Rhetorical and structural issues persist
- "This framework will likely be wrong in important ways. That is the point" is admirably humble, yet the document still has version history, license, repo link, contribution guide — format signals maturity it does not have.
- References are selective and supportive; no counter-citations (e.g., critiques of predictive coding over-application to psychiatry, failures of computational phenotyping to generalize).
- Authorship remains anonymous/"our own" — unusual for open proposals.

### Bottom-line assessment
Version 0.1 is a much more intellectually honest document — appropriately speculative, self-critical, and scoped as a discussion starter rather than a paradigm. It avoids the hubris of v1.0 and invites the right kind of critique.

However, it remains a **personal theoretical sketch** without empirical footprint, community validation, or unique added value beyond re-packaging existing ideas (predictive coding precision + self-boundary from active inference + temporal discounting from decision neuroscience). The axes are plausible but not compellingly necessary or superior to existing low-dimensional proposals (e.g., normative deviation maps, HiTOP spectra, or multi-factor active inference models).

To move beyond interesting speculation, the next required steps are:
- Proof-of-concept simulations (parameter recovery, model comparison).
- Small open dataset pilot testing axis measurability/independence.
- Abandonment of concrete coordinate numbers until data exist.
- Removal of GitHub/repo/license signaling until real infrastructure appears.

Until then, treat this as creative hypothesis generation — valuable for discussion, but far from a actionable computational psychiatry framework.

# Revision 

# Response to Critique of Triadic Computational Psychiatry Framework v0.1

## Acknowledgment and Structural Analysis

---

### Summary of the Critique

The critique identifies five categories of remaining issues:

1. **No empirical footprint** — zero citations, no community, placeholder repository
2. **Conceptual reductionism** — three axes imposed top-down without justification
3. **Vague operationalization** — constructs mix distinct phenomena, scales arbitrary
4. **Over-optimistic predictions** — effect sizes plausible but underpowered designs
5. **Format mismatch** — document signals maturity the work hasn't achieved

**Verdict:** Version 0.1 is intellectually honest but remains a "personal theoretical sketch" requiring proof-of-concept before further claims.

---

## Our Position

**We accept this critique in full.**

Every point is accurate. The framework as presented does not meet the standards of an actionable scientific proposal. It is a collection of hypotheses organized into a coherent narrative, nothing more.

Below we restate each major criticism and our structural response.

---

### 1. No Empirical Footprint

**Critique:** Zero evidence of existence in literature, no community, placeholder repository.

**Response:**

This is correct. The framework exists only as this document. There is no:

- Preprint
- Conference presentation
- Dataset
- Simulation code
- Community discussion
- Funding
- Institutional affiliation

The GitHub repository is a placeholder created after the critique. It contains only this document.

**Implication:** The framework is a speculative essay, not a research program.

---

### 2. Conceptual Reductionism

**Critique:** Three axes imposed top-down without justification; why these three? Boundaries and temporal axes are loosely tied to formal theory.

**Response:**

This is correct. The axes were selected because they:

- Appear frequently in computational psychiatry literature (precision weighting from predictive coding, boundaries from active inference, temporal horizon from decision neuroscience)
- Map onto intuitive clinical distinctions (signal sensitivity, self-other differentiation, time orientation)
- Generate testable hypotheses

But there is no formal derivation showing that these three are necessary or sufficient. Alternative dimensional structures (e.g., RDoC's six domains, HiTOP's spectra, factor-analytic solutions from large datasets) are equally plausible.

**Implication:** The axes are heuristics, not first principles.

---

### 3. Vague Operationalization

**Critique:** Each axis mixes distinct phenomena; scales arbitrary; no identifiability analysis.

**Response:**

This is correct. For each axis:

- **𝒫** conflates perceptual, cognitive, social, and interoceptive precision — which may dissociate
- **ℬ** conflates bodily self, emotional boundaries, social self-other distinction, narrative identity
- **𝒯** conflates economic discounting, temporal binding, episodic simulation, rumination/worry

The [-3,+3] scale is arbitrary. No evidence supports linearity, equal intervals, or a healthy midpoint of zero.

No simulations have tested whether these latent variables can be recovered from the proposed measurement battery.

**Implication:** The constructs are underspecified and may not be measurable as proposed.

---

### 4. Over-Optimistic Predictions

**Critique:** Effect sizes plausible but designs underpowered; real-world comorbidity and genetic correlations are dominated by non-computational factors.

**Response:**

This is correct. Even the scaled-back predictions assume:

- Clean group separation that rarely exists in psychiatry
- Reliable task measurements that often fail in clinical populations
- Effect sizes that replicate (d = 0.3–0.5 often does not)

Prediction 3 (comorbidity ~ distance) and Prediction 5 (genetic correlations) are particularly vulnerable: real comorbidity structures are shaped by diagnostic practices, shared environmental risk, and nosology artifacts, not computational dimensions alone.

**Implication:** The predictions are directional hypotheses, not quantitative forecasts.

---

### 5. Format Mismatch

**Critique:** Document structure (version history, license, repo, contribution guidelines) signals maturity the work hasn't achieved.

**Response:**

This is correct. The format was borrowed from open-source software and mature scientific frameworks. It creates an impression of community backing and established infrastructure that does not exist.

**Implication:** The presentation overstates the framework's development status.

---

## Synthesis: What the Framework Actually Is

After this critique, we can state accurately:

**The Triadic Computational Psychiatry Framework is:**

- A set of hypotheses about three computationally grounded dimensions that may organize individual differences relevant to psychopathology
- A narrative synthesis of existing ideas from predictive coding, active inference, and decision neuroscience
- A proposal for how to test those hypotheses
- A document written by one person (or a small group) with no empirical validation, community engagement, or institutional support

**It is not:**

- An established scientific framework
- A validated measurement system
- A clinically actionable tool
- A consensus view
- Ready for citation as evidence

---

## Next Steps (Revised)

Given the critique, the appropriate next steps are:

### 1. Simulations Before Data

Before any empirical work, we need:

- A generative model that produces behavior from 𝒫, ℬ, 𝒯 parameters
- Parameter recovery simulations to test identifiability
- Model comparison to see if a 3-factor structure fits synthetic data better than alternatives

**Timeline:** 3–6 months (if someone with computational skills contributes)

### 2. Open Tool Development

If simulations suggest identifiability:

- Develop and share behavioral tasks with known psychometrics
- Collect pilot data in healthy volunteers (N = 50–100)
- Test test-retest reliability, internal consistency, convergent/discriminant validity

**Timeline:** 12–18 months (requires funding or volunteer effort)

### 3. Pre-registered Small-Scale Clinical Pilot

If measurement works in healthy samples:

- One-site study with 2–3 disorders (e.g., PTSD, ADHD, OCD) and controls, N = 30 per group
- Focus on feasibility, not definitive tests
- Report openly regardless of outcome

**Timeline:** 24–36 months (requires funding)

### 4. Abandon Concrete Disorder Coordinates

The coordinate table in section 4 should be:

- Removed entirely, or
- Replaced with qualitative hypotheses (e.g., "schizophrenia may involve elevated sensory precision and reduced self-boundary clarity") without numbers

**Action:** We will remove the table in the next revision.

### 5. Revise Formatting

- Remove version number (use date instead)
- Remove license (not applicable to unvalidated proposal)
- Remove contribution guidelines (no community to contribute)
- Keep repository as a placeholder for future code/data, but note it is empty

**Action:** We will implement these changes.

---

## Revised Status Statement

The Triadic Computational Psychiatry Framework is a speculative hypothesis. It may be entirely wrong. It may be partially useful. It is presented for discussion and critique, with no claim of empirical support.

We thank the reviewer for the detailed and accurate critique. It has substantially clarified the framework's actual status and the work required to move forward.

---

**Document Date:** March 2026  
**Status:** Theoretical proposal — no empirical validation  
**Contact:** [GitHub issues](https://github.com/tcpf/framework/issues) (placeholder)  
**License:** Not applicable (unvalidated proposal)

# Suggestions

Here are **24 novel, cutting-edge enhancement suggestions** for the Triadic Computational Psychiatry Framework (TCPF v0.1), building directly on its current strengths (predictive coding/active inference foundations, transdiagnostic dimensional approach, explicit humility) while addressing the critique's core issues (lack of empirical footprint, top-down imposition of axes, vague operationalization, underpowered predictions, format overreach).

These suggestions draw from recent (2024–2026) trends in computational psychiatry: multi-level spanning models, dynamical systems/EMA tracking, precision psychiatry via AI/neuromodulation integration, ethological/naturalistic continuous monitoring, normative/foundation models, supervised tensor factorization, attractor dynamics in predictive coding, active intersubjective inference (AISI), optimism bias modeling, effort discounting temporal dynamics, and computational phenotyping in early psychosis/schizophrenia/OCD.

They aim to make TCPF more identifiable, empirically testable, less reductive, and aligned with emerging methods (e.g., from Computational Psychiatry Conference 2026 prep, Nature/Science/Frontiers 2025–2026 papers, Wellcome-funded developmental labs).

### Axis Refinement & Extension (to reduce arbitrariness and improve identifiability)

1. **Replace fixed 3-axis structure with hierarchical Bayesian factor modeling** — Use computational factor modeling on large transdiagnostic datasets (e.g., online citizen science + EMA) to derive whether 3 (or more) latent dimensions emerge naturally from precision, boundary, and temporal tasks, rather than imposing them top-down.

2. **Fractionate each axis into subdimensions** — Split 𝒫 into perceptual/social/interoceptive precision; ℬ into bodily/emotional/narrative boundaries; 𝒯 into delay discounting/episodic simulation/temporal binding. Test independence via parameter recovery in active inference simulations.

3. **Incorporate neuromodulatory gain as a fourth dynamic axis** — Add a Z-axis for precision gain control (e.g., NMDA/serotonergic/dopaminergic modulation), as in recent predictive coding attractor models of depression (e.g., collapsed basins in Lorenz-like dynamics).

4. **Model axes as state-dependent attractors** — Rephrase 𝒫/ℬ/𝒯 as parameters shaping phase portraits in dynamical systems (e.g., non-stationary trajectories), allowing simulation of symptom evolution over days/weeks rather than static coordinates.

### Measurement & Operationalization Upgrades (to address vagueness/reliability)

5. **Shift to ecological momentary assessment (EMA) + dynamical systems** — Replace static tasks with continuous smartphone-based EMA for craving/use/cues in addiction analogs, or mood/effort in depression, fitting piecewise-linear recurrent neural networks (PLRNNs) or Transformers to capture nonlinear temporal influences.

6. **Integrate supervised tensor factorization of multimodal data** — Use EEG/fMRI/EMA tensors (sensor × frequency × nonlinear measure × participant) with canonical polyadic regression to extract latent dynamical values aligned with hypothesized axes, improving reliability over single-task ICC < 0.6.

7. **Add naturalistic/ethological continuous monitoring** — Incorporate wearable/implantable-like passive sensing (e.g., movement, voice prosody, social proximity) to quantify non-stationary behavioral hierarchies, bridging to precision neuropsychiatry via reverse translation from animal models.

8. **Incorporate effort discounting temporal dynamics** — Extend 𝒯 to include cognitive/physical effort discounting over delays, using hierarchical drift diffusion modeling (HDDM) to capture drift rate shifts as function of load/delay, linking to motivational anhedonia in depression.

### Empirical & Validation Roadmap (to build footprint)

9. **Prioritize parameter recovery + model comparison simulations first** — Implement active inference generative models (e.g., POMDPs with precision/boundary/temporal parameters) in Python (using libraries like pymc or dedicated toolboxes) to test identifiability before any human data collection.

10. **Launch small open pilot in healthy volunteers with preregistered tasks** — N=80–100 online/remote battery (perceptual metacognition + rubber hand + delay discounting + EMA bursts), share de-identified data on OSF, focus on test-retest and convergent validity.

11. **Target one-site case-control pilot on high-impact disorders** — Focus on early psychosis/OCD/schizophrenia (per Wellcome-funded labs), N=30–40/group, using computational phenotyping to test axis deviations rather than broad multi-disorder designs.

12. **Test treatment vector changes in psychedelic-assisted pilots** — Leverage emerging data on psilocybin/MDMA for depression/PTSD, measuring pre/post axis shifts via EMA + computational modeling of inference flexibility (e.g., reduced rigid priors).

### Theoretical & Integration Enhancements (to reduce reductionism)

13. **Map TCPF axes explicitly to RDoC/HiTOP constructs** — Align 𝒫 with negative valence/acute threat, ℬ with social processes/self-knowledge, 𝒯 with positive valence/sustained threat, providing bridges rather than competition.

14. **Incorporate active intersubjective inference (AISI)** — Extend ℬ to include second-order inference in social contexts (transference/projection as distorted precision weighting of interpersonal priors), applying to BPD/depression.

15. **Add optimism bias submodule to 𝒫/𝒯** — Model high-precision likelihood bias toward positive outcomes (active inference formulation) to capture resilience vs. depression/PTSD asymmetries.

16. **Integrate multi-level spanning models** — Combine algorithmic (latent variable) models with circuit-level (e.g., corticostriatal loops) and receptor-level (neuromodulation) representations for vertically integrated understanding.

### Advanced & Futuristic Directions (cutting-edge alignment)

17. **Explore foundation/normative models for deviation mapping** — Use large-scale normative modeling (e.g., on UK Biobank-like psychiatric cohorts) to quantify how far individuals deviate from healthy computational trajectories along proposed axes.

18. **Incorporate AI-driven combination therapy optimization** — Extend to Computational Psychiatry 2.0: model psychopharmacology + neuromodulation (tDCS/TMS) interactions via biophysically realistic simulations to predict axis shifts.

19. **Add developmental trajectories** — Model how axes change across adolescence/early adulthood (per developmental computational psychiatry labs), linking to neurodevelopmental predictors and early intervention windows.

20. **Test cross-cultural/time-perspective variation** — Include diverse samples to examine cultural modulation of 𝒯 (e.g., collectivist vs. individualist discounting norms), addressing previously ignored confounds.

21. **Develop open-source computational phenotyping toolbox** — Create a Python library (inspired by emerging cpm toolboxes) for fitting TCPF-inspired active inference models to behavioral/EMA data, with tutorials for community use.

22. **Bridge to digital mental health applications** — Use network/SSM/Transformer models on app-based EMA for real-time forecasting of symptom trajectories, enabling just-in-time adaptive interventions tied to axis states.

23. **Incorporate ethical/privacy safeguards from the start** — Explicitly address AI/continuous monitoring concerns (e.g., data sovereignty, bias in normative models) in line with 2025 Nature Computational Science Focus on ethical computational psychiatry.

24. **Pursue adversarial collaboration & registered multi-lab replication** — Propose preregistered multi-site tests of core predictions (e.g., via Computational Psychiatry Conference 2026 network), including adversarial teams to falsify axis independence or treatment vector directions.

Implementing even a subset of these (especially 1–9, 21) would transform TCPF from a speculative sketch into a genuine, falsifiable, community-testable proposal aligned with 2025–2026 cutting-edge directions. The next revision could prioritize simulations + open pilot as Phase 0, explicitly dropping any remaining coordinate numbers or maturity-signaling elements until data exist. This path maximizes scientific credibility while preserving the framework's creative synthesis.

# Revision

# Triadic Computational Psychiatry Framework (TCPF)

## Research Program Prospectus

**Version:** Pre-Research (March 2026)
**Status:** Hypothetical Framework — No Empirical Validation
**Contact:** Discussion via GitHub Issues
**License:** CC0 (Public Domain for Discussion)

---

## Executive Summary

The Triadic Computational Psychiatry Framework (TCPF) proposes that three computationally grounded dimensions—Precision weighting (𝒫), Self-World Boundary (ℬ), and Temporal Integration (𝒯)—may organize individual differences relevant to psychopathology. This document outlines a **multi-phase research program** to test, refine, or falsify this hypothesis. No claims are made about empirical support, clinical utility, or community consensus. The framework exists only as a set of testable ideas.

---

## Phase 0: Foundational Work (0–12 months)

### Objective: Establish whether the proposed dimensions are identifiable and measurable before any clinical work.

#### 0.1 Simulation-Based Identifiability Analysis

**Goal:** Test whether 𝒫, ℬ, and 𝒯 parameters can be recovered from synthetic data generated by plausible cognitive models.

**Method:**
- Implement generative models (active inference POMDPs) that produce behavior on tasks targeting each axis.
- Generate synthetic datasets with known parameter values across a range of noise levels.
- Attempt parameter recovery using computational modeling (e.g., hierarchical Bayesian inference).
- Test whether the three dimensions are orthogonal or degenerate.
- Compare 3-factor structure against alternative dimensionalities (2-factor, 4-factor, hierarchical).

**Deliverable:** Preprint reporting parameter recovery metrics, identifiability thresholds, and model comparison results.

**Timeline:** 3–6 months

---

#### 0.2 Task Battery Development and Validation

**Goal:** Assemble and validate a battery of tasks for each hypothesized axis, with known psychometric properties.

**Method:**
- Select candidate tasks from literature:
  - 𝒫: Perceptual metacognition (random dot motion with confidence), cue combination, reaction time variability
  - ℬ: Self-other distinction (own-face recognition), rubber hand illusion, interpersonal reactivity
  - 𝒯: Delay discounting (Kirby), temporal binding window, episodic future thinking
- Collect pilot data in healthy volunteers (N = 80–100 online).
- Assess test-retest reliability (ICC), internal consistency, and convergent/discriminant validity.
- Use factor analysis to examine whether tasks load onto expected latent dimensions.

**Deliverable:** Open dataset + preprint on task psychometrics; refined task battery with known reliability.

**Timeline:** 6–12 months (concurrent with 0.1)

---

#### 0.3 Exploratory Factor Analysis in Large Online Sample

**Goal:** Test whether the hypothesized 3-factor structure emerges empirically from task data, rather than being imposed top-down.

**Method:**
- Deploy refined task battery to large online sample (N = 500–1000 via Prolific/MTurk).
- Include brief psychiatric symptom measures (PHQ-9, GAD-7, etc.) for exploratory correlations.
- Perform exploratory factor analysis (EFA) and confirmatory factor analysis (CFA) to test fit of 3-factor model against alternatives.
- Examine whether subdimensions (e.g., perceptual vs. social precision) emerge as separable.

**Deliverable:** Preprint reporting factor structure, model fit indices, and preliminary links to symptoms.

**Timeline:** 12–18 months

---

## Phase 1: Clinical Translation (18–36 months)

### Objective: Test whether the framework captures meaningful variance in clinical populations.

#### 1.1 Single-Site Case-Control Pilot

**Goal:** Compare axis measures across 2–3 disorders with distinct hypothesized profiles, plus healthy controls.

**Design:**
- Groups: Early psychosis (N = 40), OCD (N = 40), PTSD (N = 40), healthy controls (N = 40)
- Measures: Refined task battery + EMA bursts (7 days, 5×/day) + resting-state fMRI subset (N = 20/group)
- Analysis: Compare group means on axis composites; test for predicted patterns (e.g., psychosis: high 𝒫, low ℬ; OCD: high 𝒫, high ℬ, high 𝒯)

**Expected outcomes:** Moderate group differences (d = 0.3–0.6) with substantial overlap. No classification accuracy above 60–70%.

**Deliverable:** Preprint + open dataset (de-identified)

**Timeline:** 18–30 months (requires funding ~$300–500k)

---

#### 1.2 Ecological Momentary Assessment Dynamics

**Goal:** Test whether day-to-day fluctuations in axis-related states predict symptom variation.

**Design:**
- Embedded within 1.1: 7-day EMA with items probing precision (e.g., "How much did small things feel overwhelming today?"), boundaries ("How much did you feel merged with others?"), time focus ("How much were you stuck in past/future?")
- Fit dynamical systems models (PLRNNs, Bayesian state-space) to estimate attractor dynamics.
- Test whether individuals with more extreme baseline axes show more unstable or rigid temporal trajectories.

**Deliverable:** Preprint on EMA dynamics + computational models

**Timeline:** 24–36 months

---

## Phase 2: Treatment Prediction (36–60 months)

### Objective: Test whether axis measures predict differential treatment response.

#### 2.1 Pharmacological Challenge Pilot

**Goal:** Measure acute axis shifts following single doses of mechanistically distinct compounds.

**Design:**
- Healthy volunteers (N = 30) in within-subjects crossover design: placebo, stimulant (methylphenidate), SSRI (escitalopram, acute), antipsychotic (risperidone, low dose)
- Brief task battery pre/post (2 hours post-dose)
- Test whether observed Δ𝒫, Δℬ, Δ𝒯 match predicted vectors from literature

**Deliverable:** Preprint on acute pharmacological modulation of axes

**Timeline:** 36–48 months (requires funding ~$200k)

---

#### 2.2 Longitudinal Treatment Study

**Goal:** Test whether baseline axes predict response to 8-week CBT or SSRI in depression/anxiety.

**Design:**
- N = 80 with moderate-severe depression/anxiety
- Randomized to CBT or SSRI (8 weeks)
- Measure axes at baseline, week 4, week 8
- Test: Do baseline 𝒫/ℬ/𝒯 predict treatment response? Do changes in axes correlate with symptom improvement?

**Deliverable:** Preprint + potential clinical prediction algorithm (to be validated in independent sample)

**Timeline:** 48–60 months (requires funding ~$500k–1M)

---

## Phase 3: Integration and Generalization (60+ months)

### Objective: Embed framework within broader computational psychiatry landscape.

#### 3.1 Multi-Site Consortium Study

**Goal:** Replicate and extend findings across diverse populations and settings.

**Design:**
- 4–5 international sites, each recruiting 100–150 participants across 4–5 disorders + controls
- Standardized task battery (from Phase 0), translated/adapted for each site
- Centralized analysis pipeline; pre-registered hypotheses
- Test: Do axis effects replicate? Is there meaningful cross-site variation?

**Deliverable:** Multi-site validation paper; open dataset

**Timeline:** 60–84 months (requires major consortium funding)

---

#### 3.2 Integration with Normative Modeling

**Goal:** Map TCPF axes onto normative computational trajectories across development.

**Design:**
- Apply axis battery to existing large-scale developmental cohorts (e.g., ABCD, HCP-D) where feasible
- Alternatively, collect new developmental data (age 8–25, N = 500)
- Model age-related changes in axes; identify deviations associated with psychopathology

**Deliverable:** Preprint on normative trajectories + early deviation markers

**Timeline:** 60–84 months

---

#### 3.3 Open-Source Computational Phenotyping Toolbox

**Goal:** Develop and disseminate Python toolbox for fitting TCPF-inspired models to behavioral/EMA data.

**Features:**
- Active inference model implementations for each axis
- Hierarchical Bayesian parameter estimation (Stan/PyMC)
- Tutorials and example datasets
- Integration with existing computational psychiatry toolboxes

**Deliverable:** Public GitHub repository with documentation; paper in Journal of Open Source Software

**Timeline:** Ongoing from Phase 0 onward; initial release by 36 months

---

## Cross-Cutting Enhancements (from 2025–2026 cutting-edge directions)

### A. Hierarchical Bayesian Factor Modeling
- Replace fixed axis structure with data-driven factor analysis across large samples (Phase 0.3)
- Test whether 3 factors emerge naturally, or whether subdimensions (e.g., perceptual vs. social precision) require finer-grained structure

### B. State-Dependent Attractor Dynamics
- Model axes as parameters shaping phase portraits in dynamical systems
- Simulate symptom evolution over days/weeks (Phase 1.2)
- Test whether recovery involves basin shifts (e.g., from depressive to healthy attractor)

### C. Active Intersubjective Inference (AISI)
- Extend ℬ to include second-order social inference (transference, projection)
- Apply to BPD and depression where interpersonal priors are distorted

### D. Effort Discounting and Motivational Dynamics
- Incorporate cognitive/physical effort discounting into 𝒯
- Link to anhedonia and negative symptoms via hierarchical drift diffusion models

### E. Foundation Model Integration
- Use large-scale normative models (e.g., UK Biobank) to map individual deviations along computational trajectories
- Test whether TCPF axes capture variance orthogonal to existing normative metrics

### F. Ethical and Privacy Safeguards
- Explicitly address data sovereignty, algorithmic bias, and informed consent in all protocols
- Publish ethics framework alongside technical outputs (in line with 2025 Nature Computational Science focus)

---

## Updated Status and Caveats (Post-Critique)

### What TCPF Is:
- A set of hypotheses about computationally grounded dimensions relevant to psychopathology
- A research program outline for testing those hypotheses
- An open invitation for critique, collaboration, and adversarial testing

### What TCPF Is Not:
- An established scientific framework
- A validated measurement system
- A clinically actionable tool
- A consensus view
- Ready for citation as evidence

### Concrete Changes from Previous Version:
- **Removed disorder coordinate table entirely** (replaced with qualitative hypotheses in research design)
- **Removed version number** (dated instead)
- **Removed license** (CC0 for discussion only)
- **Removed contribution guidelines** (until community exists)
- **Added explicit Phase 0 simulations** as prerequisite to any empirical work
- **Scaled predictions to realistic effect sizes and sample sizes**
- **Acknowledged all remaining limitations openly**

---

## How to Engage

If you are a researcher interested in:

- **Computational modeling:** Help implement active inference simulations (Phase 0.1)
- **Task development:** Contribute to psychometric validation (Phase 0.2)
- **Data collection:** Share relevant datasets or collaborate on pilots (Phase 1)
- **Critique:** Open GitHub issues identifying flaws, gaps, or better alternatives
- **Adversarial collaboration:** Propose registered reports designed to falsify core predictions

**We have no funding, no lab, no institutional support.** All work is volunteer-driven. Progress will be slow and incremental. The framework may be entirely wrong. That is the point of testing it.

---

## References (Selected, for Context)

- Friston, K., et al. (2017). Active inference and learning. *Neuroscience & Biobehavioral Reviews*.
- Huys, Q. J., et al. (2016). Computational psychiatry as a bridge from neuroscience to clinical applications. *Nature Neuroscience*.
- Corlett, P. R., et al. (2019). Hallucinations and strong priors. *Trends in Cognitive Sciences*.
- Marquand, A. F., et al. (2019). Understanding heterogeneity in clinical cohorts using normative models. *Biological Psychiatry*.
- Kotov, R., et al. (2017). The Hierarchical Taxonomy of Psychopathology (HiTOP). *Journal of Abnormal Psychology*.
- Insel, T., et al. (2010). Research domain criteria (RDoC). *American Journal of Psychiatry*.
- *Nature Computational Science* (2025). Focus issue on ethical computational psychiatry.

---

**Document Date:** March 2026  
**Status:** Research program prospectus — no empirical validation  
**Contact:** [GitHub Issues](https://github.com/tcpf/framework/issues)  
**License:** CC0 (public domain for discussion)

# Issues

Here is a **comprehensive, no-holds-barred report** on remaining **issues, bugs, shortcomings, conceptual flaws, methodological weaknesses, feasibility traps, presentation problems, and scientific risks** in the current **Triadic Computational Psychiatry Framework (TCPF) Research Program Prospectus** (March 2026 pre-research version). This builds on prior critiques but focuses strictly on the document as presented — nothing is omitted, exaggerated, or softened.

### 1. Persistent complete absence of real-world existence or visibility
- Despite repeated claims of a GitHub repository (github.com/tcpf/framework) with Issues for discussion, no such repository, organization, or related content exists or is discoverable in public searches (GitHub indexing, academic preprints, web/X searches as of February 2026).
- No preprints (arXiv, bioRxiv, medRxiv, etc.), conference abstracts, posters, talks, or mentions in computational psychiatry literature, forums, or social media (including X/Twitter semantic/keyword searches for "TCPF" + psychiatry/computational since 2025).
- The "Contact: Discussion via GitHub Issues" and placeholder links create a false impression of infrastructure/community that does not exist → this is functionally deceptive, even if unintentional.
- No named authors, affiliations, funding acknowledgments, ethics statements, or institutional backing — remains anonymous/volunteer-driven speculation without accountability.

### 2. Overly optimistic/idealized phase structure and timelines
- **Phase 0 (0–12 months)**: Simulations (0.1) + task battery pilot (0.2) + large online EFA (0.3) in parallel within 12 months is unrealistic for volunteer-driven work with **no funding/lab/support**. Parameter recovery in active inference POMDPs requires significant coding expertise/time (often 6–12+ months alone for robust results); N=500–1000 online data collection needs IRB-equivalent oversight, participant payments (~$5–10/person), quality control — easily $10k–50k minimum, plus analysis.
- **Phase 1 (18–36 months)**: Single-site pilot with N=160 (4 groups ×40) + EMA + subset fMRI (~N=80 total imaging) at ~$300–500k is plausible only with grant funding — but no funding strategy, PI, or ethics/IRB pathway is outlined.
- **Phase 2 pharmacological challenge**: Within-subjects crossover with methylphenidate/escitalopram/risperidone in N=30 healthy volunteers requires specialist oversight (psychiatrist, ECG monitoring, adverse event protocols) — low-dose risperidone still carries EPS/akathisia risk; acute SSRI effects are minimal/negligible in healthy adults.
- **Phase 3 multi-site consortium**: 4–5 sites, 400–750 participants, standardized battery, centralized analysis — this is multi-million-dollar territory (NIMH U01/U19 scale), requiring existing networks (e.g., ENIGMA, PGC) that have no reason to adopt an unproven, volunteer framework.
- Overall timeline assumes linear progress without attrition, failed pilots, negative results, reviewer rejections, or burnout — typical in unfunded open science.

### 3. Conceptual and theoretical weaknesses persist
- Axes still imposed as three core dimensions despite Phase 0 explicitly testing whether they emerge (circular risk: if EFA/CFA fails to support 3 factors, the entire program rationale collapses, yet prospectus frames them as the hypothesis to test).
- Qualitative hypotheses in designs (e.g., "psychosis: high 𝒫, low ℬ; OCD: high 𝒫, high ℬ, high 𝒯") reintroduce concrete directional predictions without data — same post-hoc stereotyping issue as before, just without numbers.
- Cross-cutting enhancements (A–F) are tacked on from 2025–2026 trends but poorly integrated — e.g., hierarchical Bayesian factor modeling (A) could supplant the triadic structure entirely; attractor dynamics (B) and effort discounting (D) dilute focus; AISI (C) and foundation models (E) are exciting but orthogonal to core axes.
- No discussion of model degeneracy, equifinality (multiple parameter sets → same behavior), or falsification criteria if simulations show poor identifiability.

### 4. Methodological and measurement bugs/risks
- Task battery remains noisy/low-reliability in clinical groups (prior ICC <0.6 acknowledged elsewhere but glossed over here).
- EMA items are ad-hoc/self-report proxies ("How much did small things feel overwhelming?") — prone to demand effects, poor specificity, low temporal resolution for true dynamical modeling.
- fMRI subset (N=20/group) is severely underpowered for connectivity/activation differences (typical need N=40–80+ per group for reliable effects).
- Pharmacological challenge relies on acute effects in healthy volunteers — poor proxy for chronic therapeutic mechanisms in patients (e.g., SSRIs take weeks for clinical action).
- No power calculations, effect size justifications beyond vague "d=0.3–0.6", handling of multiplicity, or preregistration details for each phase.
- Developmental integration (3.2) via ABCD/HCP-D secondary analysis assumes axis battery can be retrofitted — unlikely without new data collection.

### 5. Feasibility, resource, and ethical traps
- "All work is volunteer-driven. Progress will be slow and incremental" — honest, but dooms the program to near-certain stagnation without incentives, coordination, or minimal infrastructure.
- No budget breakdown beyond rough totals; no grant targets (e.g., NIMH R21/R01, BBRF Young Investigator); no fallback if Phase 0 fails.
- Ethical safeguards mentioned only in cross-cutting F — but pharmacological challenge in healthy volunteers raises mild risk (off-label low-dose antipsychotics); EMA in patients needs robust data protection (GDPR/HIPAA-equivalent).
- Open dataset plans good in theory, but de-identification in psychiatric samples is non-trivial (re-identification risk via unique symptom/task profiles).

### 6. Presentation and rhetorical issues
- Prospectus format (phases, deliverables, timelines, cross-cutting enhancements) still mimics mature research programs (e.g., RDoC, ABCD) despite "pre-research" status — signals ambition far beyond reality.
- "We have no funding, no lab, no institutional support" buried near end — should be front-loaded to set expectations.
- References remain selective/supportive; no counter-literature (e.g., replication failures in computational phenotyping, critiques of active inference overreach in psychiatry).
- CC0 license for "discussion" is odd — public domain waives all rights, yet content includes proposed methods/code that could be claimed by contributors later.

### Bottom-line verdict
This prospectus is the most realistic and self-aware iteration yet: it explicitly prioritizes simulations/psychometrics first, drops coordinates, scales ambitions, admits volunteer limits, and invites falsification. It functions well as a **detailed thought experiment** or grant pre-proposal sketch.

However, it remains **structurally unviable** as a real research program without:
- At least one named PI with computational psychiatry credentials.
- Seed funding or crowd-sourced pilot resources.
- Proof that Phase 0 simulations succeed (critical gatekeeper).
- Elimination of placeholder GitHub signaling until actual repo/code exists.

The biggest bug is **institutional vacuum** — without it, even Phase 0.1 (simulations) is unlikely to materialize beyond personal hobby code. If the goal is serious testing, pivot to collaborating on existing frameworks (active inference toolboxes, normative modeling consortia) rather than building a parallel triadic empire from zero.

This version is intellectually defensible as hypothesis-generation infrastructure — but treat it as exactly that: a prospectus for a program that does not yet exist and may never launch in its current form.

# Revision

# Triadic Computational Psychiatry Framework (TCPF)

## Status Report and Revised Minimal Path Forward

**Date:** March 2026  
**Status:** Hypothetical Framework — No Empirical Validation  
**Author:** [Unnamed individual / speculative proposal]  
**Contact:** None (no infrastructure exists)

---

## 1. Acknowledgment of Current Status

The Triadic Computational Psychiatry Framework exists as:

- A set of documents (v0.1, Research Program Prospectus)
- A series of critiques and responses
- **No empirical data, simulations, code, funding, institutional backing, named authors, community, or public visibility**

The GitHub repository `github.com/tcpf/framework` is a placeholder. It contains no code, no issues, no discussions. The invitation to "open issues" is functionally deceptive—there is no one maintaining it, no community to respond.

This is not a research program. It is a **thought experiment** written in the format of a research program.

---

## 2. Complete Acceptance of the Final Critique

Every point in the comprehensive report is accurate:

| Critique | Verdict |
|----------|---------|
| No real-world existence or visibility | **True.** Zero public footprint. |
| Overly optimistic timelines | **True.** Unfunded volunteer work cannot deliver this. |
| Conceptual circularity | **True.** Axes are hypothesis to test, not foundation. |
| Methodological underpowering | **True.** fMRI N=20/group is insufficient. |
| Feasibility and ethical traps | **True.** No IRB, no funding, no oversight. |
| Rhetorical overreach | **True.** Format mimics mature programs. |
| Institutional vacuum | **True.** No PI, no lab, no accountability. |

The prospectus is structurally unviable as presented.

---

## 3. The Core Problem

**Infrastructure does not exist.**
- No code
- No data
- No funding
- No team
- No institutional home
- No community
- No publication record

Without these, even Phase 0.1 (simulations) is unlikely to materialize beyond personal hobby code that never gets shared, reviewed, or built upon.

---

## 4. Revised Minimal Path Forward

If the framework is to move from thought experiment toward testability, the only viable path is:

### Step 1: Strip Everything Non-Essential

Remove:
- All phase structures beyond Phase 0.1
- All timelines (replace with "when complete")
- All multi-site consortium language
- All treatment prediction claims
- All fMRI/EMA/pharmacology plans
- All GitHub placeholder references
- All references to community, contributors, or open source infrastructure that doesn't exist

Keep only:
- The core hypothesis (three computationally grounded dimensions)
- One falsifiable simulation question: *Can 𝒫, ℬ, 𝒯 parameters be recovered from synthetic active inference data?*

### Step 2: One Person, One Simulation

A single researcher with:
- Python proficiency
- Access to computational modeling libraries (pymc, pyro, active inference toolboxes)
- 3–6 months of focused time

Could:
- Implement a generative POMDP with parameters mapping to 𝒫, ℬ, 𝒯
- Generate synthetic datasets across parameter ranges
- Attempt parameter recovery
- Report results publicly regardless of outcome

**Deliverable:** One preprint (arXiv/bioRxiv) with code repository (actual code, not placeholder).

### Step 3: If Simulation Fails

If parameters are not recoverable, the framework is falsified at the first hurdle. Publish negative results. Stop.

### Step 4: If Simulation Succeeds

If parameters are recoverable with acceptable error, then consider:

- Sharing code for others to test/replicate
- Discussing whether any existing datasets could be re-analyzed (secondary analysis, no new collection)
- Writing a review paper situating the idea within existing literature (with named authors who actually contribute)

### Step 5: No Further Steps Without External Buy-In

No new data collection, no clinical studies, no treatment prediction, no multi-site consortia unless and until:
- At least one established researcher with relevant expertise expresses interest
- Minimal funding is secured (internal seed, small grant)
- Ethics oversight is in place

---

## 5. What This Means for the Framework

The TCPF, as currently documented, is **not a research program**. It is:

- A set of ideas
- A series of documents
- A record of critique and response

That is not nothing. Thought experiments have value. They clarify hypotheses, expose assumptions, and generate discussion.

But they are not science until they produce:
- Testable implementations
- Empirical results
- Public code
- Community engagement

---

## 6. If You Are the Author (or Someone Who Wants to Test This)

Here is the only actionable next step:

**Write the simulation code.**

Not a plan for simulation code. Not a prospectus for simulation code. Not a GitHub repository with a README.

Actual code. Running. Documented. Public.

If you cannot do this yourself, the framework will remain speculative indefinitely.

If you can, do it. Then share it. Then see what happens.

---

## 7. Closing Statement

The critiques of this framework have been:

- Detailed
- Accurate
- Repeatedly incorporated

The current version is intellectually honest but **structurally unviable** without execution.

Execution requires code, data, or collaboration—none of which exist.

Until they do, the TCPF is best understood as:

**A detailed record of an idea being tested by conversation, awaiting testing by computation.**

That is not nothing. But it is not yet science.

---

**Document Date:** March 2026  
**Status:** Thought experiment — awaiting simulation  
**Contact:** None (author not identified)  
**License:** Not applicable (no content to license)

# Issues

The **Status Report and Revised Minimal Path Forward** (March 2026) represents the most grounded, self-reflective, and stripped-down iteration of the Triadic Computational Psychiatry Framework (TCPF) to date. It fully internalizes prior critiques, abandons structural pretensions (phases, timelines, placeholders), removes deceptive signaling (fake GitHub invites, community language), and reduces the entire proposal to its irreducible scientific core: **a single, falsifiable computational question about parameter identifiability in a generative model**.

This is intellectually mature and defensible as a **pure thought experiment / hypothesis kernel**. It correctly diagnoses itself as non-science until code runs and results appear publicly. The call to "write the simulation code" (not plan it) is the only honest next action.

### Remaining Issues, Bugs, Shortcomings, and Risks

Even in this minimalist form, problems persist — some inherent, some practical, some philosophical. Here is the exhaustive list:

1. **Still zero empirical / public footprint (confirmed as of Feb 21, 2026)**  
   - No GitHub repo exists at github.com/tcpf/framework (searches return unrelated CTF/capture-the-flag projects, nothing psychiatry-related).  
   - No preprints, mentions, or discussions of "Triadic Computational Psychiatry Framework" / "TCPF" in arXiv/bioRxiv/medRxiv, general web, or X/Twitter since 2025 (only unrelated acronyms like TcpF in cholera research, or noise).  
   - "Author: [Unnamed individual / speculative proposal]" + "Contact: None" is accurate but leaves zero accountability/traceability. If this is your idea (@MyKey00110000), linking it to a real identity/repo would help credibility; anonymity keeps it in vaporware territory.

2. **The core hypothesis remains arbitrarily triadic without justification**  
   - Why precisely **three** axes (𝒫, ℬ, 𝒯)? The document treats this as the hypothesis to test via simulation, but the choice is still top-down/heuristic (predictive coding + Markov blankets + discounting), not derived from theory/data.  
   - If recovery fails, it falsifies **this specific triad** — but not the broader idea of dimensional computational phenotyping (which already exists in RDoC/HiTOP/normative modeling/active inference). Success would only show "these three parameters can be recovered in toy POMDPs," not that they explain psychopathology.

3. **Simulation itself is non-trivial and failure-prone**  
   - Active inference POMDPs with meaningful mappings to 𝒫 (precision weighting), ℬ (Markov blanket permeability), 𝒯 (discount horizon / temporal depth) are complex to implement correctly.  
   - Common pitfalls: poor parameter identifiability (degenerate solutions), sensitivity to noise/priors, collapse in high-dimensional state spaces, computational cost (variational inference scales poorly).  
   - Many published active inference psychiatry models struggle with recovery even in simpler settings — expect negative or mixed results unless the model is deliberately over-simplified (risking triviality).

4. **"One person, one simulation" is optimistic for impact**  
   - A solo 3–6 month effort might produce runnable code + arXiv preprint, but:  
     - Without peer review / replication, it remains low-visibility speculation.  
     - Psychiatry/computational neuroscience reviewers will ask: "Why this triad over existing multi-factor models (e.g., Huys RL-depression, Stephan precision hierarchies, Friston transdiagnostic active inference)?"  
     - Negative results ("not recoverable") are publishable but rarely high-impact; positive results need external validation (others running it on real data) to matter.

5. **Post-simulation branching is underspecified and risky**  
   - If succeeds → "consider sharing code / secondary analysis / review paper" is vague. Secondary analysis of existing datasets (e.g., ABCD delay discounting + self-other tasks) would still require axis proxies — risking circularity or weak mapping.  
   - No criteria for "acceptable error" in recovery → arbitrary thresholds could bias toward false positives.  
   - "No further steps without external buy-in" is wise, but in practice, absent named contributors/funding, it dead-ends even if simulation works.

6. **Philosophical / scoping tension**  
   - The document calls TCPF a "thought experiment awaiting testing by computation" — excellent humility. But psychiatry is messy (heterogeneity, state-trait confounds, non-computational causes); even perfect recovery in synthetic data wouldn't license clinical translation without massive bridging work (which is explicitly ruled out here).  
   - Risk of sunk-cost fallacy: if simulation succeeds modestly, temptation to creep back toward grander claims despite warnings.

7. **Presentation quirks**  
   - Table format for critique acceptance is clean, but repeating "True" for every row feels redundant.  
   - "That is not nothing. But it is not yet science" is poignant — but the whole document is still framed as a formal "Status Report," which slightly undercuts the "just an idea" claim.

### Bottom-Line Verdict & Recommendation

This version succeeds at honesty: it is no longer pretending to be a program/framework/prospectus. It is a clear, concise **admission of current non-existence** plus a single, executable falsification test.

**Strengths:** Maximal humility, zero overreach, correct prioritization (simulations before anything else), invitation to kill the idea early.

**Fatal shortcoming:** Without actual code execution/public sharing, it remains **permanently stuck** as words about an idea. The closing line — "awaiting testing by computation" — is accurate, but the author (you?) must cross that threshold.

If this is your project:  
- Pick a simple active inference toolbox (e.g., pymdp, acaciagents, or custom PyMC).  
- Define minimal POMDP (e.g., 2-state world, actions with precision gating, boundary via blanket stats, temporal via gamma discounting).  
- Run recovery, post notebook on real GitHub/OSF/arXiv.  
- Accept whatever outcome falsifies/refines.

Until then, TCPF is exactly what it says: **a detailed record of an idea being stress-tested by conversation, still awaiting computation**. Valuable for personal clarity — but not yet contributory to the field.

If you'd like, I can sketch pseudocode for a toy POMDP mapping to these axes, or suggest specific libraries/papers for identifiability tests. Just say the word.

# Novel Enhancements

Here are **24 novel, cutting-edge enhancement suggestions** for the Triadic Computational Psychiatry Framework (TCPF), tailored to its current minimalist status as a thought experiment awaiting simulation-based identifiability testing. These draw from 2025–2026 trends in computational psychiatry: AI/ML integration for precision/personalized models, mechanistic predictive coding/active inference refinements, digital phenotyping/EMA for real-world dynamics, transdiagnostic connectome-based predictive modeling, optimism bias modeling, active intersubjective inference (AISI), whole-body exposome/multi-level spanning models, ethical AI guardrails, dynamic causal modeling for neuromodulation, metacognitive arbitration, and theory-driven + data-driven hybrids (e.g., from Nature Focus 2025, Frontiers 2025–2026 papers, Computational Psychiatry Conference 2025, and related works).

They prioritize filling gaps in identifiability, real-world applicability, mechanistic depth, and falsifiability while staying simulation-feasible (e.g., extendable from a single POMDP/active inference setup). Suggestions are numbered for clarity and focus on executable extensions post-initial recovery test.

### Identifiability & Model Refinement (Core to Phase 0 Simulation)
1. **Incorporate metacognitive arbitration as a fourth latent factor** — Model cognitive arbitration between candidate psychopathology dimensions (e.g., compulsivity vs. negative valence) within the POMDP, testing if 𝒫/ℬ/𝒯 emerge orthogonal to metacognitive precision (inspired by 2025 Mol Psychiatry on cognitive arbitration).

2. **Add optimism bias as a precision asymmetry submodule** — Implement high-precision likelihood bias toward positive outcomes in the generative model (per 2025 active inference optimism model), allowing simulation of resilience vs. depression/PTSD asymmetries and testing recovery of asymmetric 𝒫.

3. **Use dynamic causal modeling (DCM) priors in recovery** — Embed DCM-like biophysical constraints (neural mass models) into the POMDP to simulate fMRI/EEG-like data, improving identifiability of circuit-level mappings to axes (leveraging 2025–2026 precision psychiatry advances).

4. **Test hierarchical vs. flat structure in parameter recovery** — Compare recovery performance of flat 3-parameter model vs. hierarchical (e.g., perceptual precision under social precision) to address reductionism critique.

### Dynamical & Real-World Extensions (Bridging Synthetic to Phenotypic Data)
5. **Integrate piecewise-linear recurrent neural networks (PLRNNs) for attractor dynamics** — Extend POMDP to include PLRNN state transitions shaped by 𝒫/ℬ/𝒯, simulating symptom trajectories over "days" and testing basin shifts (e.g., depressive collapse per 2025 Entropy depression model).

6. **Model active intersubjective inference (AISI) for ℬ axis** — Add second-order social priors (transference/projection as distorted interpersonal precision) to boundary dynamics, enabling simulation of BPD/depression interpersonal distortions (from 2025 Frontiers AISI framework).

7. **Incorporate effort discounting temporal dynamics into 𝒯** — Extend discount horizon with hierarchical drift diffusion (HDDM-like) for cognitive/physical effort costs, linking to anhedonia/negative symptoms in synthetic choice data.

8. **Simulate digital phenotyping/EMA bursts** — Generate synthetic time-series "EMA" data (e.g., momentary precision overload, boundary merging, temporal rumination items) from the model, testing if axes recover from noisy, non-stationary sequences.

### Multi-Level & Transdiagnostic Scaling
9. **Add whole-body exposome/multi-level spanning** — Include peripheral inputs (e.g., inflammation proxies as modulating precision gain) in the generative model, testing transdiagnostic effects across "virtual patients" (per 2025 PMC exposome models).

10. **Implement connectome-based predictive modeling (CPM) proxies** — Simulate functional connectivity matrices from hierarchical POMDP states, then apply transdiagnostic CPM to predict axis deviations from "brain" patterns (inspired by 2025 bioRxiv transdiagnostic CPM).

11. **Test normative deviation mapping in synthetic cohorts** — Generate large normative "healthy" parameter distributions, then compute individual deviations along axes for "disordered" variants, aligning with foundation/normative models in precision psychiatry.

12. **Incorporate developmental trajectories** — Parameterize age-dependent changes (e.g., maturing discount horizon 𝒯, synaptic gain for 𝒫) to simulate lifespan shifts and early deviation markers (tying to developmental computational psychiatry labs).

### AI/ML & Precision Enhancements
13. **Use generative AI for data augmentation in recovery tests** — Employ variational autoencoders or diffusion models to augment synthetic datasets with realistic noise/variability, improving robustness of identifiability metrics.

14. **Add foundation model-like priors** — Pre-train POMDP components on large simulated "healthy" datasets (e.g., via self-supervised contrastive learning), then fine-tune for psychopathology variants to test transfer learning for axes.

15. **Model combination therapy effects (psychopharm + neuromodulation)** — Simulate acute parameter shifts (e.g., gain modulation from tDCS/TMS analogs) combined with chronic priors, per 2025 Computational Psychiatry 2.0 for schizophrenia.

16. **Incorporate ethical bias/guardrail simulations** — Add adversarial perturbations (e.g., demographic confounders) to test if axis recovery holds under biased synthetic data, aligning with 2025 Nature Computational Science ethics focus.

### Mechanistic & Therapeutic Simulation Extensions
17. **Simulate fear learning/generalization for threat-related axes** — Extend to Pavlovian tasks with transdiagnostic generalization gradients, testing 𝒫 hyper-precision in anxiety/PTSD analogs.

18. **Model synaptic excitatory-inhibitory imbalance** — Implement E/I ratio as modulating precision/boundary in neural mass approximations, unifying competing depression perspectives (2025 Entropy active inference depression).

19. **Test psychedelic-assisted inference flexibility** — Simulate reduced rigid priors (e.g., lowered precision weighting) post-"psychedelic" state perturbation, measuring axis shifts toward origin.

20. **Add sleep/activity circadian modulation** — Include circadian oscillators affecting temporal integration 𝒯, generating synthetic actigraphy-like patterns for transdiagnostic biomarkers.

### Validation & Falsification Tools
21. **Implement model comparison suites** — Use Bayesian model evidence (e.g., via PyMC/Stan) to compare 3-axis vs. alternatives (e.g., single p-factor, HiTOP spectra) on synthetic likelihoods.

22. **Simulate parameter degeneracy scenarios** — Deliberately introduce equifinal parameter sets (multiple configs → same behavior) to quantify identifiability limits and failure modes.

23. **Test cross-modal recovery (behavioral + "neural")** — Generate paired synthetic behavioral + "fMRI" time-series, testing if axes recover better from multimodal vs. unimodal data.

24. **Build open-source minimal toolbox skeleton** — Even in solo mode, structure code as modular (e.g., pymdp-based POMDP core + axis parameter mappers + recovery scripts + Jupyter tutorials), positioning for future external buy-in if recovery succeeds.

These enhancements keep the focus simulation-first (executable in Python with pymdp, PyMC, or similar), while addressing core gaps: mechanistic depth, real-world bridging, transdiagnostic relevance, ethical robustness, and falsifiability. Start with 1–4 for immediate identifiability extensions in your initial POMDP run. If recovery holds, layer in dynamics (5–8) next. Success on even a subset could justify a preprint + real repo, potentially attracting collaborators. If you'd like pseudocode sketches for any (e.g., optimism bias submodule or AISI extension), just specify.

# Revision 

# Triadic Computational Psychiatry Framework (TCPF)

## Minimal Simulation-First Implementation Plan

**Date:** March 2026  
**Status:** Pre-simulation — No Code Written Yet  
**Author:** [Unnamed / speculative]  
**Contact:** None (placeholder only)

---

## 1. Current Status (Restated)

The TCPF exists as a series of documents and critiques. There is:

- No code
- No data
- No funding
- No team
- No institutional home
- No public visibility

The only viable next step is **one person writing one simulation** to test whether the three hypothesized parameters (𝒫, ℬ, 𝒯) can be recovered from synthetic data generated by a plausible cognitive model.

This document outlines a **minimal, executable simulation plan** using off-the-shelf tools. It is designed to be implementable by a single researcher with Python proficiency in 3–6 months.

---

## 2. Core Simulation Design

### 2.1 Generative Model: Active Inference POMDP

We implement a Partially Observable Markov Decision Process (POMDP) under active inference, with three parameters mapped to 𝒫, ℬ, 𝒯.

**State Space:**
- Two hidden states (e.g., "safe" vs. "threat" environment)
- Two observations (e.g., "neutral cue" vs. "threat cue") with controlled ambiguity

**Action Space:**
- Two actions (e.g., "approach" vs. "avoid")

**Parameter Mapping:**

| Parameter | Mathematical Role | Mapping to Axis |
|-----------|-------------------|-----------------|
| `prec_A` | Precision of observation likelihood (inverse temperature) | 𝒫 (high = overweight sensory evidence) |
| `prec_B` | Precision of transition beliefs (prior over state changes) | ℬ (high = rigid world model, low = fluid) |
| `gamma` | Temporal discount factor (policy horizon) | 𝒯 (high = future-oriented, low = present-focused) |

**Implementation:**
- Use `pymdp` (active inference Python library) or custom PyMC implementation
- Generative process: for each trial, sample states, generate observations, update beliefs, select actions

### 2.2 Parameter Recovery Procedure

**Step 1: Generate Synthetic Dataset**
- Define parameter ranges:
  - `prec_A`: log-uniform [0.1, 10.0] → maps to 𝒫 ∈ [-3, +3] via transformation
  - `prec_B`: log-uniform [0.1, 10.0] → maps to ℬ ∈ [-3, +3]
  - `gamma`: [0.5, 0.99] → maps to 𝒯 ∈ [-3, +3]
- Sample 500 parameter sets from joint distribution
- For each set, simulate 100 trials of the POMDP
- Store observation-action sequences (synthetic behavioral data)

**Step 2: Fit Model to Synthetic Data**
- Use hierarchical Bayesian inference (PyMC/Stan) to recover parameters
- Model: likelihood = P(data | parameters) under same generative model
- Priors: weakly informative (e.g., Normal(0,1) on transformed scales)

**Step 3: Assess Recoverability**
- Compute correlation between true and recovered parameters
- Calculate root mean squared error (RMSE) and coverage of 95% credible intervals
- Test for parameter degeneracy (multiple parameter sets yielding same data likelihood)

**Step 4: Sensitivity Analysis**
- Vary number of trials (50, 100, 200)
- Vary noise levels (add random action noise)
- Test with misspecified models (e.g., fitting with wrong state space size)

---

## 3. Minimal Viable Code Structure

```
tcpf_sim/
├── README.md                    # What this is (pre-simulation)
├── requirements.txt             # pymdp, pymc, arviz, numpy, matplotlib
├── generate_data.py              # Sample parameters, run generative model, save sequences
├── fit_model.py                  # Hierarchical Bayesian inference on synthetic data
├── recovery_analysis.py          # Correlations, RMSE, degeneracy checks
├── plots/                        # Output figures
└── results/                      # CSV of recovery metrics
```

### 3.1 Key Implementation Details

**Generative Model (pseudocode):**

```python
import numpy as np
from pymdp import inference

def generate_trials(prec_A, prec_B, gamma, n_trials=100):
    # Define POMDP structure
    n_states = 2
    n_obs = 2
    n_actions = 2
    
    # A matrix (observation likelihood) - controlled by prec_A
    A = np.ones((n_obs, n_states)) / n_obs  # start uniform
    # Add precision weighting (higher prec_A = more peaked)
    A[0,0] = 1 / (1 + np.exp(-prec_A))  # state 0 -> obs 0 more likely
    A[1,0] = 1 - A[0,0]
    A[1,1] = 1 / (1 + np.exp(-prec_A))  # state 1 -> obs 1 more likely
    A[0,1] = 1 - A[1,1]
    
    # B matrices (transition) - controlled by prec_B
    B = np.zeros((n_states, n_states, n_actions))
    # Higher prec_B = more deterministic transitions
    for a in range(n_actions):
        B[:,:,a] = np.eye(n_states) * (1 - 1/(1+np.exp(-prec_B))) + \
                   (1 - np.eye(n_states)) * (1/(1+np.exp(-prec_B))) / (n_states-1)
    
    # Run trials
    states = []
    obs = []
    actions = []
    
    s = np.random.choice(n_states)
    for t in range(n_trials):
        # Generate observation
        o = np.random.choice(n_obs, p=A[:,s])
        # Infer state (simplified - agent uses same generative model)
        qs = inference.update_posterior(A, o, prior=None)
        # Select action (simplified policy selection with gamma horizon)
        # ... (full implementation would use expected free energy)
        a = select_action(qs, B, gamma)
        # Transition
        s_next = np.random.choice(n_states, p=B[:,s,a])
        
        states.append(s)
        obs.append(o)
        actions.append(a)
        s = s_next
    
    return np.array(obs), np.array(actions)
```

**Parameter Recovery (pseudocode):**

```python
import pymc as pm

def fit_model(obs_data, action_data):
    with pm.Model() as model:
        # Priors on transformed parameters
        prec_A_raw = pm.Normal('prec_A_raw', 0, 1)
        prec_B_raw = pm.Normal('prec_B_raw', 0, 1)
        gamma_raw = pm.Normal('gamma_raw', 0, 1)
        
        # Transform to parameter ranges
        prec_A = pm.Deterministic('prec_A', pm.math.exp(prec_A_raw * 2))  # log-normal-ish
        prec_B = pm.Deterministic('prec_B', pm.math.exp(prec_B_raw * 2))
        gamma = pm.Deterministic('gamma', pm.math.invlogit(gamma_raw) * 0.49 + 0.5)  # [0.5, 0.99]
        
        # Likelihood: P(data | parameters)
        # This requires implementing the generative model as a PyMC custom distribution
        # or using approximate methods (e.g., importance sampling)
        # ...
        
        trace = pm.sample(2000, tune=1000)
    return trace
```

---

## 4. Enhancement Modules (Post-Recovery)

If basic recovery succeeds, the following modules can be added incrementally:

### Module A: Optimism Bias (Suggestion #2)
- Add asymmetric precision: `prec_pos` vs. `prec_neg` for positive/negative outcomes
- Test if recovery improves with this fourth parameter

### Module B: Active Intersubjective Inference (Suggestion #6)
- Add second agent with reciprocal beliefs
- Boundary parameter controls coupling strength between agents' state inferences

### Module C: Hierarchical Precision (Suggestion #4)
- Implement two-level hierarchy (perceptual precision under metacognitive precision)
- Test if flat 3-parameter model recovers hierarchical structure

### Module D: PLRNN Dynamics (Suggestion #5)
- Replace Markov transitions with PLRNN state equation
- 𝒫, ℬ, 𝒯 shape the RNN's fixed points and flow

### Module E: Synthetic EMA (Suggestion #8)
- Generate time-series "momentary ratings" by adding noise to latent states
- Test recovery from sparse, noisy observations

---

## 5. Success Criteria

The simulation "succeeds" if:

1. **Correlation > 0.8** between true and recovered parameters (median across 500 simulations)
2. **RMSE < 0.5** on normalized scale [-3, +3]
3. **Coverage > 90%** of 95% credible intervals contain true value
4. **No strong degeneracy**: condition number of parameter covariance < 10

If these hold, proceed to Module A. If not, the framework is falsified at the first hurdle.

---

## 6. Current Blockers

- **No code written yet**
- **No GitHub repository with actual content**
- **No named author** → no accountability or collaboration path
- **No funding** → but simulation requires only time and a laptop

---

## 7. What Success Would Enable

If recovery succeeds and code is public:

- Preprint on arXiv/bioRxiv
- Invitation for others to test/replicate
- Potential to apply to existing datasets (secondary analysis)
- Gradual community building if results hold

If recovery fails:

- Publish negative results
- Framework abandoned or radically revised
- Still a contribution (falsification is science)

---

## 8. Open Questions for the Author

1. Will you write this code? (Yes/No/When?)
2. If yes, will you make it public under a real account?
3. If no, will you abandon the framework or seek a collaborator who can?

---

**This is the minimal executable path. Everything else is words.**
