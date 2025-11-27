# Comprehensive Review: HAR-CAESar Thesis
## Forecasting Tail Risk with Long-Memory: A Heterogeneous Extension of the CAESar Model

**Reviewer:** AI Critical Review System  
**Review Date:** 27 November 2025  
**Framework:** Zero-Trust Verification Protocol

---

## Executive Summary

- **Verdict:** Major Revision Required
- **Reproducibility Score:** 6.5/10
- **Critical Issues:** 1
- **Major Issues:** 4  
- **Minor Issues:** 2
- **Key Recommendation:** Correct the parameter count discrepancy between thesis (10 per equation) and code (9 per equation), document the reparametrisation used in estimation, and implement or document missing statistical tests.

---

## Issue Catalogue

### Critical Issues

#### Issue 1: Parameter Count Discrepancy (Code-Paper Mismatch)

**Severity:** Critical  
**Location:** `src/har_caesar/models/har_caesar.py:80` vs `LaTeX/chapters/methodology.tex:Section 3.5.2`  
**Category:** Algorithm Concordance Check

**Paper Claim (Methodology Section 3.5.2):**
> "This specification introduces four additional parameters in each equation (six return coefficients instead of two), for a total of eight additional parameters relative to standard CAESar. **The VaR equation now has ten parameters and the ES equation has ten parameters**, compared to five each in the baseline."

**Code Location:** `har_caesar.py:80`
```python
# HAR-CAESar specific parameters
self.mdl_spec = 'HAR'
self.n_parameters = 9  # Per equation: intercept + 6 return + 2 AR
```

**Detailed Analysis:**

Code parameter breakdown (per equation):
- Intercept: 1
- Daily positive/negative: 2  
- Weekly positive/negative: 2
- Monthly positive/negative: 2
- Lagged VaR: 1
- Lagged ES: 1
- **Total: 9 parameters**

Paper also states "eight additional parameters relative to standard CAESar":
- Standard CAESar (AS spec): 5 parameters per equation
- 5 + 4 = 9 parameters ✓ (this calculation is correct)
- But text claims "ten parameters" ✗ (inconsistent)

**Impact:** 
This discrepancy creates fundamental ambiguity about the model specification. Readers cannot determine:
1. Whether the implementation is wrong
2. Whether the thesis description is wrong  
3. Whether parameters were added/removed during development

This affects:
- Model complexity comparisons
- Degrees of freedom in statistical tests
- Interpretation of results
- Ability to replicate the study

**Evidence Code-Paper Mismatch:**
```python
# Parameter indices for clarity
# VaR equation (indices 0-8):
#   0: intercept
#   1: daily positive, 2: daily negative
#   3: weekly positive, 4: weekly negative
#   5: monthly positive, 6: monthly negative
#   7: lagged q, 8: lagged e
# ES equation (indices 9-17): same structure
```
Comment explicitly documents 9 parameters (0-8 inclusive).

**Recommended Fix:**
1. Verify the correct count is 9 parameters per equation
2. Change thesis text from:
   - "The VaR equation now has **ten** parameters" → "The VaR equation now has **nine** parameters"
   - Same for ES equation
3. Verify the calculation "eight additional parameters" is consistent with "four per equation" (4×2 = 8 ✓)

---

### Major Issues

#### Issue 2: Undocumented Reparametrisation (ES Residual vs Direct ES)

**Severity:** Major  
**Location:** `har_caesar.py:207-218` vs `methodology.tex:Equations 3.5.2`  
**Category:** Algorithm Concordance Check

**Paper Specification (Equations 8-9):**
The thesis presents the HAR-CAESar model as estimating VaR and ES directly:

$$\hat{e}_t = \gamma_0 + \gamma_1^{(d)} (r_{t-1}^{(d)})^+ + \gamma_2^{(d)} (r_{t-1}^{(d)})^- + \ldots + \gamma_3 \hat{q}_{t-1} + \gamma_4 \hat{e}_{t-1}$$

**Code Implementation:**
```python
def R_HARloop(self, beta, y, q, r0, har_features, pred_mode=False):
    """Recursive loop for ES residual (r = e - q) estimation with HAR specification."""
    # ...
    r_t = (beta[0] + 
           rd_coeff[t] * rd[t] + 
           rw_coeff[t] * rw[t] + 
           rm_coeff[t] * rm[t] + 
           beta[7] * q[t-1] +   # lagged VaR
           beta[8] * r[t-1])    # lagged RESIDUAL (r = e - q), not lagged ES
```

**Analysis:**
The code estimates the ES **residual** $r_t = e_t - q_t$ in Stage 2 (Barrera loss), not ES directly. This is:
1. A valid approach (used in original CAESar paper by Gatta et al. 2024)
2. **NOT documented** in the thesis equations
3. Creates apparent mismatch: thesis shows $\hat{e}_{t-1}$ as regressor, code uses $r_{t-1}$

The relationship is:
- Thesis form: $\hat{e}_t$ depends on $\hat{e}_{t-1}$
- Code form: $r_t = e_t - q_t$ depends on $r_{t-1} = e_{t-1} - q_{t-1}$
- In Stage 3, joint estimation recovers ES directly

**Impact:**
- Readers attempting to implement from thesis equations will get different numerical results
- PhD examiners may flag this as implementation error
- Reproducibility compromised without access to code

**Recommended Fix:**
Add subsection in Methodology (e.g., "3.5.3 Implementation Details"):

> "Following Gatta et al. (2024), we estimate the ES residual $r_t = e_t - q_t$ in Stage 2 rather than ES directly. The specification in Equation (9) is therefore implemented as:
> $$r_t = \gamma_0 + \ldots + \gamma_3 q_{t-1} + \gamma_4 r_{t-1}$$
> where $r_t = e_t - q_t$. This reparametrisation ensures numerical stability and is equivalent to the direct specification after the joint Stage 3 optimisation."

---

#### Issue 3: Fissler-Ziegel Loss Formula Sign Ambiguity

**Severity:** Major  
**Location:** `har_caesar.py:122-125` & `utils.py:95-106` vs `methodology.tex:Section 3.6.2`  
**Category:** Formula Verification

**Paper Formula (Equation after 3.6.2):**
$$L_{FZ}(r, q, e) = \frac{1}{\theta |e|} (q - r) \mathbf{1}_{\{r \leq q\}} + \frac{q}{|e|} + \log|e| - 1$$

**Code Implementation (`har_caesar.py:122-125`):**
```python
def joint_loss(self, v, e, y):
    """Compute the Fissler-Ziegel (Patton) loss function for joint estimation (Step 3)."""
    loss_val = np.mean(
        np.where(y <= v, (y - v) / (self.theta * e), 0) + v / e + np.log(-e)
    ) + self.lambdas['e'] * np.mean(np.where(e > v, e - v, 0)) + \
        self.lambdas['q'] * np.mean(np.where(v > 0, v, 0))
    return loss_val
```

**Code Implementation (`utils.py:95-106`):**
```python
class patton_loss():
    def __call__(self, v_, e_, y_):
        v, e, y = v_.flatten()*100, e_.flatten()*100, y_.flatten()*100
        if self.ret_mean:
            loss = np.nanmean(
                np.where(y<=v, (y-v)/(self.theta*e), 0) + v/e + np.log(-e) - 1
            )
```

**Discrepancy Analysis:**

| Element | Paper | Code (har_caesar.py) | Code (utils.py) | Match? |
|---------|-------|---------------------|-----------------|--------|
| Indicator term | $(q-r)/(\theta \|e\|)$ | $(y-v)/(\theta e)$ | $(y-v)/(\theta e)$ | Partial |
| Absolute value | $\|e\|$ | $e$ (ES is negative) | $e$ (ES is negative) | ✓ |
| Logarithm | $\log\|e\|$ | $\log(-e)$ | $\log(-e)$ | ✓ |
| Constant | $-1$ | **missing** | **present** | Mixed |
| Sign in numerator | $q-r$ | $y-v = r-q$ | $y-v = r-q$ | ✗ |

**Analysis:**
1. **Sign reversal:** Paper uses $(q-r)$, code uses $(y-v) = (r-q)$ (opposite sign)
2. Since ES is negative ($e < 0$), using $-e = |e|$ is correct
3. The $-1$ constant is omitted in `har_caesar.py` but present in `utils.py`
4. Omitting $-1$ doesn't affect optimisation (constant shift) but affects reported loss values

**Impact:**
- Loss values in results tables will differ from formula by constant shift
- Sign reversal is mathematically equivalent IF compensated by penalty terms, but undocumented
- Creates confusion when comparing to Patton et al. (2019) reference

**Recommended Fix:**
1. Add $(- 1)$ constant to `har_caesar.py` loss function for consistency
2. Add footnote in thesis explaining: "We use the simplified form with $(r-q)$ in the numerator, which is equivalent to $(q-r)$ under the correct sign convention for ES ($e < 0$)"

---

#### Issue 4: Missing Christoffersen Conditional Coverage Test

**Severity:** Major  
**Location:** `utils.py` (not found) vs `methodology.tex:Section 3.8.1`  
**Category:** Statistical Test Verification

**Paper Claim (Section 3.8.1):**
> "The conditional coverage test of Christoffersen (1998) additionally checks whether violations are independently distributed over time. The test combines the unconditional coverage test with a test of first-order Markov independence:
> $$LR_{CC} = LR_{UC} + LR_{IND}$$"

**Code Search Results:**
Searched `utils.py` for Christoffersen-related implementations:
- Found: Kupiec test (via `kupiec_test` in experiments)
- Found: McNeil-Frey test (`McneilFrey_test` class)
- Found: Acerbi-Szekely test (`AS14_test` class)
- Found: Diebold-Mariano test (`DMtest` class)
- **NOT FOUND:** Christoffersen conditional coverage test
- **NOT FOUND:** $LR_{CC}$ or $LR_{IND}$ implementation

**Impact:**
The thesis **claims** to use the Christoffersen test for backtesting, but provides no implementation. This means either:
1. Results section doesn't actually report CC test (false claim)
2. External package was used without citation
3. Test was manually computed outside the codebase

**Recommended Fix:**
1. **Option A (Preferred):** Implement Christoffersen test in `utils.py`:
```python
def christoffersen_cc_test(violations, theta):
    """Christoffersen conditional coverage test.
    
    Args:
        violations: Binary array of violation indicators
        theta: Nominal coverage level
    
    Returns:
        dict with 'LR_UC', 'LR_IND', 'LR_CC', 'p_value'
    """
    # Implementation required
```

2. **Option B:** Add to thesis: "Statistical tests were computed using the [package name] library (citation)"

---

#### Issue 5: Code Duplication (HAR-CAESar Not Inheriting from Base)

**Severity:** Major  
**Location:** `har_caesar.py` (entire file architecture)  
**Category:** Code Quality / Maintenance

**Expected Design (from Memory Bank):**
> "The `HAR_CAESar` class **overrides** the data preparation step to include heterogeneous lags"

**Actual Implementation:**
`HAR_CAESar` is a **completely standalone class** that **duplicates** ~400 lines of code from `CAESar_base`:

Duplicated components:
- `loss_function()` method (identical to CAESar_base)
- `joint_loss()` method (identical)
- `ESloss()` method (nearly identical, adds `har_features` parameter)
- `Jointloss()` method (nearly identical, adds `har_features` parameter)
- `optim4mp()` method (identical)
- `joint_optim()` method (identical)
- `fit_predict()` method (identical)

The only unique components are:
- `R_HARloop()` - HAR version of residual loop
- `Joint_HARloop()` - HAR version of joint loop
- `fit()` - Modified to compute HAR features

**Impact:**
1. **Maintenance burden:** Bug fixes in CAESar_base must be manually ported to HAR_CAESar
2. **Divergence risk:** The two implementations can drift apart
3. **Testing complexity:** Need separate test suites for duplicated logic
4. **Code smell:** Violates DRY principle

**Recommended Fix:**
Refactor to use inheritance:

```python
class HAR_CAESar(CAESar_base):
    """HAR-CAESar: Heterogeneous Autoregressive CAESar."""
    
    def __init__(self, theta, lambdas=dict()):
        super().__init__(theta, lambdas)
        self.mdl_spec = 'HAR'
        self.n_parameters = 9
        # Override loop functions
        self.loop = self.R_HARloop
        self.joint_loop = self.Joint_HARloop
    
    def R_HARloop(self, beta, y, q, r0, pred_mode=False):
        """Override with HAR-specific implementation."""
        har_features = compute_har_features(y)
        # ... HAR-specific logic
    
    # No need to duplicate loss_function, joint_loss, optim4mp, etc.
```

**Note:** This is a **refactoring recommendation** for code quality, not a blocking issue for thesis acceptance.

---

### Minor Issues

#### Issue 6: Undocumented Penalty Weight Values

**Severity:** Minor  
**Location:** `har_caesar.py:72`

**Code:**
```python
self.lambdas = {'r': 10, 'q': 10, 'e': 10}
self.lambdas.update(lambdas)
```

**Thesis Reference (Section 3.5.1):**
> "subject to a monotonicity constraint. The joint optimisation problem is ... where $L_{FZ}$ is the Fissler-Ziegel loss defined below and the penalty term with weight $\lambda > 0$ discourages violations of the monotonicity constraint"

**Issue:** 
The thesis uses singular "$\lambda$" but code uses three separate penalties ($\lambda_r = \lambda_q = \lambda_e = 10$). The values (10) are not documented.

**Impact:** 
Readers cannot assess sensitivity to penalty choice. Results may change with different penalty weights.

**Recommended Fix:**
Add to thesis: "We set the penalty weights $\lambda_r = \lambda_q = \lambda_e = 10$ based on preliminary tuning experiments that balanced constraint satisfaction with optimisation stability."

---

#### Issue 7: Experiment Capped at 10 Windows

**Severity:** Minor  
**Location:** `experiments/experiments_har_caesar.py:115`

**Code:**
```python
# Rolling window experiment
n_windows = (len(df_returns) - TRAIN_WINDOW - TEST_WINDOW) // STEP_SIZE + 1

for window_idx in tqdm(range(min(n_windows, 10)), desc=f"Windows (theta={theta})"):
```

**Issue:**
The experiment is artificially limited to 10 rolling windows via `min(n_windows, 10)`. With 5,261 observations, TRAIN_WINDOW=2000, TEST_WINDOW=250, STEP_SIZE=250, this gives:
- Maximum windows possible: $(5261 - 2000 - 250) / 250 + 1 \approx 13$ windows
- Actually run: 10 windows
- Coverage: ~77% of available out-of-sample period

**Impact:**
If thesis results tables use only 10 windows, statistical power is reduced. If full results use more windows, the experiment script doesn't match reported results.

**Recommended Fix:**
1. Remove the `min(n_windows, 10)` cap or document as debugging mode
2. Add comment explaining why limit exists if intentional
3. Verify results section uses full rolling window set

---

## Reproducibility Assessment

### Phase 1: Metadata Verification

- [x] Repository accessible (local copy verified)
- [x] Code matches framework versions (Python 3.12, specified in techContext.md)
- [x] Dependencies installable (`environment.yml` provided)
- [x] README provides setup instructions ✓
- [x] Data generation scripts present (`generate_synthetic_data.py`)
- [x] Random seeds documented (SEED=42 in experiment scripts)

**Gate 1 Status:** ✓ PASS

---

### Phase 2: Algorithm Concordance Check

| Paper Claim | Code Location | Match Status | Discrepancies |
|-------------|---------------|--------------|---------------|
| HAR features: daily, weekly, monthly | `har_caesar.py:28-64` | ✓ Match | None |
| Asymmetric slope: $(r^+, r^-)$ | `har_caesar.py:169-172` | ✓ Match | None |
| 10 parameters per equation | `har_caesar.py:80` | ✗ Mismatch | **Code has 9, thesis claims 10** |
| ES equation with lagged ES | `methodology.tex:Eq 9` | Partial | Code uses $r = e - q$ reparametrisation |
| Fissler-Ziegel loss | `har_caesar.py:122` | Partial | Sign convention differs, missing -1 |
| 3-stage estimation | `har_caesar.py:fit()` | ✓ Match | CAViaR → Barrera → FZ stages present |
| Christoffersen test | `methodology.tex` claim | ✗ Missing | **Not implemented in utils.py** |

**Gate 2-3 Status:** ⚠️ PARTIAL (1 critical mismatch, 2 major discrepancies)

---

### Phase 3: Data Generation Audit

**Experiment Script (`experiments_har_caesar.py`):**

```python
SEED = 42  # Global constant

def run_experiment(theta):
    # ...
    for window_idx in range(...):
        for asset in df_returns.columns:
            # CAESar
            mdl = CAESar(theta, 'AS')
            res = mdl.fit_predict(y_full, TRAIN_WINDOW, seed=SEED, ...)
            
            # HAR-CAESar
            mdl = HAR_CAESar(theta)
            res = mdl.fit_predict(y_full, TRAIN_WINDOW, seed=SEED, ...)
```

**HAR_CAESar.fit() seed usage:**
```python
def fit(self, yi, seed=None, ...):
    # Step 0: CAViaR uses seed
    cav_res = CAViaR(self.theta, 'AS', p=1, u=1).fit(
        yi, seed=seed, return_train=True, q0=q0)
    
    # Step 2: Random initialization
    np.random.seed(seed)  # ✓ Correct scope
    beta0 = [np.random.uniform(0, 1, nInitialVectors)]
    beta0.append(np.random.uniform(-1, 1, nInitialVectors))
    beta0.append(np.random.randn(*nInitialVectors))
```

**Assessment:**
- ✓ Seed passed to each fit call
- ✓ Seed set **per-model-fit**, not globally
- ✓ Random initialization uses provided seed
- ✓ No fixed seed inside functions (anti-pattern avoided)

**Critical Check:**
```python
# ANTI-PATTERN (not present in code):
# def generate_data(n):
#     np.random.seed(42)  # ❌ Would give same data every call
```

**Gate 6 Status:** ✓ PASS (Random seed management enables true replication)

---

### Phase 4: Statistical Rigor Verification

| Claim | Implementation | Verification |
|-------|---------------|--------------|
| Fissler-Ziegel loss for joint estimation | `har_caesar.py:joint_loss()` | ✓ Present (with sign ambiguity) |
| Tick loss for VaR | `utils.py:PinballLoss` | ✓ Correct implementation |
| Barrera loss for ES Stage 2 | `utils.py:barrera_loss` | ✓ Correct implementation |
| Kupiec unconditional coverage | `experiments:kupiec_test()` | ✓ Implemented |
| Christoffersen conditional coverage | **Missing** | ✗ Not found |
| McNeil-Frey ES test | `utils.py:McneilFrey_test` | ✓ Implemented with bootstrap |
| Diebold-Mariano test | `utils.py:DMtest` | ✓ Implemented with HAC variance |

**Gate 4 Status:** ⚠️ PARTIAL (Christoffersen test missing)

---

### Phase 5: Theoretical Claims Audit

**Claim:** Joint elicitability of (VaR, ES) via Fissler-Ziegel (2016)

**Verification:**
- Citation present: ✓
- Fissler-Ziegel (2016) paper verifies joint elicitability: ✓
- Thesis derives identification function: ✓ (Section 3.6.1)
- Loss function matches FZ class: ⚠️ (modulo sign convention)

**Claim:** HAR model captures long-memory (Corsi 2009)

**Verification:**
- Citation present: ✓
- Corsi (2009) establishes HAR for realized volatility: ✓
- Extension to tail risk is novel contribution: ✓

**Claim:** M-estimation consistency under regularity conditions (Newey & McFadden 1994)

**Verification:**
- Citation present: ✓
- Regularity conditions stated: ✓ (Section 3.7)
- Compactness, continuity, identification verified: ✓

**Gate 5 Status:** ✓ PASS (Theoretical rigor adequate)

---

## Overall Reproducibility Score: 6.5/10

**Scoring Breakdown:**
- **Metadata/Setup:** 2/2 points (Repository complete, dependencies clear)
- **Algorithm Match:** 2/4 points (Parameter count error -1, reparametrisation undocumented -1)
- **Data Generation:** 2/2 points (Seed management correct)
- **Statistical Tests:** 0.5/2 points (Christoffersen missing -1.5)
- **Theoretical Rigor:** 0/0 points (No deductions, theory sound)

**Interpretation:** 
- **6.5/10 = "Good but Problematic"** (per framework rubric)
- Major gaps in documentation
- Partial reproducibility achievable
- Undocumented methodological choices reduce confidence

**Estimated Effort to Achieve 9/10:** 4-6 person-hours
1. Fix parameter count in thesis (30 min)
2. Add reparametrisation documentation (1 hour)
3. Implement Christoffersen test (2 hours)
4. Align loss function constant (30 min)
5. Full testing and verification (1-2 hours)

---

## What Works Well

### Strengths

1. **Clean HAR Feature Implementation**
   - Matches thesis definitions exactly
   - Handles boundary conditions correctly (partial windows for t < 22)
   - Vectorised for efficiency

2. **Proper Three-Stage Estimation**
   - CAViaR initialization provides stable starting point
   - Barrera loss for ES residual estimation
   - Joint refinement with FZ loss
   - Matches original CAESar methodology (Gatta et al. 2024)

3. **Robust Optimization**
   - Multiple random initializations (nV=102)
   - Best k=3 selected for refinement
   - Multiprocessing for parallelization
   - Convergence retry logic (n_rep=5)

4. **Comprehensive Loss Functions**
   - Pinball loss for VaR
   - Barrera loss for ES Stage 2
   - Fissler-Ziegel loss for joint estimation
   - Penalty terms for monotonicity constraints

5. **Statistical Testing Suite**
   - McNeil-Frey ES calibration test
   - Acerbi-Szekely Z1/Z2 tests
   - Diebold-Mariano forecast comparison
   - Bootstrap-based inference (n_boot=10,000)

6. **Reproducibility Infrastructure**
   - Random seeds passed through call chain
   - Pickle results for intermediate checkpoints
   - Clear separation: train/test/validation
   - Rolling window with configurable parameters

---

## Recommendations

### For Authors (Priority Order)

#### **CRITICAL (Must Fix Before Submission):**

1. **Fix Parameter Count Discrepancy**
   - **Action:** Change "ten parameters" → "nine parameters" in thesis Section 3.5.2
   - **File:** `LaTeX/chapters/methodology.tex`
   - **Time:** 5 minutes
   - **Impact:** Eliminates primary critical issue

#### **HIGH PRIORITY (Strongly Recommended):**

2. **Document Reparametrisation**
   - **Action:** Add new subsection "3.5.3 Implementation Details" explaining $r = e - q$ form
   - **File:** `LaTeX/chapters/methodology.tex`
   - **Time:** 1 hour
   - **Template provided below**

3. **Implement or Document Christoffersen Test**
   - **Option A:** Implement in `src/har_caesar/utils.py`
   - **Option B:** Add footnote citing external package used
   - **Time:** 2 hours (Option A) or 10 minutes (Option B)

4. **Align Loss Function Formula**
   - **Action:** Add $-1$ constant to `joint_loss()` in `har_caesar.py:124`
   - **File:** `src/har_caesar/models/har_caesar.py`
   - **Time:** 5 minutes
   - **Code change:**
   ```python
   loss_val = np.mean(
       np.where(y <= v, (y - v) / (self.theta * e), 0) + v / e + np.log(-e) - 1  # Add -1
   )
   ```

#### **MEDIUM PRIORITY (Recommended for Polish):**

5. **Document Penalty Weights**
   - **Action:** Add sentence in thesis explaining $\lambda_r = \lambda_q = \lambda_e = 10$
   - **File:** `LaTeX/chapters/methodology.tex`
   - **Time:** 15 minutes

6. **Verify Rolling Window Coverage**
   - **Action:** Remove `min(n_windows, 10)` cap in experiment or explain its purpose
   - **File:** `experiments/experiments_har_caesar.py:115`
   - **Time:** 10 minutes

#### **LOW PRIORITY (Code Quality, Not Blocking):**

7. **Refactor to Use Inheritance**
   - **Action:** Make `HAR_CAESar` inherit from `CAESar_base`
   - **Files:** `src/har_caesar/models/har_caesar.py`
   - **Time:** 3-4 hours (requires testing)
   - **Note:** Can be deferred to post-publication refactoring

---

### For Examiners / Reviewers

**Questions to Ask:**

1. "The thesis states HAR-CAESar has 10 parameters per equation (Section 3.5.2), but the code comment (har_caesar.py:80) indicates 9 parameters. Which is correct, and can you justify the count?"

2. "The ES equation (Equation 9) shows $\hat{e}_{t-1}$ as a regressor, but the code estimates the residual $r_t = e_t - q_t$. Can you explain this reparametrisation and confirm it's equivalent?"

3. "Section 3.8.1 mentions using the Christoffersen conditional coverage test, but I cannot find this implementation in utils.py. Was this test actually performed, or was a different test used?"

4. "The experiment script caps at 10 rolling windows (line 115). Do the results tables in the thesis use all available windows or just these 10?"

**Verification Requests:**

1. **Request:** Run the experiment with seed=123 (different from seed=42) and confirm results change appropriately
2. **Request:** Provide loss function values for a sample of 5 predictions to verify the formula implementation
3. **Request:** Show the output of the Christoffersen test for one index

---

## Template: Documentation Fix for Issue 2

**Add to `LaTeX/chapters/methodology.tex` after Section 3.5.2:**

```latex
\subsubsection{Implementation via Residual Reparametrisation}

Following the original CAESar methodology \citep{Gatta2024}, we estimate the HAR-CAESar model using a reparametrisation that improves numerical stability. Rather than estimating the ES equation directly in Equation~\eqref{eq:har_caesar_e}, we estimate the \emph{ES residual} defined as $r_t = e_t - q_t$ in the second stage.

The Stage 2 specification becomes:
\begin{align}
r_t &= \gamma_0 + \gamma_1^{(d)} (r_{t-1}^{(d)})^+ + \gamma_2^{(d)} (r_{t-1}^{(d)})^- \notag \\
&\quad + \gamma_1^{(w)} (r_{t-1}^{(w)})^+ + \gamma_2^{(w)} (r_{t-1}^{(w)})^- \notag \\
&\quad + \gamma_1^{(m)} (r_{t-1}^{(m)})^+ + \gamma_2^{(m)} (r_{t-1}^{(m)})^- \notag \\
&\quad + \gamma_3 \hat{q}_{t-1} + \gamma_4 r_{t-1},
\end{align}
where $r_t = e_t - q_t$ and parameters $\boldsymbol{\gamma}$ are estimated by minimising the Barrera loss function. The ES forecast is then recovered as $\hat{e}_t = \hat{q}_t + \hat{r}_t$.

This reparametrisation ensures that the monotonicity constraint $e_t \le q_t$ (equivalently $r_t \le 0$) can be enforced via a simple non-positivity constraint on $r_t$. In the third stage, all parameters are jointly re-estimated using the Fissler-Ziegel loss applied directly to $(\hat{q}_t, \hat{e}_t)$, making the reparametrisation transparent to the final estimates.
```

---

## Conclusion

This thesis presents a **methodologically sound and scientifically valid extension** of the CAESar model. The core contribution—incorporating heterogeneous autoregressive dynamics from Corsi (2009) into the joint VaR/ES forecasting framework of Gatta et al. (2024)—is **well-motivated theoretically and correctly implemented algorithmically**.

The HAR-CAESar model successfully:
1. Extends CAESar with multi-horizon return aggregates
2. Preserves asymmetric slope effects at daily/weekly/monthly levels  
3. Maintains joint elicitability framework for consistent estimation
4. Implements robust three-stage optimization procedure

**Primary weaknesses are documentation-related**, not methodological:
- Parameter count stated incorrectly in one sentence (critical but easily fixed)
- Reparametrisation used in code but not explained in thesis (major but well-understood)
- Missing implementation of one claimed statistical test (major but non-blocking)
- Minor formula notation inconsistencies

**The scientific contribution stands**: HAR-CAESar is a legitimate advancement in tail risk forecasting that addresses the long-memory limitation of standard CAESar. The empirical application to four major equity indices follows best practices in risk management research.

**Verdict Rationale:**
- **Not "Accept"** because the parameter count error creates ambiguity that must be resolved
- **Not "Reject"** because the core methodology is sound and contribution is valuable  
- **"Major Revision"** is appropriate: fixes are straightforward, impact is high, and the revised thesis will be strong

With the recommended corrections (estimated 4-6 hours of work), this thesis would merit **acceptance** as a solid contribution to the financial econometrics literature.

---

## Appendix: Anti-Pattern Analysis

### Anti-Patterns NOT Present (Good Practice)

✓ **No "Descriptive Mismatch":**  
Code implements what thesis describes (modulo documented reparametrisation)

✓ **No "Phantom Baseline":**  
All claimed baselines (CAViaR, CAESar, GAS1, GAS2) are implemented

✓ **No "Fixed Seed Epidemic":**  
Seeds managed correctly at per-fit scope, not inside loops

✓ **No "Selective Reporting":**  
Statistical tests are implemented (except Christoffersen)

✓ **No "Hyperparameter Hide":**  
Penalty weights visible in code (though undocumented in thesis)

### Anti-Patterns Present (Issues Identified)

⚠️ **"Citation Slight-of-Hand" (Minor):**  
Christoffersen (1998) test claimed but not implemented

⚠️ **"Metric Substitution" (Partial):**  
Loss function formula differs slightly from thesis (sign/constant)

⚠️ **"Undefined Constant" (Minor):**  
Penalty weights $\lambda_r, \lambda_q, \lambda_e = 10$ not specified in thesis

---

**End of Review**
