import numpy as np
import itertools
import hashlib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================
# Reproducibility
# ============================================================
np.random.seed(0)
torch.manual_seed(0)

# ============================================================
# Bareiss controls (kept outside class to avoid changing your template)
# ============================================================
USE_BAREISS_MINOR_CERT = True          # set False to disable
BAREISS_K_MAX = 6                      # search minors up to k x k
BAREISS_SAMPLES_PER_K = 2000           # random samples per k if not exhaustive
BAREISS_EXHAUSTIVE_LIMIT = 20000       # if (#rowCombos * #colCombos) <= this, do exhaustive

# ============================================================
# Basic checks / utilities
# ============================================================

def checkRule1(A: np.ndarray) -> bool:
    """Check if every element in A is in {-1, 0, 1}."""
    return np.isin(A, [-1, 0, 1]).all()

def det_2x2(A, r0, r1, c0, c1):
    return A[r0, c0] * A[r1, c1] - A[r0, c1] * A[r1, c0]

def check_all_2x2_minors(A: np.ndarray) -> bool:
    """
    Necessary condition for TU (NOT sufficient in general):
    All 2x2 minors must have determinant in {-1,0,1}.
    """
    n, m = A.shape
    if n < 2 or m < 2:
        return True
    for r0 in range(n):
        for r1 in range(r0 + 1, n):
            for c0 in range(m):
                for c1 in range(c0 + 1, m):
                    d = det_2x2(A, r0, r1, c0, c1)
                    if d not in (-1, 0, 1):
                        return False
    return True

def transposeMatrix(A: np.ndarray) -> np.ndarray:
    return A.T

# ============================================================
# Bareiss algorithm (exact determinant) + minor-based non-TU proof
# ============================================================

def _nCk_count(n: int, k: int) -> int:
    """Compute n choose k as int without importing extra modules."""
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    num = 1
    den = 1
    for i in range(1, k + 1):
        num *= (n - (k - i))
        den *= i
    return num // den

def bareiss_det_int(M: np.ndarray) -> int:
    """
    Exact determinant of an integer square matrix using the Bareiss algorithm.
    Returns a Python int (big-int safe).
    """
    M = np.asarray(M)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("bareiss_det_int expects a square matrix")

    n = M.shape[0]
    if n == 0:
        return 1
    if n == 1:
        return int(M[0, 0])

    A = M.astype(object).copy()
    det_sign = 1
    prev_pivot = 1

    for k in range(n - 1):
        # Pivoting if needed
        if A[k, k] == 0:
            swap = None
            for i in range(k + 1, n):
                if A[i, k] != 0:
                    swap = i
                    break
            if swap is None:
                return 0
            A[[k, swap], :] = A[[swap, k], :]
            det_sign *= -1

        pivot = A[k, k]

        for i in range(k + 1, n):
            for j in range(k + 1, n):
                val = A[i, j] * pivot - A[i, k] * A[k, j]
                if k > 0:
                    if prev_pivot == 0:
                        return 0
                    # Bareiss division should be exact
                    if val % prev_pivot != 0:
                        raise ArithmeticError(
                            f"Bareiss non-exact division at k={k}: {val} / {prev_pivot}"
                        )
                    val //= prev_pivot
                A[i, j] = val

        # Clean up
        for i in range(k + 1, n):
            A[i, k] = 0
        for j in range(k + 1, n):
            A[k, j] = 0

        prev_pivot = pivot

    return int(det_sign * A[n - 1, n - 1])

def find_nonTU_minor_bareiss(
    A: np.ndarray,
    k_max: int = 6,
    samples_per_k: int = 2000,
    exhaustive_limit: int = 20000,
    seed: int = 0,
):
    """
    Try to find a k×k minor with |det| > 1 (this is a PROOF of NOT TU).
    Returns (det, rows, cols) if found, else None.
    """
    n, m = A.shape
    if min(n, m) < 3:
        return None

    rng = np.random.RandomState(seed)
    k_max = min(k_max, n, m)

    for k in range(3, k_max + 1):
        rcount = _nCk_count(n, k)
        ccount = _nCk_count(m, k)
        total = rcount * ccount

        if total <= exhaustive_limit:
            # Exhaustive search
            for rows in itertools.combinations(range(n), k):
                for cols in itertools.combinations(range(m), k):
                    sub = A[np.ix_(rows, cols)]
                    d = bareiss_det_int(sub)
                    if abs(d) > 1:
                        return (int(d), tuple(rows), tuple(cols))
        else:
            # Random sampling
            for _ in range(samples_per_k):
                rows = tuple(sorted(rng.choice(n, k, replace=False)))
                cols = tuple(sorted(rng.choice(m, k, replace=False)))
                sub = A[np.ix_(rows, cols)]
                d = bareiss_det_int(sub)
                if abs(d) > 1:
                    return (int(d), rows, cols)

    return None

# ============================================================
# STRUCTURE-AWARE CERTIFICATES (fast proofs for common families)
# ============================================================

def peel_unit_rows(A: np.ndarray):
    """
    Split A into:
      - core: rows that are NOT unit rows
      - unit_rows_idx: indices of unit rows
    Unit row = exactly one nonzero, and that entry is ±1.
    """
    unit_rows = []
    core_rows = []
    for i in range(A.shape[0]):
        nz = np.flatnonzero(A[i])
        if len(nz) == 1 and abs(int(A[i, nz[0]])) == 1:
            unit_rows.append(i)
        else:
            core_rows.append(i)
    return A[core_rows, :], unit_rows

def is_node_arc_incidence(A: np.ndarray) -> bool:
    """
    Node–arc incidence (directed graph): each column has exactly one +1 and one -1.
    Such matrices are TU.
    """
    if not checkRule1(A):
        return False
    n, m = A.shape
    for j in range(m):
        col = A[:, j]
        if np.count_nonzero(col) != 2:
            return False
        if np.count_nonzero(col == 1) != 1:
            return False
        if np.count_nonzero(col == -1) != 1:
            return False
    return True

def is_graph_incidence_01(A: np.ndarray) -> bool:
    """
    0/1 incidence of an undirected graph: each column has exactly two 1's.
    """
    if not np.isin(A, [0, 1]).all():
        return False
    n, m = A.shape
    for j in range(m):
        if np.count_nonzero(A[:, j]) != 2:
            return False
        if np.count_nonzero(A[:, j] == 1) != 2:
            return False
    return True

def bipartite_check_from_incidence(A: np.ndarray):
    """
    If A is 0/1 graph incidence, build the graph on row-nodes and check bipartiteness.
    Returns (is_bipartite, colors or None).
    """
    if not is_graph_incidence_01(A):
        return (False, None)

    n, m = A.shape
    adj = [[] for _ in range(n)]
    for j in range(m):
        rows = np.flatnonzero(A[:, j])
        u, v = int(rows[0]), int(rows[1])
        adj[u].append(v)
        adj[v].append(u)

    color = [-1] * n
    for start in range(n):
        if color[start] != -1:
            continue
        color[start] = 0
        queue = [start]
        while queue:
            u = queue.pop()
            for v in adj[u]:
                if color[v] == -1:
                    color[v] = 1 - color[u]
                    queue.append(v)
                elif color[v] == color[u]:
                    return (False, None)
    return (True, color)

def has_TU_certificate(A: np.ndarray):
    """
    Return (True, details) if we detect a known TU certificate.

    IMPORTANT FIX:
    - First try certificates on the full matrix.
    - If that fails, peel unit rows (e_i^T rows) and try again on the core.
      Appending unit rows preserves TU, so TU(core) => TU(full A).
    """
    # 1) Direct node-arc incidence on whole A
    if is_node_arc_incidence(A):
        return (True, {"type": "node_arc_incidence_full"})

    # 2) Direct bipartite 0/1 incidence on whole A
    is_bip, colors = bipartite_check_from_incidence(A)
    if is_bip:
        return (True, {"type": "01_graph_incidence_bipartite_full", "row_colors": colors})

    # 3) Peel unit rows and retry
    core, unit_rows_idx = peel_unit_rows(A)
    if len(unit_rows_idx) > 0:
        if is_node_arc_incidence(core):
            return (True, {
                "type": "node_arc_incidence_after_peel_unit_rows",
                "peeled_unit_rows": unit_rows_idx,
                "core_shape": core.shape
            })
        is_bip2, colors2 = bipartite_check_from_incidence(core)
        if is_bip2:
            return (True, {
                "type": "01_graph_incidence_bipartite_after_peel_unit_rows",
                "peeled_unit_rows": unit_rows_idx,
                "core_shape": core.shape,
                "row_colors": colors2
            })

    return (False, {})

def has_nonTU_certificate(A: np.ndarray):
    """
    Return (True, details) if we detect a known NON-TU certificate.
    """
    # If A is 0/1 graph incidence but NOT bipartite, it is NOT TU.
    if is_graph_incidence_01(A):
        is_bip, _ = bipartite_check_from_incidence(A)
        if not is_bip:
            return (True, {"type": "01_graph_incidence_nonbipartite"})
    return (False, {})

# ============================================================
# Exact TU check for small row count using Ghouila–Houri (3^n)
# ============================================================

def ghouila_houri_exact(A: np.ndarray, max_rows: int = 12):
    """
    Exact TU check using Ghouila–Houri via enumeration over all subsets+signings.

    GH theorem: A is TU iff for every subset R of rows, there exists a signing (+/-)
    of the rows in R such that every column sum is in {-1,0,1}.

    We enumerate assignments per row in {0, +1, -1}:
      0   => row not in subset
      +1  => row in subset with + sign
      -1  => row in subset with - sign

    This enumerates 3^n combinations and marks which subsets are "satisfied".
    A is TU iff all subsets are satisfied.

    Returns:
      - True  : TU (proved)
      - False : Not TU (proved)
      - None  : skipped (too many rows)
    """
    n, m = A.shape
    if n > max_rows:
        return None

    A_int = A.astype(int, copy=False)

    satisfied = np.zeros(1 << n, dtype=bool)
    sums0 = np.zeros(m, dtype=int)

    def rec(i: int, mask: int, sums: np.ndarray):
        if satisfied.all():
            return
        if i == n:
            if np.all((sums >= -1) & (sums <= 1)):
                satisfied[mask] = True
            return

        row = A_int[i]

        rec(i + 1, mask, sums)                 # 0
        rec(i + 1, mask | (1 << i), sums + row) # +1
        rec(i + 1, mask | (1 << i), sums - row) # -1

    rec(0, 0, sums0)

    return bool(satisfied.all())

# ============================================================
# Feature Extraction (deterministic)
# ============================================================

def _stable_seed_from_matrix(A: np.ndarray) -> int:
    h = hashlib.md5(A.tobytes()).hexdigest()
    return int(h[:8], 16)

def extract_features(A: np.ndarray) -> np.ndarray:
    """
    Feature vector for NN (deterministic).
    """
    n, m = A.shape
    features = []

    # Dimensions
    features += [float(n), float(m), float(n * m)]

    # Sparsity and entry stats
    features.append(float(np.mean(A == 0)))
    features.append(float(np.mean(A == 1)))
    features.append(float(np.mean(A == -1)))
    features.append(float(np.mean(np.abs(A))))
    features.append(float(np.std(A)))

    # Row/col sum stats
    row_sums = np.sum(A, axis=1)
    col_sums = np.sum(A, axis=0)
    features += [
        float(np.mean(row_sums)), float(np.std(row_sums)), float(np.max(np.abs(row_sums))),
        float(np.mean(col_sums)), float(np.std(col_sums)), float(np.max(np.abs(col_sums))),
    ]

    # Structure features
    nnz_per_col = np.count_nonzero(A, axis=0)
    features.append(float(np.mean(nnz_per_col == 2)))

    if checkRule1(A):
        plus_per_col = np.sum(A == 1, axis=0)
        minus_per_col = np.sum(A == -1, axis=0)
        features.append(float(np.mean((nnz_per_col == 2) & (plus_per_col == 1) & (minus_per_col == 1))))
    else:
        features.append(0.0)

    if np.isin(A, [0, 1]).all():
        ones_per_col = np.sum(A == 1, axis=0)
        features.append(float(np.mean((nnz_per_col == 2) & (ones_per_col == 2))))
    else:
        features.append(0.0)

    # Deterministic sample of 2x2 det violations
    if n >= 2 and m >= 2:
        rng = np.random.RandomState(_stable_seed_from_matrix(A))
        samples = min(80, (n * (n - 1) // 2) * (m * (m - 1) // 2))
        det_viol = 0
        for _ in range(samples):
            r = rng.choice(n, 2, replace=False)
            c = rng.choice(m, 2, replace=False)
            d = det_2x2(A, r[0], r[1], c[0], c[1])
            if abs(d) > 1:
                det_viol += 1
        features.append(float(det_viol / samples) if samples > 0 else 0.0)
    else:
        features.append(0.0)

    # Rank
    features.append(float(np.linalg.matrix_rank(A)))

    return np.array(features, dtype=np.float32)

# ============================================================
# Neural model
# ============================================================

class TUClassifier(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

# ============================================================
# Data generation (aligned with your matrices)
# ============================================================

def generate_TU_matrix(max_n: int, max_m: int) -> np.ndarray:
    matrix_type = np.random.choice(['node_arc', 'bipartite_01', 'interval'])

    if matrix_type == 'node_arc':
        n = np.random.randint(2, max_n + 1)
        m = np.random.randint(2, max_m + 1)
        A = np.zeros((n, m), dtype=int)
        for col in range(m):
            u, v = np.random.choice(n, 2, replace=False)
            A[u, col] = 1
            A[v, col] = -1
        return A

    if matrix_type == 'bipartite_01':
        n_total = np.random.randint(4, max_n + 1)
        m_edges = np.random.randint(2, max_m + 1)
        n_left = np.random.randint(1, n_total)
        n_right = n_total - n_left
        A = np.zeros((n_total, m_edges), dtype=int)
        for e in range(m_edges):
            u = np.random.randint(0, n_left)
            v = np.random.randint(0, n_right)
            A[u, e] = 1
            A[n_left + v, e] = 1
        return A

    # interval
    n = np.random.randint(2, max_n + 1)
    m = np.random.randint(2, max_m + 1)
    A = np.zeros((n, m), dtype=int)
    for i in range(n):
        start = np.random.randint(0, m)
        length = np.random.randint(1, min(6, m - start + 1))
        A[i, start:start+length] = 1
    return A

def generate_non_TU_matrix(max_n: int, max_m: int) -> np.ndarray:
    kind = np.random.choice(['odd_cycle', 'has_2', 'random'])

    if kind == 'odd_cycle':
        cycle_len = int(np.random.choice([3, 5, 7]))
        n = cycle_len
        m = cycle_len
        A = np.zeros((n, m), dtype=int)
        for e in range(m):
            u = e
            v = (e + 1) % n
            A[u, e] = 1
            A[v, e] = 1
        return A

    if kind == 'has_2':
        n = np.random.randint(2, max_n + 1)
        m = np.random.randint(2, max_m + 1)
        A = np.random.choice([0, 1, -1], size=(n, m), p=[0.7, 0.2, 0.1])
        i = np.random.randint(0, n)
        j = np.random.randint(0, m)
        A[i, j] = 2
        return A

    n = np.random.randint(2, max_n + 1)
    m = np.random.randint(2, max_m + 1)
    A = np.random.choice([0, 1, -1, 2], size=(n, m), p=[0.65, 0.2, 0.1, 0.05])
    return A

def generate_training_data(num_samples=5000, max_n=15, max_m=25):
    X, y = [], []
    while len(X) < num_samples:
        if np.random.random() < 0.5:
            A = generate_TU_matrix(max_n, max_m)
            label = 1
        else:
            A = generate_non_TU_matrix(max_n, max_m)
            label = 0

        feats = extract_features(A)
        X.append(feats)
        y.append(label)

    return np.array(X), np.array(y)

# ============================================================
# Training
# ============================================================

def train_model(num_samples=8000, epochs=60, max_n=15, max_m=25):
    print("Generating training data...")
    X, y = generate_training_data(num_samples=num_samples, max_n=max_n, max_m=max_m)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).reshape(-1, 1)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test).reshape(-1, 1)

    model = TUClassifier(input_size=X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training neural network...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_t)
                test_loss = criterion(test_outputs, y_test_t)
                preds = (test_outputs > 0.5).float()
                acc = (preds == y_test_t).float().mean()
                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"Loss={loss.item():.4f} TestLoss={test_loss.item():.4f} Acc={acc.item():.4f}")

    return model, scaler

# ============================================================
# Hybrid TU Checker (FIXED LOGIC)
# ============================================================

class HybridTUChecker:
    def __init__(self, model=None, scaler=None, gh_max_rows: int = 12):
        self.model = model
        self.scaler = scaler
        self.use_nn = (model is not None and scaler is not None)
        self.gh_max_rows = int(gh_max_rows)

    def nn_score(self, A: np.ndarray) -> float:
        if not self.use_nn:
            return float("nan")
        feats = extract_features(A)
        feats_scaled = self.scaler.transform(feats.reshape(1, -1))
        feats_tensor = torch.FloatTensor(feats_scaled)
        self.model.eval()
        with torch.no_grad():
            return float(self.model(feats_tensor).item())

    def check(self, A: np.ndarray, method='hybrid', confidence_threshold=0.8):
        """
        Returns:
          - is_TU: True / False / None
            True  => proved TU
            False => proved NOT TU
            None  => inconclusive (NN hint only)
        """
        results = {
            'is_TU': None,
            'confidence': 0.0,
            'methods_used': [],
            'details': {}
        }

        # Rule1 quick fail => NOT TU proved
        if not checkRule1(A):
            results['details']['rule1'] = False
            results['is_TU'] = False
            results['confidence'] = 1.0
            results['methods_used'].append('rule1')
            results['details']['status'] = "proved_not_TU (entry outside {-1,0,1})"
            return results
        results['details']['rule1'] = True

        # Non-TU certificate (proof)
        nonTU, nonTU_info = has_nonTU_certificate(A)
        results['details']['nonTU_certificate'] = nonTU_info if nonTU else None
        if nonTU:
            results['is_TU'] = False
            results['confidence'] = 1.0
            results['methods_used'].append('nonTU_certificate')
            results['details']['status'] = "proved_not_TU (certificate)"
            return results

        # TU certificate (proof)  <-- now includes peeling unit rows
        tu_cert, tu_info = has_TU_certificate(A)
        results['details']['TU_certificate'] = tu_info if tu_cert else None
        if tu_cert:
            results['is_TU'] = True
            results['confidence'] = 1.0
            results['methods_used'].append('TU_certificate')
            results['details']['status'] = "proved_TU (certificate)"
            if method in ('hybrid', 'all', 'nn_only') and self.use_nn:
                s = self.nn_score(A)
                results['details']['nn_prediction'] = s
                results['methods_used'].append('neural_network')
            return results

        # ============================================================
        # Bareiss: minor certificate for NOT TU (proof if found)
        # (Added without changing your class/template signature)
        # ============================================================
        if USE_BAREISS_MINOR_CERT and min(A.shape[0], A.shape[1]) >= 3:
            minor = find_nonTU_minor_bareiss(
                A,
                k_max=BAREISS_K_MAX,
                samples_per_k=BAREISS_SAMPLES_PER_K,
                exhaustive_limit=BAREISS_EXHAUSTIVE_LIMIT,
                seed=_stable_seed_from_matrix(A),
            )
            results['details']['bareiss_minor_certificate'] = minor
            results['methods_used'].append('bareiss_minor_search')
            if minor is not None:
                detv, rows, cols = minor
                results['is_TU'] = False
                results['confidence'] = 1.0
                results['details']['status'] = f"proved_not_TU (Bareiss minor |det|>1, det={detv})"
                return results
        else:
            results['details']['bareiss_minor_certificate'] = None

        # Exact GH check if small enough (proof)
        gh = ghouila_houri_exact(A, max_rows=self.gh_max_rows)
        results['details']['ghouila_houri_exact'] = gh
        if gh is True:
            results['is_TU'] = True
            results['confidence'] = 1.0
            results['methods_used'].append('ghouila_houri_exact')
            results['details']['status'] = "proved_TU (Ghouila–Houri exact)"
            return results
        if gh is False:
            results['is_TU'] = False
            results['confidence'] = 1.0
            results['methods_used'].append('ghouila_houri_exact')
            results['details']['status'] = "proved_not_TU (Ghouila–Houri exact)"
            return results
        # else gh is None => skipped

        # NN screening (hint only)
        if method in ('hybrid', 'nn_only', 'all') and self.use_nn:
            s = self.nn_score(A)
            results['details']['nn_prediction'] = s
            results['methods_used'].append('neural_network')

            if method == 'nn_only':
                results['is_TU'] = (s > 0.5)
                results['confidence'] = s if results['is_TU'] else (1.0 - s)
                results['details']['status'] = "nn_only (NOT a proof)"
                return results

        # Necessary checks (still not proofs)
        minors_ok = check_all_2x2_minors(A)
        minors_ok_T = check_all_2x2_minors(A.T)
        results['details']['all_2x2_minors_ok'] = minors_ok
        results['details']['all_2x2_minors_ok_transpose'] = minors_ok_T
        results['methods_used'].append('2x2_minors')
        results['methods_used'].append('transpose_2x2_minors')

        if not minors_ok or not minors_ok_T:
            results['is_TU'] = False
            results['confidence'] = 1.0
            results['details']['status'] = "proved_not_TU (violated 2x2 minor necessary condition)"
            return results

        # Inconclusive: passed necessary checks, but no proof
        nn_pred = results['details'].get('nn_prediction', None)
        if nn_pred is not None:
            results['confidence'] = float(nn_pred) if nn_pred > 0.5 else float(1.0 - nn_pred)
            results['details']['status'] = "inconclusive (passed necessary checks; NN hint only)"
        else:
            results['confidence'] = 0.50
            results['details']['status'] = "inconclusive (passed necessary checks; no proof)"

        results['is_TU'] = None
        return results

    def detailed_report(self, A: np.ndarray):
        print("=" * 70)
        print("TOTALLY UNIMODULAR MATRIX ANALYSIS (Fixed Hybrid)")
        print("=" * 70)
        print(f"Matrix shape: {A.shape[0]} x {A.shape[1]}")
        print(f"Elements: {A.size}")
        print(f"Sparsity: {np.mean(A == 0) * 100:.2f}%")
        print()

        res = self.check(A, method='all')
        print("RESULT:")
        print(f"Is TU: {res['is_TU']}   (True=proved TU, False=proved NOT TU, None=inconclusive)")
        print(f"Confidence: {res['confidence'] * 100:.2f}%")
        print()
        print("DETAILS:")
        for k, v in res['details'].items():
            print(f"  {k}: {v}")
        print()
        print("Methods used:", ", ".join(res['methods_used']))
        print("=" * 70)
        return res

# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    # Your matrix (as provided)
    A = np.array([
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [-1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, -1, -1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, -1, -1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, -1, -1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    ], dtype=int)

    print("Training neural network model...")
    model, scaler = train_model(num_samples=8000, epochs=60, max_n=15, max_m=25)
    print()

    checker = HybridTUChecker(model, scaler, gh_max_rows=12)
    checker.detailed_report(A)
