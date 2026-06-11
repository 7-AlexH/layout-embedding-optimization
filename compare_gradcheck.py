"""External (torch-free) comparator for the Phase 1 gradcheck dumps.

The in-process comparison in gradcheck_phase1.exe is unreliable on Windows
because torch_cpu.dll clobbers callee-saved xmm14/15 and MSVC parks FP
constants there. This script applies the plan's tolerances (atol 1e-9,
rtol 1e-7) to the raw dumped values instead.
"""
import sys

ATOL = 1e-9
RTOL = 1e-7


def load(path):
    with open(path) as f:
        lines = f.read().split("\n")
    loss = float(lines[0].split()[1])
    grads = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        idx, gx, gy = line.split()
        grads[int(idx)] = (float(gx), float(gy))
    return loss, grads


def main():
    dense_loss, dense = load("gradcheck_phase1_dense.txt")
    sparse_loss, sparse = load("gradcheck_phase1_sparse.txt")

    ok = True
    print(f"dense  loss = {dense_loss:.17g}")
    print(f"sparse loss = {sparse_loss:.17g}")
    ld = abs(dense_loss - sparse_loss)
    lr = ld / max(abs(dense_loss), 1e-300)
    print(f"loss abs diff = {ld:.3g}, rel diff = {lr:.3g}")
    if ld > ATOL and lr > RTOL:
        print("FAIL: loss mismatch")
        ok = False

    if set(dense) != set(sparse):
        print(f"FAIL: leaf sets differ (dense {len(dense)}, sparse {len(sparse)})")
        ok = False

    max_abs = max_rel = 0.0
    n_bad = 0
    worst = []
    for idx in sorted(set(dense) & set(sparse)):
        for c in (0, 1):
            a, b = dense[idx][c], sparse[idx][c]
            ad = abs(a - b)
            rd = ad / max(abs(a), 1e-300)
            max_abs = max(max_abs, ad)
            if ad > ATOL:
                max_rel = max(max_rel, rd)
            worst.append((ad, rd, idx, c, a, b))
            if ad > ATOL and rd > RTOL:
                n_bad += 1
    worst.sort(reverse=True)
    print(f"leaves = {len(dense)}, max abs diff = {max_abs:.3g}, "
          f"max rel diff (abs>atol) = {max_rel:.3g}, failures = {n_bad}")
    print("worst 5 by abs diff:")
    for ad, rd, idx, c, a, b in worst[:5]:
        print(f"  leaf {idx}[{c}]: dense {a:.17g} vs sparse {b:.17g} (abs {ad:.3g}, rel {rd:.3g})")
    if n_bad:
        ok = False

    print("\nPHASE 1 GRADCHECK (external):", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
