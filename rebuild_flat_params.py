import csv
from pathlib import Path

folder = Path("hawkes_mle_fixed_beta_lbfgsb_eta010")

mu_file = folder / "fixed_beta_mle_mu.csv"
N_file = folder / "fixed_beta_mle_branching_matrix_N.csv"
out_file = folder / "fixed_beta_mle_params_flat.csv"

def parse_float_cells(row):
    nums = []
    for cell in row:
        cell = str(cell).strip()
        try:
            nums.append(float(cell))
        except ValueError:
            pass
    return nums

# Read mu: 6 rows, one mu value per event type
mu = []
with open(mu_file, newline="") as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        nums = parse_float_cells(row)
        if nums:
            mu.append(nums[-1])

if len(mu) != 6:
    raise RuntimeError(f"Expected 6 mu values, got {len(mu)} from {mu_file}")

# Read N: 6 rows, each with 6 branching values
N = []
with open(N_file, newline="") as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        nums = parse_float_cells(row)
        if len(nums) >= 6:
            N.append(nums[-6:])

if len(N) != 6:
    raise RuntimeError(f"Expected 6 N rows, got {len(N)} from {N_file}")

headers = ["beta", "T", "start_time_seconds_original"]
values = [10.0, 23400.0, 34200.0]

for i in range(6):
    headers.append(f"mu_{i}")
    values.append(mu[i])

for i in range(6):
    for j in range(6):
        headers.append(f"N_{i}{j}")
        values.append(N[i][j])

with open(out_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerow(values)

print(f"Wrote {out_file}")
print(f"mu count: {len(mu)}")
print(f"N shape: {len(N)} x {len(N[0])}")
