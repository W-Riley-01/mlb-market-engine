import pandas as pd

# Check the vault
vault = pd.read_parquet('./data/master_physics_vault.parquet', columns=None)
print("VAULT columns related to woba:")
print([c for c in vault.columns if 'woba' in c.lower() or 'estimated' in c.lower()])
print()

# Check the current CQM (pre-env)
cqm = pd.read_parquet('./data/contact_matrix.parquet')
print("CQM columns related to woba:")
print([c for c in cqm.columns if 'woba' in c.lower() or 'estimated' in c.lower()])
print()

# Check the env-merged CQM (what the resolver actually reads)
cqm_env = pd.read_parquet('./data/contact_matrix_env.parquet')
print("CQM_ENV columns related to woba:")
print([c for c in cqm_env.columns if 'woba' in c.lower() or 'estimated' in c.lower()])G.")