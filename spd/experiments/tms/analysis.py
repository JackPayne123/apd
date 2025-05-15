# %%
import pandas as pd

df = pd.read_csv("/Users/jack/projects/mech_interp/apd_recreate/apd/wandb_export_2025-05-15T10_50_24.639+10_00.csv")

# %%

print("Impact of lr:")
print(df.groupby('lr')['total_loss'].agg(['mean', 'std']))
print("\nImpact of schatten_coeff:")
print(df.groupby('schatten_coeff')['total_loss'].agg(['mean', 'std']))
print("\nImpact of topk_recon_coeff:")
print(df.groupby('topk_recon_coeff')['total_loss'].agg(['mean', 'std']))



# %%
top_n = 5
print(f"\nTop {top_n} runs by total_loss:")
print(df.sort_values(by='total_loss').head(top_n)[['lr', 'schatten_coeff', 'topk_recon_coeff', 'total_loss']])

# %%


print("\nCorrelations with total_loss:")
print(df[['lr', 'schatten_coeff', 'topk_recon_coeff', 'total_loss']].corr()['total_loss'])
# %%
