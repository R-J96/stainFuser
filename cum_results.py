# %%
import pandas as pd

sf = pd.read_csv('output/fsimCompSFDebug2.csv')
mac = pd.read_csv('output/fsimCompMacDebug2.csv')
nst = pd.read_csv('output/fsimCompNstDebug2.csv')
# %%
sf_other = pd.read_csv('output/sfRes.csv')
mac_other = pd.read_csv('output/macRes.csv')
nst_other = pd.read_csv('output/nstRes.csv')
# %%
sf = pd.merge(sf, sf_other, on=['idx', 'ssim_sf'])
mac = pd.merge(mac, mac_other, on=['idx', 'ssim_macenko'])
nst = pd.merge(nst, nst_other, on=['idx', 'ssim_nst'])
sf = sf.reset_index()
mac = mac.reset_index()
nst = nst.reset_index()
# %%
indices = sf.nlargest(125, columns='ssim_sf').index
indices_large = sf.nlargest(250, columns='ssim_sf').index
# %%
sf.iloc[indices].mean()
# %%
mac.iloc[sf.nlargest(125, columns='ssim_sf').idx].mean()
# %%
mac.nlargest(500, 'ssim_macenko').mean()
# %%
sf.iloc[indices_large].mean()
# %%
