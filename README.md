# Major_Project
# What is “downscaling” for satellite air quality?
<img width="300" height="168" alt="image" src="https://github.com/user-attachments/assets/19da3fa4-3b67-4206-a28f-bb8689cf55e6" />

**Downscaling** means taking coarse-resolution satellite products (e.g., 1–10 km) and producing higher resolution surface-level pollutant maps (e.g., 100–1000 m) suitable for city or neighborhood analysis. You combine satellite measurements (AOD, tropospheric NO₂, etc.), meteorology, land-use and ground monitor observations with ML/DL models (random forest, XGBoost, CNN/U-Net, super-resolution nets) to learn the mapping from coarse columns to local surface concentrations. This approach is widely used to provide high-resolution PM₂.₅/NO₂ maps where ground monitors are sparse. ([ACS Publications][1], [PMC][2])

---

# Typical data sources (start here)

* **TROPOMI / Sentinel-5P** — tropospheric NO₂, SO₂, etc. (high spatial fidelity for columns). ([Sentinel Online][3], [cmr.earthdata.nasa.gov][4])
* **MODIS MAIAC (MCD19A2)** — aerosol optical depth (AOD) at \~1 km (useful predictor for PM₂.₅). ([Google for Developers][5], [modis.gsfc.nasa.gov][6])
* **OpenAQ / national monitoring networks** — in situ PM₂.₅, PM₁₀, NO₂ ground observations for training/validation. ([openaq.org][7], [OpenAQ Docs][8])
* **Meteorology / reanalysis** — ERA5 or MERRA variables (temperature, RH, wind) as predictors. (Useful in most models.)
* **Land-use / roads / population / elevation** — static predictors for spatial patterns (LUR features).
* Satellite Level-2/Level-3 archives and Earth Engine make access easier. ([Google for Developers][5], [cmr.earthdata.nasa.gov][9])

---

# High-level project plan (practical steps)

1. **Define target & area** — pollutant (PM₂.₅ or NO₂), spatial extent (city/region), temporal resolution (daily, monthly).
2. **Collect data**

   * Download satellite AOD or tropospheric columns for your dates. (TROPOMI/MODIS/MAIAC). ([Sentinel Online][3], [Google for Developers][5])
   * Get ground monitor data from OpenAQ or local EPA equivalents. ([OpenAQ Docs][8])
   * Download meteorology/reanalysis for same dates.
3. **Preprocess**

   * Reproject & resample all data to common grid(s): coarse grid (satellite) and target fine grid.
   * Match satellite pixels to nearest ground station observations (spatio-temporal pairing).
   * Feature engineering: local land-use buffers, distance to major road, elevation, meteorology lags.
4. **Modeling approaches** *(choose one or combine)*

   * **Statistical/ML**: Random Forest / XGBoost using satellite AOD/columns + met + LU features to predict station PM₂.₅, then apply model to whole city grid to downscale. Good baseline. ([ACS Publications][1])
   * **Super-resolution / CNN / U-Net**: Use coarse satellite maps + auxiliary features as input to a U-Net to predict high resolution PM₂.₅ maps (learns spatial patterns). Recent work shows good results in urban contexts. ([SpringerLink][10], [ScienceDirect][11])
   * **Hybrid**: First cal/align satellite to surface (ML), then apply super-resolution CNN for spatial sharpening. ([American Meteorological Society Journals][12])
5. **Train / validate**

   * Use cross-validation that is spatially stratified (leave-locations-out) to avoid overfitting to monitor clusters.
   * Metrics: RMSE, MAE, R² and spatial error maps; also uncertainty estimates (quantile regression or ensembles). ([PMC][13])
6. **Produce maps & analyze**

   * Apply trained model to the high-resolution target grid and visualize (GeoTIFF / web map).
7. **Document, test, and iterate** — try different models, more predictors, or different loss functions.

---

# Minimal runnable code skeleton (Python)

Below is a compact pipeline skeleton you can paste into a Jupyter notebook. It shows: (A) fetch/prepare (assumes you have downloaded rasters and station CSV), (B) train a RandomForest baseline, (C) predict to high-res grid. This is a **starter** — production requires more careful preprocessing and hyperparameter tuning.

```python
# Requirements (conda/pip): rasterio xarray rioxarray geopandas pandas scikit-learn numpy joblib matplotlib
import os, numpy as np, pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# --- 1) Load station data (OpenAQ or local) ---
stations = pd.read_csv("stations_pm25.csv", parse_dates=["date"])
# columns: station_id, lat, lon, date, pm25

# --- 2) Load a satellite raster (coarse) and a high-res baseline raster for target grid ---
sat_path = "satellite_aod_coarse.tif"   # e.g., daily MODIS/MAIAC AOD aggregated
tgt_ref = "highres_baseline.tif"        # an empty raster that defines high-res grid (transform/shape/CRS)

def read_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1)
        meta = src.meta.copy()
    return arr, meta

sat_arr, sat_meta = read_raster(sat_path)
tgt_arr, tgt_meta = read_raster(tgt_ref)

# --- 3) Sample satellite values at station locations ---
import rasterio.sample
from shapely.geometry import Point

sta_gdf = gpd.GeoDataFrame(stations, geometry=gpd.points_from_xy(stations.lon, stations.lat), crs="EPSG:4326")
# reproject station points to raster CRS if needed
if sta_gdf.crs != sat_meta['crs']:
    sta_gdf = sta_gdf.to_crs(sat_meta['crs'])

samples = []
with rasterio.open(sat_path) as src:
    for geom in sta_gdf.geometry:
        for val in src.sample([(geom.x, geom.y)]):
            samples.append(val[0])
sta_gdf['sat_aod'] = samples

# Merge back with original stations (match dates if needed)
df = pd.DataFrame(sta_gdf.drop(columns='geometry'))
df = df[['station_id','date','lat','lon','pm25','sat_aod']].dropna()

# --- 4) Train/test split (quick) ---
X = df[['sat_aod']]   # add meteorology / landuse cols for better performance
y = df['pm25']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 5) Random Forest baseline ---
rf = RandomForestRegressor(n_estimators=200, max_depth=15, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)), "R2:", r2_score(y_test, pred))

# --- 6) Apply model to whole high-res grid ---
# Resample sat_arr -> highres grid (nearest or bilinear)
resampled = np.empty((tgt_meta['height'], tgt_meta['width']), dtype=np.float32)
reproject(
    source=sat_arr,
    destination=resampled,
    src_transform=sat_meta['transform'],
    src_crs=sat_meta['crs'],
    dst_transform=tgt_meta['transform'],
    dst_crs=tgt_meta['crs'],
    resampling=Resampling.bilinear
)
# predict per-pixel (reshape)
flat = resampled.flatten()
mask = np.isfinite(flat)
pred_flat = np.full(flat.shape, np.nan)
pred_flat[mask] = rf.predict(flat[mask].reshape(-1,1))
pred_map = pred_flat.reshape(tgt_meta['height'], tgt_meta['width'])

# Save predicted map
out_meta = tgt_meta.copy()
out_meta.update({"dtype":"float32", "count":1})
with rasterio.open("predicted_pm25_highres.tif","w", **out_meta) as dst:
    dst.write(pred_map.astype(np.float32), 1)

# Save model
joblib.dump(rf, "rf_pm25_model.joblib")
```

**Notes & next steps:**

* Replace the single predictor `sat_aod` with a feature matrix: AOD, TROPOMI NO₂, ERA5 wind/T/ humidity, distance-to-road, landuse fractions, time features (day of year).
* For spatial ML you must handle missing values (cloud gaps) — use gap-filling (temporal interpolation) or translate to a two-step model (ML to calibrate, then CNN to spatially sharpen).
* For deep approaches, create input patches of coarse data + static covariates and train a U-Net to output high-resolution PM₂.₅. Recent papers show state-of-the-art gains. ([SpringerLink][10], [ScienceDirect][11])

---

# Suggested ML/DL architectures

* **Random Forest / XGBoost / LightGBM** — robust baseline, interpretable feature importance. Good for limited compute. ([ACS Publications][1])
* **U-Net / ResUNet (encoder-decoder CNNs)** — learn multi-scale spatial patterns; pair coarse maps + static high-res covariates as inputs. Recent "AirQ-ResUNet" style papers use residual U-Net for urban PM₂.₅. ([SpringerLink][10])
* **Ensembles / Stacked models** — combine RF + CNN for better bias/variance tradeoff. ([PMC][13])

---

# Practical tips & gotchas

* **Cloud gaps** in AOD & satellite products: either mask days, fill with temporal interpolation, or use model architectures that can accept missing channels.
* **Train/test leakage**: use spatial cross-validation (hold out regions/monitors), not random splits.
* **Unit mismatch**: satellite AOD/columns are not surface concentrations — ML model learns relationship but include meteorology and landuse to aid the mapping. ([ACS Publications][1])
* **Scale & compute**: High-res city-wide daily maps may require chunking and parallel IO (dask/xarray).
* **Uncertainty**: use quantile forests, bootstrap ensembles, or Bayesian deep nets to quantify uncertainty.

---

# Curated links (read these first)

**Data & APIs**

* Sentinel-5P TROPOMI (product info & access). ([Sentinel Online][3], [cmr.earthdata.nasa.gov][4])
* MODIS MAIAC (MCD19A2) on Google Earth Engine (1 km AOD). ([Google for Developers][5], [modis.gsfc.nasa.gov][6])
* NASA MODIS AOD catalog / Earthdata. ([NASA Earthdata][14])
* OpenAQ — global ground monitor API & docs. ([openaq.org][7], [OpenAQ Docs][8])
* TROPOMI docs & tools (Tropomi.eu). ([tropomi.eu][15])

**Papers & methods**

* Estimating PM2.5 with Random Forest + AOD + meteorology (classic approach). ([ACS Publications][1])
* Ensemble and high-resolution PM2.5 mapping (example, 1 km resolution). ([PMC][2])
* Recent DL works (AirQ-ResUNet; multiscale UNet improvements). ([SpringerLink][10], [ScienceDirect][11])
* A general downscaling/downscaling-calibration method paper (random forest spatial downscaling examples). ([HESS][16])

---

# Learning resources & tutorials

* **Google Earth Engine (GEE)** — great for quick satellite data retrieval and pre-aggregation (look up MAIAC and MODIS datasets in GEE). Example dataset page: MCD19A2. ([Google for Developers][5])
* **OpenAQ docs & examples** — how to fetch ground data: OpenAQ docs. ([OpenAQ Docs][8])
* **Hands-on workflows / example repos** — search GitHub for “PM2.5 downscaling random forest” or “MAIAC PM2.5” for Jupyter notebooks that demonstrate end-to-end pipelines (modeling + mapping). (Start with keywords from the papers above.)

---

# If you want, I can (pick one)

* **A.** Turn the above skeleton into a full Jupyter notebook with downloadable files (data-loading cells for GEE, preprocessing, training, and mapping).
* **B.** Build a U-Net notebook (patch generator, model, training loop, evaluation) for super-resolution downscaling.
* **C.** Walk through a specific city/region (e.g., Delhi) using public data — I’ll find the exact datasets & write the code to produce a first high-res daily PM₂.₅ map.

Tell me which one you want and I’ll produce the full notebook / code next. (If you pick C, tell me the region and pollutant.)

---

## Quick recap / most important starter links

* Sentinel-5P TROPOMI — product & download. ([Sentinel Online][3], [Data.gov][17])
* MODIS MAIAC (MCD19A2) on Google Earth Engine. ([Google for Developers][5])
* OpenAQ API (ground monitors). ([OpenAQ Docs][8])
* RF/ML PM2.5 mapping paper (method example). ([ACS Publications][1])
* Recent deep models (AirQ-ResUNet, MDS-UNet examples). ([SpringerLink][10], [ScienceDirect][11])

---

If you want I’ll now: **(A)** produce a full runnable Jupyter notebook covering data download (GEE), preprocessing, RF baseline, and saving outputs, or **(B)** produce a U-Net based super-resolution notebook. Which do you want?

[1]: https://pubs.acs.org/doi/10.1021/acs.est.7b01210?utm_source=chatgpt.com "Estimating PM2.5 Concentrations in the Conterminous United States ..."
[2]: https://pmc.ncbi.nlm.nih.gov/articles/PMC7063579/?utm_source=chatgpt.com "An Ensemble-based Model of PM2.5 Concentration across the ..."
[3]: https://sentinels.copernicus.eu/data-products/-/asset_publisher/fp37fc19FN8F/content/sentinel-5-precursor-level-2-nitrogen-dioxide?utm_source=chatgpt.com "Sentinel-5 Precursor Level 2 Nitrogen Dioxide"
[4]: https://cmr.earthdata.nasa.gov/search/concepts/C1442068511-GES_DISC.html?utm_source=chatgpt.com "Sentinel-5P TROPOMI Tropospheric NO2 1-Orbit L2 7km x 3.5km V1 ..."
[5]: https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MCD19A2_GRANULES?utm_source=chatgpt.com "MCD19A2.061: Terra & Aqua MAIAC Land Aerosol Optical Depth ..."
[6]: https://modis.gsfc.nasa.gov/data/dataprod/mod04.php?utm_source=chatgpt.com "MODIS Aerosol Product - NASA"
[7]: https://openaq.org/?utm_source=chatgpt.com "OpenAQ"
[8]: https://docs.openaq.org/about/about?utm_source=chatgpt.com "About the API - OpenAQ Docs"
[9]: https://cmr.earthdata.nasa.gov/search/concepts/C2839237275-GES_DISC.html?utm_source=chatgpt.com "HAQAST Sentinel-5P TROPOMI Nitrogen Dioxide (NO2) CONUS ..."
[10]: https://link.springer.com/chapter/10.1007/978-3-031-97992-7_3?utm_source=chatgpt.com "AirQ-ResUNet: A Residual U-Net Based Deep Learning Surrogate ..."
[11]: https://www.sciencedirect.com/science/article/abs/pii/S026974912402061X?utm_source=chatgpt.com "Improving PM2.5 and PM10 predictions in China from WRF_Chem ..."
[12]: https://journals.ametsoc.org/view/journals/aies/3/4/AIES-D-24-0028.1.xml?utm_source=chatgpt.com "A Deep Learning Framework for Satellite-Derived Surface PM 2.5 ..."
[13]: https://pmc.ncbi.nlm.nih.gov/articles/PMC7643812/?utm_source=chatgpt.com "Ensemble-Based Deep Learning for Estimating PM2.5 over ..."
[14]: https://www.earthdata.nasa.gov/dashboard/data-catalog/modis-aod?utm_source=chatgpt.com "MODIS Aerosol Optical Depth (AOD) (Select Events) - NASA Earthdata"
[15]: https://www.tropomi.eu/documents-and-information?utm_source=chatgpt.com "Documents and information | TROPOMI Observing Our Future"
[16]: https://hess.copernicus.org/articles/25/5667/2021/hess-25-5667-2021.pdf?utm_source=chatgpt.com "[PDF] Easy-to-use spatial random-forest-based downscaling-calibration ..."
[17]: https://catalog.data.gov/dataset/sentinel-5p-tropomi-tropospheric-no2-1-orbit-l2-7km-x-3-5km-v1-s5p-l2-no2-at-ges-disc/resource/efe0e86c-d033-4b0e-b772-76863bf362c7?utm_source=chatgpt.com "Sentinel-5P TROPOMI Tropospheric NO2 1-Orbit L2 7km x 3.5km V1 ..."
