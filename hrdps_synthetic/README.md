# hrdps_synthetic: atmosphere-only synthetic interferograms from HRDPS

This folder generates synthetic interferograms that contain **only tropospheric
phase**, for training a deformation detector alongside synthetic magmatic-source
data. It is a standalone addition. It imports RAiDER from the installed
environment and does not modify any RAiDER file.

Each sample is a north-up 20 km (E–W) × 30 km (N–S) raster at 10 m spacing
(2000 × 3000 pixels) in the AOI's UTM zone. It holds the differential slant
tropospheric delay between two dates, expressed as RCM C-band phase.

## Requirements

- The RAiDER conda environment from branch `add_ecc_weather_model`. The code imports
  `RAiDER.models.hrdps`, `RAiDER.models.weatherModel`, `RAiDER.cli.raider`,
  `RAiDER.delayFcns` and `RAiDER.losreader`, so it is tied to this branch's versions.
- Network access to `dd.weather.gc.ca` (HRDPS) and the Copernicus GLO-30 bucket (DEM).
- Commands run **from the repository root**, so `hrdps_synthetic` is importable.

## Workflow

```bash
# 0. AOIs: table, KMZ for review, ocean check of every footprint
python -m hrdps_synthetic.aois --list
python -m hrdps_synthetic.aois --kmz                # writes hrdps_synthetic/synthetic_aois.kmz
python -m hrdps_synthetic.aois --check-land         # GLO-30 reads, ~25 s per AOI

# 1. HRDPS backfill -> per-AOI epoch delay cubes (rerun any time; finished work is skipped)
python -m hrdps_synthetic.download --workdir /data/hrdps_synth --dry-run
python -m hrdps_synthetic.download --workdir /data/hrdps_synth --workers 8

# 2. Sample plan -> GeoTIFFs
python -m hrdps_synthetic.generate --workdir /data/hrdps_synth --dry-run
python -m hrdps_synthetic.generate --workdir /data/hrdps_synth --limit 5 --quicklook 5   # look first
python -m hrdps_synthetic.generate --workdir /data/hrdps_synth --workers 16
```

The MSC Datamart keeps only the **latest ~30 days**. One download run collects
everything available; no scheduled job is needed. To add newer days later,
rerun the download within 30 days, then rerun `generate --replan`.

Every top-level command has `--dry-run`, which reports downloads, disk use and
run time and writes nothing.

## What the pipeline does

### 1. Download (`download.py`, `hrdps_archive.py`)

For each archived day and each pass direction, the downloader builds one epoch
per AOI at the mean acquisition time: **ascending 02:02 UTC** and
**descending 14:43 UTC**. These means come from the RCM SLC records in the VRRC
API dump, and you can change them with `--asc-time` and `--desc-time`.

1. Download the 84 GRIBs (TMP, SPFH, HGT on 28 pressure levels) for each
   bracketing hour. The hours are 02/03 UTC (00Z run +2/+3 h) and 14/15 UTC
   (12Z run +2/+3 h). Downloads use RAiDER's `_fetch_grib`.
2. Decode each GRIB once with eccodes. Regrid it from HRDPS's rotated lat/lon
   grid onto a **true regular lat/lon grid** for every AOI. The regridding is
   bilinear in rotated-pole index space, using the grid definition in the GRIB.
3. Run RAiDER's HRDPS weather model on each regridded hour (`load` then `write`).
   This gives zenith wet and hydrostatic delay cubes on a fixed height grid,
   from −100 m to 20.5 km with 50 m steps below 3 km.
4. Blend the two hours to the target minute with RAiDER's `combine_weather_files`.
   Keep only `wet_total` and `hydro_total`, as float32, in
   `epochs/<AOI>/<asc|desc>/HRDPS_<AOI>_<YYYYMMDDTHHMM>.nc`.

GRIBs and hourly files are deleted as the run goes. Peak transient disk use is
about 150 MB of GRIBs.

### 2. Generate (`generate.py`, `sampling.py`, `geometry.py`, `slant.py`, `dem.py`)

- **No duplicates.** Each (AOI, direction, date pair) is rendered exactly once,
  with one random incidence and one random scene shift.
- **Validation holds out both AOIs and dates.** Training samples use training
  AOIs on training dates only, and validation samples use validation AOIs on
  validation dates only. AOIs whose scenes can overlap stay in the same split:
  Garibaldi with Cayley, Edziza North with South, and Hoodoo with Lava Fork.
  The number of validation dates is chosen to minimise the AOIs needed.
- **Geometry.** The heading is 348° (ascending) or 192° (descending), and the
  radar looks right, so the look azimuth is heading + 90°. Scene-centre
  incidence is drawn from U(20°, 55°), and incidence varies across track on a
  spherical Earth with the satellite at 600 km. That is about 1.5–2.5° across a
  north-up scene; the corners of the rotated look geometry extend the
  cross-track span past 20 km.
- **Placement.** Volcano scenes are shifted at random while the AOI polygon
  stays inside. Meager's polygon is 24 km wide, so its east-west shift is
  limited to ±1 km. The 36 point AOIs shift up to ±5 km on each axis.
- **Delay.** ΔZTD = (wet + hydro)_ref − (wet + hydro)_sec is interpolated
  trilinearly at each pixel's latitude, longitude and GLO-30 height (10 m UTM,
  bilinear from 30 m, WGS84 ellipsoidal as in RAiDER). Then
  ΔSTD = ΔZTD / cos(incidence), the projected mapping RAiDER's `Conventional`
  LOS uses. Ray tracing is not used until it has been validated.
- **Phase.** unwrapped = 4π/λ × ΔSTD with λ = 0.055466 m (RCM, 5.405 GHz).
  There is **no referencing**; the absolute differential delay is kept.
  wrapped = mod(unwrapped + π, 2π) − π. The sign convention matches
  `convert_delays_to_phase_change.sh` (reference − secondary).

### Outputs

```
<workdir>/plan.csv, plan_meta.json         deterministic plan (seeded)
<workdir>/samples/<split>/<AOI>/<id>_unw.tif   float32 radians, DEFLATE, 512 tiles
<workdir>/samples/<split>/<AOI>/<id>_wrp.tif
<workdir>/manifest.csv                     one row per finished sample
<workdir>/dem/<AOI>_glo30_utm10m.tif       per-AOI 10 m DEM covering every scene position
<workdir>/quicklook/<id>.png               with --quicklook N
```

Sample ids are `<split>_<AOI>_<asc|desc>_<refdate>_<secdate>`. Besides the plan
columns, the manifest records:

- the look azimuth and the incidence range in the scene;
- the **LOS unit vector (ground → satellite, ENU)** at the scene centre;
- the scene centre (lat/lon and UTM), EPSG code, origin and size;
- DEM range and ΔSTD statistics;
- the output paths.

To project magmatic deformation onto the same LOS, use
`d_los = d_east·los_east + d_north·los_north + d_up·los_up`, which is positive
toward the satellite. Recompute per pixel with `geometry.incidence_field` if you
need the incidence ramp.

## Measured on 2026-10-13 (Docker `raider_dev`, ~2 MB/s link)

| Item | Value |
|---|---|
| Archive on the Datamart | 30 days (2026-09-14 to 2026-10-13), all cycles, forecast hours 0–48 |
| GRIB download per valid hour | 84 files, ~140 MB, ~75 s |
| Full backfill (45 AOIs, 30 days, both directions) | 2,700 epochs, ~16.6 GB downloaded, ~2.2 GB kept |
| Per day and direction (3 AOIs) | ~4–5 min, dominated by download |
| Epoch file | ~0.8 MB (zlib float32, 134 heights) |
| Sample (both GeoTIFFs) | ~26 MB compressed, 48 MB uncompressed |
| 10,000 samples | ~260 GB, ~8–9 CPU-hours (~1.5 s render + compression each) |
| Per-AOI 10 m DEM | 20–30 MB, ~25–40 s to build |

**Cross-check against RAiDER's own HRDPS path.** Meager, 2026-10-12 02:02 UTC,
`raider.py` cube with `center_time` interpolation, compared at 0–2500 m:

- Wet delay: mean difference 0.0 mm. Scatter is 4 mm (std) at sea level and
  0.4 mm at 2500 m, from the corrected georeferencing.
- Hydrostatic delay: a uniform −6 mm offset at every height. The dense fixed
  height grid integrates the upper troposphere more accurately than RAiDER's
  27 levels. A uniform offset cancels between dates.

## AOIs

There are 45 AOIs in `synthetic_aois.kmz`:

- **9 volcano targets.** Their polygons ship in `volcano_aois.geojson`, extracted
  from `test/HRDPS_testing_August2026/api_dump.json`, which is not tracked by git.
- **36 BC centroids** spread across BC's terrain and climate regimes: Vancouver
  Island and Haida Gwaii, South Coast, Southern Interior, Kootenays and Rockies,
  Central and Coast Mountains, Northwest, and Northeast. None of their scenes
  can overlap each other or the volcanoes.

With 29 usable days, the planned split (seed 0) puts 18 AOIs in validation:
Garibaldi, Cayley, Edziza North and South, and 14 BC centroids spread over
every region. Use `--seed` to draw a different split. The split recorded in the
KMZ is a preview. `generate` recomputes it from the dates actually on disk.

**Why 36 new AOIs.** With about 29 usable days, the best split is 10 validation
dates and 19 training dates. 8,500 training and 1,500 validation samples then
need 25 training and 17 validation AOIs, 42 in total, which is 33 new. Three
more cover a day lost to a partial purge or a missing forecast hour.

## Validation and tests

```bash
python -m pytest -p no:cacheprovider hrdps_synthetic/tests                      # offline, a few seconds
HRDPS_TEST_GRIB=<any HRDPS GRIB2> HRDPS_NETWORK_TESTS=1 python -m pytest hrdps_synthetic/tests   # + GRIB and DEM checks
```

The tests cover:

- the rotated-grid transform against pyproj and against cfgrib's coordinates for
  the full grid (agreement within 1e-5°);
- regridding of an analytic field;
- interpolation against `RAiDER.delayFcns.getInterpolators`, and the projection
  and phase wrapping;
- the incidence ramp and LOS signs;
- plan uniqueness, disjoint splits and determinism;
- `--dry-run` writing nothing;
- the KMZ round trip;
- a small end-to-end render.

## Things to know

- **Turbulence.** HRDPS (2.5 km) cannot resolve turbulence below about 10–15 km,
  so samples are smoother than real interferograms.
- **Projected mapping.** It ignores horizontal gradients along the ray. At 55°
  the wet-layer ray travels about 15 km sideways.
- **One season.** All dates come from one ~30-day window.
- **Model top.** HRDPS stops at 50 hPa (about 20.5 km), so absolute zenith
  hydrostatic delay is about 11 cm short. This is nearly identical between
  dates and largely cancels in the difference.
- **RAiDER issues worked around here, without editing RAiDER:**
  - `RAiDER.models.weatherModel.checkContainment_raw` returns False for any
    regional raw file. That makes `prepareWeatherModel` re-download HRDPS through
    RAiDER's own reader every time, so the synthetic path calls the weather
    model's `load` and `write` directly.
  - `RAiDER.models.hrdps.download_hrdps_file` treats the rotated grid as regular,
    using row and column means. Grid rows tilt 6–16° from true parallels in BC,
    which misplaces data by kilometres at crop edges.
  - `RAiDER.losreader.inc_hd_to_enu` puts east = sin(i)·cos(h+90°). That is not
    a clockwise-from-north heading convention, so the generator computes the
    LOS vector itself and takes only cos(incidence), which is identical either way.
