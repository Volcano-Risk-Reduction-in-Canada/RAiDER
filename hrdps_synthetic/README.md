# hrdps_synthetic: atmosphere-only synthetic interferograms from HRDPS

This folder generates synthetic interferograms that contain **only tropospheric
phase**, for training a deformation detector alongside synthetic magmatic-source
data. It is a standalone addition. It imports RAiDER from the installed
environment and does not modify any RAiDER file.

Each sample holds the differential slant tropospheric delay between two dates,
expressed as RCM C-band phase.

- **Layout.** By default each sample is in **radar geometry**, the same array
  layout as the synthetic magmatic interferograms: 3000 rows along-track by
  2000 columns across-track at 10 m. The two can therefore be summed pixel for
  pixel. North-up UTM rasters are still available.
- **Products.** Each sample has unwrapped phase, wrapped phase, and a
  **false-colour RGB** of the wrapped phase, which is now a standard output.
- **Size and chips.** The scene size is adjustable, and scenes can be cut into
  fixed-size chips such as 500 × 500 pixels for Detectron2.

See [Conventions](#conventions) and [Scene size and chips](#scene-size-and-chips).

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

# 3. Optional: chips from scenes that already exist
python -m hrdps_synthetic.tile --workdir /data/hrdps_synth --chip-size 500 --dry-run
python -m hrdps_synthetic.tile --workdir /data/hrdps_synth --chip-size 500 --workers 8

# 4. Optional: RGB for wrapped phase written before RGB was standard (scenes and chips)
python -m hrdps_synthetic.rgb --workdir /data/hrdps_synth --dry-run
python -m hrdps_synthetic.rgb --workdir /data/hrdps_synth --workers 8
```

Epochs (`epochs/`) and DEMs (`dem/`) always live in the workdir. `--output-dir`
(generate, tile, rgb) puts the plan, manifests, samples and chips in a separate
folder. That way one download can serve several renderings, for example the
older north-up run and a new radar-geometry run:

```bash
python -m hrdps_synthetic.generate --workdir /data/hrdps_synth --output-dir /data/hrdps_synth/radar
```

The MSC Datamart keeps only the **latest ~30 days**. One download run collects
everything available; no scheduled job is needed. To add newer days later,
rerun the download within 30 days, then rerun `generate --replan`.

Every top-level command has `--dry-run`, which reports downloads, disk use and
run time and writes nothing.

## Settings file

Every option can also be set in a YAML file passed with `--config`. All defaults
and command-line options stay as they are. Precedence, lowest to highest:
**built-in defaults < settings file < command line.**

```bash
python -m hrdps_synthetic.config --example > my_settings.yaml   # annotated template with all defaults
python -m hrdps_synthetic.config --check my_settings.yaml        # validate for every command
python -m hrdps_synthetic.download --config my_settings.yaml
python -m hrdps_synthetic.generate --config my_settings.yaml --limit 5 --quicklook 5   # command line still wins
python -m hrdps_synthetic.generate --config my_settings.yaml --print-config           # show resolved settings
```

```yaml
workdir: /data/hrdps_synth   # shared by download, generate and tile
width_km: 20.0               # sizes download's weather crops and generate's scenes
chip_size: [500]             # top level: generate makes chips, and tile uses the same size
generate:
  n_train: 8500
  n_val: 1500
  no_scenes: true            # chips only
download:
  workers: 8
```

How the file is read:

- **Keys.** Every key is a command-line option name; dashes and underscores both work.
- **Top level and sections.** Top-level keys apply to every command that has
  that option. The `download:`, `generate:` and `tile:` sections apply to one
  command only and take precedence over top-level keys.
- **null.** `null`, or leaving a key out, keeps the built-in default.
- **Validation.** Unknown keys, wrong types, invalid choices and wrong value
  counts are rejected with the key named.
- **Times.** `14:43` reads as a time, not the YAML 1.1 base-60 integer, quoted or not.
- **Switching flags back off.** A flag switched on in the file can be switched
  off on the command line: `--no-keep-grib`, `--no-keep-hourly`,
  `--no-grib-only`, and `--scenes` to undo `no_scenes: true`.
- **Saved settings.** Each real run writes its resolved settings, with the
  command line and config path, to `<workdir>/settings_<command>.yaml`.

The template is `config.example.yaml` in this folder. A test checks that its
values stay equal to the built-in defaults.

## Conventions

These are chosen on their own merits. The magmatic repository is used only as a
reference for the colour ramp and the file layout.

| Item | Convention |
|---|---|
| Phase | `phase = 4π/λ × (STD_sec − STD_ref)`. Positive = more delay, i.e. longer range, on the later date, the same sign as subsidence. **No change = 0 rad.** No referencing. |
| Wrapped phase | The unwrapped phase reduced to [−π, π): `mod(phase + π, 2π) − π`, so 0 stays 0. |
| Wavelength | 0.055466 m, from c / 5.405 GHz (RCM). |
| RGB | The 9-stop cyclic ramp of `applyEVPhaseColorRamp.m`, position = (wrapped + π) / 2π. **0 rad = red (255, 0, 0)** at the ramp midpoint, ±π = teal (0, 191, 169). 3-band uint8 GeoTIFF, bands named red/green/blue. |
| Radar layout | Rows along-track in flight order (row 0 = start of the swath), columns across-track from near range (column 0, satellite side) to far range: the usual SLC orientation. Ascending scenes look flipped north–south and descending scenes mirrored east–west, compared with a map. |
| Radar georeferencing | No CRS. The transform is the magmatic files' local metric frame: origin at the swath centre, x = −W/2 at column 0, y = +L/2 at row 0. The real location is in `manifest.csv` (centre, corners, UTM EPSG, flight azimuth) and in the GeoTIFF tags. |
| LOS vector | Unit vector from the ground to a right-looking satellite, in east-north-up; look azimuth = heading + 90°. It matches the magmatic `los_vector`. |
| North-up layout | `--geometry north-up`: UTM rasters, 20 km E–W × 30 km N–S. |

**Older outputs.** Runs from before 2026-10-15 used north-up layout and the
opposite sign, `4π/λ × (STD_ref − STD_sec)`, recorded in each file's
`convention` tag. `python -m hrdps_synthetic.rgb` renders their RGBs in the
current sign (`--sign auto`) and leaves their float products as written. A
radar-geometry version of an old run has to be re-rendered (about 3 s per
scene, same seed and plan), because north-up scenes cannot be rotated without
losing their corners. `generate` refuses to mix conventions in one output
folder.

**Issues noticed in the magmatic repository.** These are not copied here:

- `projection/los.py: phase_to_radians` maps zero displacement to −π, and with
  the opposite sign to the sweep. It feeds the public `compute_interferogram()`.
  The sweep (`sweep/simulate.py: build_products`) and the Dash app are correct:
  0 → 0 rad, positive = range increase.
- `constants.py` sets `WAVELENGTH_RCM = 0.05405` m. That looks like the
  5.405 GHz frequency written as centimetres; the true value is 0.055466 m. Its
  comment and the MATLAB origin use a fringe of 0.028333 m (λ = 0.0567 m), so
  three different wavelengths are in play. Magmatic fringes are therefore about
  2.6 % too dense relative to this atmosphere.
- Rows are evaluated from along-track −L/2 at row 0, but the GeoTIFF transform
  puts row 0 at y = +L/2. The manifest's `along_track_km` therefore has the
  opposite sign to the raster's y coordinate. Rasters and polygons agree with
  each other.
- Large sources (metres of LOS motion) alias at 10 m pixels. Real data would
  decorrelate there.

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
- **Delay.** ΔZTD = (wet + hydro)_sec − (wet + hydro)_ref is interpolated
  trilinearly at each pixel's latitude, longitude and GLO-30 height (10 m UTM,
  bilinear from 30 m, WGS84 ellipsoidal as in RAiDER). Then
  ΔSTD = ΔZTD / cos(incidence), the projected mapping RAiDER's `Conventional`
  LOS uses. Ray tracing is not used until it has been validated.
- **Phase.** unwrapped = 4π/λ × ΔSTD with λ = 0.055466 m (RCM, 5.405 GHz).
  There is **no referencing**; the absolute differential delay is kept.
  wrapped = mod(unwrapped + π, 2π) − π. See [Conventions](#conventions).
- **RGB.** The wrapped phase on the cyclic ramp, written as `<id>_rgb.tif`
  (`--no-rgb` to skip).

## Scene size and chips

**Scene size.** Set it in kilometres or in pixels; pixels win when both are given.

```bash
python -m hrdps_synthetic.generate --workdir W --width-km 20 --length-km 30      # default, 2000 x 3000 px
python -m hrdps_synthetic.generate --workdir W --width-px 500 --height-px 500   # 5 x 5 km scenes
python -m hrdps_synthetic.generate --workdir W --width-px 1000 --height-px 1000 --spacing 20
```

- Scene size is part of the plan. Changing it needs `--replan`, because volcano
  shift ranges and overlap groups depend on it.
- A scene can only be as large as the weather crops the download made. The
  download's `--width-km`/`--length-km` (default 20 × 30) set those crops, so
  pass the largest size you plan to render there. If a requested scene does not
  fit, `generate` stops and names the AOIs. Smaller scenes always fit.
- An AOI's DEM is rebuilt automatically if it is too small for a larger scene size.

**Chips during generation.** `--chip-size W [H]` cuts every scene into chips as
it is rendered.

```bash
python -m hrdps_synthetic.generate --workdir W --chip-size 500                  # scenes + 24 chips per scene
python -m hrdps_synthetic.generate --workdir W --chip-size 500 --no-scenes      # chips only
python -m hrdps_synthetic.generate --workdir W --chip-size 512 --chip-stride 256 --chip-edge shift
```

**Chips from existing scenes.** `python -m hrdps_synthetic.tile` takes the same
chip options and works from `manifest.csv`, so you can try other chip sizes
without re-rendering. It produces exactly the chips `generate --chip-size`
would; a test checks this.

How chips work:

- **Stride.** The default stride equals the chip size, so chips do not overlap.
  A 2000 × 3000 scene gives 4 × 6 = 24 chips of 500 × 500.
- **Edges.** With `--chip-edge drop` (default), a strip narrower than a chip is
  left out. With `shift`, one more chip is placed flush with the edge and
  overlaps its neighbour.
- **Splits.** Every chip inherits its scene's train/validation split, so no
  atmosphere crosses splits. Chips of one scene share a date pair but cover
  different ground.
- **Metadata.** Each chip is a georeferenced float32 GeoTIFF (unwrapped and
  wrapped phase). It records its own incidence at the centre and the range
  across the chip, and its LOS vector at the centre, both in `chips.csv` and in
  the GeoTIFF tags.
- **Folders.** Each chip configuration gets its own folder (`chips_500x500`,
  `chips_512x512_s256x256_shift`, ...), so several can coexist.
- **One output setting per run.** A workdir refuses to switch `--chip-size` or
  `--no-scenes` partway through a plan. Use `tile` for other chip sizes.
- **Storage.** Chips take about as much space as the scenes they come from, so
  scenes plus chips roughly doubles disk use. `--dry-run` measures it.
- **Detectron2.** Its default loader reads 8-bit images, while these chips are
  float32 radians, so they can be summed with your magmatic chips first. Read
  them with a custom dataset mapper (rasterio), or convert them after combining.

### Outputs

```
<workdir>/epochs/, <workdir>/dem/<AOI>_glo30_utm<spacing>m.tif   shared inputs
<out>/plan.csv, plan_meta.json, outputs_meta.json   deterministic plan (seeded) and output settings
<out>/settings_<command>.yaml              resolved settings of the last run of each command
<out>/samples/<split>/<AOI>/<id>_unw.tif   unwrapped phase, float32 radians
<out>/samples/<split>/<AOI>/<id>_wrp.tif   wrapped phase, float32 radians in [-pi, pi)
<out>/samples/<split>/<AOI>/<id>_rgb.tif   false-colour RGB, 3 x uint8
<out>/manifest.csv                         one row per finished sample
<out>/chips_<W>x<H>[...]/chips.csv         one row per chip (with --chip-size or tile)
<out>/chips_<W>x<H>[...]/<split>/<AOI>/<id>_rRR_cCC_{unw,wrp,rgb}.tif
<out>/quicklook/<id>.png                   with --quicklook N
```
`<out>` is `--output-dir`, which defaults to the workdir.

Sample ids are `<split>_<AOI>_<asc|desc>_<refdate>_<secdate>`. Besides the plan
columns, the manifest records:

- the look azimuth and the incidence range in the scene;
- the **LOS unit vector (ground → satellite, ENU)** at the scene centre;
- the layout, flight azimuth in the UTM grid, scene centre (lat/lon and UTM),
  corner coordinates, EPSG code, origin and size;
- the phase sign convention;
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
- chip windows and edge modes, chip incidence and georeferencing, and chips from
  `generate --chip-size` matching chips from `tile`;
- settings files: precedence, sections, validation, unquoted times, flag
  negation, the example matching the defaults, and saved run settings;
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
