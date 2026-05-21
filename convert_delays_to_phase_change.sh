
#!/bin/bash
# Calculate differential tropospheric phase correction from two RAiDER ZTD files.
#
# Usage: ./calc_phase_diff.sh <ref_ztd.nc> <sec_ztd.nc> [output_prefix]
#
# Outputs:
# <prefix>_unwrapped.tif - phase difference in radians (unbounded)
# <prefix>_wrapped.tif - phase difference wrapped to [-pi, pi]

set -e

REF=$1
SEC=$2
PREFIX=${3:-phase_diff}

if [[-z "$REF" || -z "$SEC" ]]; then
    echo "Usage: $0 <ref_ztd.nc> <sec_ztd.nc> [output_prefix]"
    exit    1
fi

# C-band wavelength (metres)
WAVELENGTH=0.0556
SCALE=$(python3 -c "import math; print(4 * math.pi / $WAVELENGTH)")

echo "Reference : $REF"
echo "Secondary : $SEC"
echo "Scale (4pi/lambda): $SCALE"

# Band 1 = surface level (z index 0, height -100 m)
gdal_calc.py \
    -A "NETCDF:${REF}:wet" --A_band=1 \
    -B "NETCDF:${REF}:hydro" --B_band=1 \
    -C "NETCDF:${SEC}:wet" --C_band=1 \
    -D "NETCDF:${SEC}:hydro" --D_band=1 \
    --outfile="${PREFIX}_unwrapped.tif" \
    --calc="(A + B - C - D) * ${SCALE}" \
    --NoDataValue=nan \
    --format=GTiff \
    --co="COMPRESS=LZW" \
    --quiet

echo "Unwrapped phase written to ${PREFIX}_unwrapped.tif"

gdal_calc.py \
    -A  "${PREFIX}_unwrapped.tif"   \
    --outfile="${PREFIX}_wrapped.tif"   \
    --calc="numpy.arctan2(numpy.sin(A), numpy.cos(A))" \
    --NoDataValue=nan \
    --format=GTiff \
    --co="COMPRESS=LZW" \
    --quiet

echo "Wrapped phase written to ${PREFIX}_wrapped.tif"