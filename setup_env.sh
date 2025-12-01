#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="rvt_env"
PYTHON_VERSION="3.11"

conda env remove -n "${ENV_NAME}" -y || true
conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -c conda-forge --strict-channel-priority -y
conda activate "${ENV_NAME}"

conda install -y -c conda-forge \
  gdal \
  rasterio \
  xdem \
  libnetcdf \
  netcdf4 \
  geopandas \
  pyproj \
  numpy \
  scipy \
  pillow

python -m pip install --upgrade pip setuptools wheel
pip install \
  rvt-py \
  py3dep \
  pynhd \
  pygeohydro