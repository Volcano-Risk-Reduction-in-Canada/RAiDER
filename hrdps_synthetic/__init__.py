"""Synthetic atmosphere-only interferograms from HRDPS (standalone; imports RAiDER read-only).

Commands (run from the repository root):

    python -m hrdps_synthetic.aois       # AOI table, KMZ export, ocean check
    python -m hrdps_synthetic.download   # HRDPS backfill -> per-AOI epoch delay cubes
    python -m hrdps_synthetic.generate   # sample plan -> GeoTIFF interferograms

See README.md in this folder.
"""

__all__ = ['aois', 'sampling', 'hrdps_archive', 'geometry', 'dem', 'slant', 'download', 'generate']
