"""Synthetic atmosphere-only interferograms from HRDPS (standalone; imports RAiDER read-only).

Commands (run from the repository root):

    python -m hrdps_synthetic.aois       # AOI table, KMZ export, ocean check
    python -m hrdps_synthetic.download   # HRDPS backfill -> per-AOI epoch delay cubes
    python -m hrdps_synthetic.generate   # sample plan -> GeoTIFF interferograms (+ optional chips)
    python -m hrdps_synthetic.tile       # chips from scenes already rendered
    python -m hrdps_synthetic.config     # settings-file template and validation (--config on every command)

See README.md in this folder.
"""

__all__ = ['aois', 'sampling', 'hrdps_archive', 'geometry', 'dem', 'slant', 'chips', 'config', 'download', 'generate', 'tile']
