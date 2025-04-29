# TODO

## Ideas
- Testing the detached, attached off, attached on three state model by looking at the speed trace variance within a step and variance between steps.
- Checking for bearing noise and locked speed (still has a stator but no longer rotate) by comparing the signal difference just when it stops / after a few seconds.

## GUI

### Core Ideas
- Quick X+iY diagnostic viewer; alert when speed is doubled but variance is also high.
- Pause detection.
- Try additional smoothing on computed reference cycles.
- Try Chi² step detection or other step/smoothing techniques on speed.
- Try interpolating phase / projecting PCA onto reference cycle instead of using nearest neighbor.
- Investigate anisotropy vs total intensity (normalize XYZ to [-1, 1]).
- Explicit linear closure correction: morph trajectory so last point returns to first.

### Quality of Life
- Allow different `update_interval`, `fraction`, and `alpha` for each computed reference.
- Save `fraction`, `alpha`, and `update_interval` to JSON.
- Try assigning index base using variance or std bias.
- Use smaller smoothing for reference cycle updates.
- Toggle speed calculation and `rev_window` input.

### Fixes
- Investigate "jumping index" when adding computed ref between two existing refs.
- Fix PCAViewer crash: cannot allocate 3.71 GiB for array (shape (497750000,)); needed for large datasets (not urgent).

### Advanced Ideas
- Dynamic threshold trigger for reference cycle update:
  - Monitor misalignment between reference cycle and PCA trajectory.
  - If misalignment exceeds ¼ of peak distance, trigger update `END_OF_CYCLE_LIMIT` early.
  - If not fixed, flag and pause, request manual reference input.
  - Resume processing from new manual ref using cache to skip unchanged segments.

- Segmented PCA.

## Tools

### Ideas
- Plot speed from multiple angular regions and compare trends to average speed.
  - Goal: separate angular variance from temporal variance.


angle polar/linear moive? for shape of bearing.
try different anlge bin sizess

average over entire traces 26 peaks?

bio archive paper

intermediate averaging see changing of the peaks?

try 100 angles? 
fourier transform of the distribution give periodicity with 26

normalize to average for the polar plot to remove pmf stuff?
what happens when a speed is skipped because less than 2 points are in that range?
does not make sense to even try 100 steps for 200 phase points. 

pca on angluar speeds?

need to care for search range in near neighbour search if we increase ref number