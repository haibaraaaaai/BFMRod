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


check variance difference from multi stator to single stator
and from na and proton type
from just when speed hit zero and a while afterwards


degasing buffer to remove oxygen with vacuum chamber
or with syringe to pull out air
SO3 (2-)
3 state literature

use angles with best speeds maybe bearing dynamics will be seen there and can be exculded from residual (stator) dynamics


https://github.com/navishwadhwa/multi-state-remodeling


Read A multi-state dynamic process confers mechano-adaptation to a biological nanomachine.
Get data with good steps, multi stators, and no laser damage.
Compare.

Prep no oxygen samples.
first try only glucose.
then try with oxygen removed solution.

fix display ref cycle
resize pca normailzed signal?


80 vs 0 speed comparison during exchange? exchange behvaiour
dwell time comparison
compare (equilibrium 80mM) with 80 mM at buffer exchange


1. Compare step statistics across SMF conditions

Now that buffer states are labeled:
	•	Split each buffer-exchange trace into segments based on your known change times.
	•	For each segment, extract:
	•	Mean dwell time
	•	Mean step size
	•	Step count
	•	Group segments into:
	•	80 mM
	•	0 mM (low SMF)
	•	Transition (exclude or separately track)
	•	📊 Plot bar charts or box plots:
	•	One plot for dwell time, one for step size
	•	Grouped by SMF condition

👉 This gives a clean answer to: “How does SMF affect stator turnover behavior?”

⸻

2. Repeat for Ficol vs no Ficol

Now that you have Ficol traces, do the same per-segment analysis, but group by:
	•	Ficol (high load)
	•	Normal (low load)

This addresses: “How does load affect step dynamics, at a given SMF?”

You can even do 2×2 grouping:
	•	80 mM + low load
	•	0 mM + low load
	•	80 mM + Ficol
	•	0 mM + Ficol (if enough data)

⸻

3. Step size distribution clustering

Use t-SNE or PCA again, but this time:
	•	Label each point by SMF condition and/or Ficol
	•	See if step shape or size profile separates meaningfully

This avoids relying only on dwell time means.

⸻

4. Highlight individual examples

From the 7 full-paired traces (80 mM + transitions), pick 1–2 clear ones and:
	•	Plot annotated step trace (like you already did)
	•	Highlight that step size drops / dwell time rises with 80→0 switch
	•	Use arrows and buffer labels

This is poster material.

⸻

Bonus

Later: Use speed histograms or kernel density estimates for each condition, to see if speed states shift under different SMF.

only the uniformly going down parts fit the step finding thing?
also i guess with koff we should only focus on the offs?
cells have no brain!
is it micro sensing or when torque is low, high k off rate