Test multiple threshold value for distance, make it CAP parameter?  GUI?
Whole file compare X_PCA[:, 0] and smooth_loop[phase[:],[0]].  GUI?
Speed from phase::

1. Most basic, simple convolution, typically up to 10.000 points. Will average the pauses with the speed ... 2. Change phi so that the speed does not depend on angle (non linearity correction). 3. Detect pauses . To do that I do non linear fitting of the phase, looking every where for the best "flat fit" rather than averaging all the speeds. It separates well traces alternating between pauses and high speeds. Tell me if you want to know more about it.

Test data

Issue with long files>>


feedback to martin? distance threshold and plotting and use traj instead of signal in cycle detection?

for gui allow for indices input
and maybe mannual input of indices within dataset while updating traj to not only account for drifts but just change in traj


update is currently only useful for when there's a drift but not traj change, how about some sort of tree that if the data stray too far we check for another traj?

animation?

Alternatively we can look to redo the reference cycle in case the trajectory has changed into something different by something that's not drift so can't be corrected with a simple update. Maybe we can just do a new reference cycle every minute instead of doing the update (and new ref cycle should handle drift anyway?) or we can use the tree again, if when the tree cannot find any neighbor for the current cycle maybe that's time to switch.  OR we can just wait for gui then we can monitor the change in trajectory and how it matches the reference cycle and mannually make changes when necessary?


I guess allow user to change smoothing factors and other factors can be useful.

Cache everything in memory once computed (raw, smoothed, PCA, ref cycles, segments).
Show timeline of ref cycle evolution: user could even scroll to compare any two segments & their assigned ref cycles.
Later on: allow real-time PCA updates or ref cycle re-calculation with different parameters (e.g., smoothing factor).
Add color-coded drift metric between segment’s PCA and assigned ref cycle.



Resolve .3f issue with tolorance?

4. Detect reference cycle with detect_cycle_bounds
	•	✔ Critical step — this must be before any updates.
	•	This uses first part of PCA data; just make sure enough points exist before updates kick in.

⸻

5. Calculate updated reference cycles (once per second)
	•	✔ Great — these can be stored in a list of (timestamp, ref_cycle).
	•	Timeline for updates = ref0_time + 1s, ref0_time + 2s, …, until end.

⸻

6. Smooth ref and PCA trajectories
	•	✔ Ref cycles: absolutely yes (use smooth_trajectory).
	•	PCA segments: optional. Smoothing them after segmentation might help visual clarity without distorting phase info.
	•	Maybe allow toggle for smoothed vs raw PCA plot?
	•	Optional idea: only smooth when plotting, not in data.

⸻

7. Divide PCA data into segments, assign ref cycle by timestamp
	•	✔ Exactly right.
	•	Associate each segment with the most recent ref cycle based on start time.
	•	If no ref cycle yet at segment time: skip plotting or use initial ref cycle.
	•	Log or mark such segments to know coverage gap.