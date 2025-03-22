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

3. Dynamic Threshold Update Trigger
Only update the reference cycle if:

Phase assignment error exceeds threshold.
Distance from PCA segment to reference is large.
This prevents updates when trajectory is unstable or noise-dominated.

So issues are 1.we need to now re-update ref cycle base on input ref cycle 1s after input ref.
2. manual input means new phase assignment but we still want to have continuity in phase
3. how do we actually input index that just does not make sense for 250k sampling rate to pick out exactly a start and end point that forms one exact cycle on pca, might be better to input a timestamp and run a  new ref cycle detection (not update) base on the timestamp.
4.need to be done on pca 3d so need to add buttons and stuff. For continuous checking probably better to let go of segment duration (since we are using 1s segmetn for update segment duration is really not that useful besides visualzation so perhaps we can instead plot all data points on a window with the same window control as the other viewer files. But then the issue is that if pyqtgraph is good at plotting a discontinuous ref cycle on top of this. and the idea would be pick the ref cycle whose midpoint is closest to the mid point of the window. 5 in case two windows are equally close need to decide in code what to do (not a big deal) 6 perhaps add a manual ref cycle button that performs a ref detection base on midpoint of current window, the new ref cycle is then immediately plotted, and all future ref need to now be updated base on this. perhaps we can also keep tabs of all the added ref cycles and can remove them or add back and see how that cahnges data and whatever
