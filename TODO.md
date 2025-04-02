Improvements:
    QOL: Allow for a button of No update, where for a input start and end time, we do not update ref cycle, but need to make sure everything runs smoothly between and after this time, so phase assignment need to continue during update, new phase assignment after NO UPDATE (and during NO UPDATE) need to use the the last ref (computed or updated) to assign phase, need to check also if a computed ref is passed during this period, in which case we need to swap to the computed ref, this is for the extreme case of pauses in real rotation leading to noise completely taking over and make traj un-trackable.
    Or just allow segmented update and fraction and alpha value and set update to the entire duration or longer

    QOL: Reduce npz file size!

    QOL: Allow for different update and fraction and alpha for each computed refs.

    QOL: Try assigning index base with variance / std bias?

    QOL: Run Segment-Based PCA?

    QOL: Preview More Segments?

    QOL: Change to a smaller smooth for updates.    

    QOL: Also save in json fraction, alpha, update interval

    Fixes: Cache jumping index? Weird stuff from adding computed ref bewteen 2 computed refs.

    Fixes: PCAViewer cannot allocate 3.71 GiB for an array with shape (497750000,) and data type float64, so need a fix for large data sets, not urgent, but will need later.

    Ideas: Try Chi2 step finding or other step finding / smoothing techinque on speed.

    Ideas: Investigate anisotropy vs total intensity (normalize XYZ to [-1, 1]).

    Ideas: Pause detection.

    Ideas: Optimize fraction and alpha values.

    Ideas: Toggle speed calculation and rev_window input.

    Ideas: Dynymic threshold trigger for ref cycle update (even manual input trigger) Basically use the same comparison plotted in pca comparison viewer and set it so if peaks in ref misalign with peaks in pca by a factor of 1/4 of the distance from this pca peak to the next or something then we trigger a update perhaps END_OF_CYCLE_LIMIT size before this happens, if still not fixed we flag it and pause later computation and ask for a manual input of ref, then we start from that again (maybe with the help of cache).

    Ideas: Try additional smoothing on computed ref cycles.

    Ideas: Explicit linear correction to enforce closure, by progressively "morphing" the trajectory to bring the last point back to the first.
