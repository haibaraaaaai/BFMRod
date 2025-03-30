Improvements:
    QOL: Switch off preview button when confirm button is hit.

    QOL: Cache for repeated calculation to reduce computation load. (Such as if we are making the same update refs with the same computed refs, we should just store that info somewhere, and only start doing computation when we get to timestamps of new computed ref)

    QOL: Optimize fraction and alpha values.

    QOL: Hide or grey out ref cycle when previewing.

    QOL: Now instead of erase the view when computed ref = 0, we just display segmented data without ref cycles.

    QOL: Increase the speed at which up down arrow keys increase / decrease indices for preview part.

    QOL: Try additional smoothing on computed ref cycles, and try ref detection with variance bias base on components.

    QOL: Allow for a button of no ref or something, basically from where the button is pressed to next ref cycle we use no ref and just pause phase index, this is for the extreme case of pauses in real rotation leading to noise completely taking over and make traj un-trackable.

    QOL: Dynymic threshold trigger for ref cycle update (even manual input trigger) Basically use the same comparison plotted in pca comparison viewer and set it so if peaks in ref misalign with peaks in pca by a factor of 1/4 of the distance from this pca peak to the next or something then we trigger a update perhaps END_OF_CYCLE_LIMIT size before this happens, if still not fixed we flag it and pause later computation and ask for a manual input of ref, then we start from that again (maybe with the help of cache).
    
    QOL: Toggle speed calculation and rev_window input.

    Fixes: PCAViewer cannot allocate 3.71 GiB for an array with shape (497750000,) and data type float64, so need a fix for large data sets, not urgent, but will need later.

    Ideas: Try Chi2 step finding or other step finding / smoothing techinque on speed.

    Ideas: Investigate anisotropy vs total intensity (normalize XYZ to [-1, 1]).

    Ideas: PCA2D.

    Ideas: Pause detection.