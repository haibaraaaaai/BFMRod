Development To-Do List
1. Main Priority
    Try Chi2 step finding.

    Toggle speed calculation and rev_window input.

    Check transient pauses paper for pause detection.

2. Other Ideas
    Investigate anisotropy vs total intensity (normalize XYZ to [-1, 1]).

    Dynamic Threshold Trigger for ref cycle update:
        Trigger update if phase error or distance exceeds threshold.

    Backward penalty for indices more than one or two units away?

    Cache for repeat calculation?

    Switch off preview when confirm

    Viewing is not updated?

    Fix update? And more smoothing? Problem is that even exact cycle on pca data is not good ref as it contains lag and noise.

    PCAViewer error unable to allocate 3.71 GiB for an array with shape (497750000,) and data type float64

    Is PCA traj distorted? like if a sample is slightly varied in one pca direction, if it's a minor direction shouldn't that correspond to like a really small insignificant change in real space? but in pca 3d it's treated with equal weight as changes in main dir?
    try pca 2d? might bring back degeneracy?

3. List of doable samples:
    2025.03.13 fileMinArab4
