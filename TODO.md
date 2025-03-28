Development To-Do List
1. Core Features & Enhancements
    Compare X_PCA[:, 0] vs smooth_loop[phase[:, 0]] across whole file.
        Use as a parameter for auto-redetect-ref?


2. Ref Cycle Logic
    Dynamic Threshold Trigger for ref cycle update:
        Trigger update only if phase error or distance exceeds threshold.
        Input timestamp → run detect_cycle_bounds() around it.

    Phase continuity after manual input:
        Use last assigned phase index before new cycle for continuity.
        In case of manual input:
            Recompute phase assignment but preserve phase continuity.

    Backward penalty for indices more than one or two units away?

3. Other Ideas
    Investigate anisotropy vs total intensity (normalize XYZ to [-1, 1]).

    Check transient pauses paper for pause detection.

    Save file data.

    Toggle speed calculation and rev_window input.

    Try Chi2 step finding.
