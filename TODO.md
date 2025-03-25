Development To-Do List
1. Core Features & Enhancements
    Compare X_PCA[:, 0] vs smooth_loop[phase[:, 0]] across whole file.

2. GUI Plans
    Input indices or timestamps for manual ref cycle creation.
        Add “Use Midpoint of Current Window” button to detect new ref cycle.

    Add ref cycle management UI:
        Add/remove ref cycles.
        Toggle visibility.
        Re-compute phase/speed.
        Optional: tagging (e.g., "test", "good").

3. Data Handling / Optimization
    Cache computed data in memory.

    Resolve .3f precision issue.

    Smooth phase data. Speed at different phases?

4. Ref Cycle Logic
    Dynamic Threshold Trigger for ref cycle update:
        Trigger update only if phase error or distance exceeds threshold.
        Input timestamp → run detect_cycle_bounds() around it.

    Phase continuity after manual input:
        Use last assigned phase index before new cycle for continuity.
        In case of manual input:
            Recompute phase assignment but preserve phase continuity.
    
    Index input may not align well with cycle — prefer timestamps.

    GUI must enable this from PCA 3D viewer.
        Add manual ref cycle button (detect cycle from current window, plot immediately).
        Update all future ref cycles based on this new one.
        Maintain a list of added ref cycles for toggle/remove.

5. Code Maintenance / Testing
    Test modulation algorithms.
        Enable module reloading for function updates.

    Test numba acceleration.

    Exponential weighting for neighboring search (λ = 40).
        Does not work well, lead to traj shrinking and getting stuck on neighburing phases.

6. Other Ideas
    Investigate anisotropy vs total intensity (normalize XYZ to [-1, 1]).

    Check transient pauses paper for pause detection.

    Recent file memory.

    Save file data.

    Check to see if linearization remove pauses (not in a nice way as we will not know it happened?)

    Try to remove pauses then just count every change in 2pi as one rev and find speed?