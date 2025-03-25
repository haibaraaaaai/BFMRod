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

    Resolve .3f precision issue from line edit input.

34. Ref Cycle Logic
    Dynamic Threshold Trigger for ref cycle update:
        Trigger update only if phase error or distance exceeds threshold.
        Input timestamp → run detect_cycle_bounds() around it.

    Phase continuity after manual input:
        Use last assigned phase index before new cycle for continuity.
        In case of manual input:
            Recompute phase assignment but preserve phase continuity.
    
    Index input may not align well with cycle — prefer timestamps.

    Confirm initial ref cycle? Then use initial ref cycle average disstances to decide whether or not to automatically update.

    Slider for index adjustment with real time tracking of index on screen.

    GUI must enable this from PCA 3D viewer.
        Add manual ref cycle button (detect cycle from current window, plot immediately).
        Update all future ref cycles based on this new one.
        Maintain a list of added ref cycles for toggle/remove.

    Backward penalty for indices more than one or two units away?

4. Other Ideas
    Investigate anisotropy vs total intensity (normalize XYZ to [-1, 1]).

    Check transient pauses paper for pause detection.

    Save file data.

    Load new tdms in tdms viewer?

    Toggle speed calculation and rev_window input.
