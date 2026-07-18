import numpy as np

def compute_monotonic_alignment(probs, epsilon=1e-10):
    """
    Finds indices 0 <= T1 < T2 < ... < T8 < N that maximize the sum of log probabilities
    for milestone classes 1 to 8.
    
    probs: numpy array of shape (N, 9), where probs[t, c] is the probability of class c at frame t.
           Class 0 is background, classes 1-8 are milestones.
    returns: list of 8 frame indices [T1, T2, ..., T8]
    """
    N = len(probs)
    if N < 8:
        raise ValueError(f"Video length {N} is too short for 8 distinct milestones.")
        
    # dp[c][t] stores the max log prob sum for milestone c (1-indexed, so 1..8) at frame t
    # Shape: (9, N) initialized to -inf
    dp = np.full((9, N), -np.inf)
    backtrack = np.zeros((9, N), dtype=int)
    
    # Initialize c = 1
    # For milestone 1, we can place it at any frame from 0 to N - 8
    for t in range(N - 7):
        dp[1, t] = np.log(probs[t, 1] + epsilon)
        
    # Fill DP table for c = 2 to 8
    for c in range(2, 9):
        # We need at least c-1 frames before milestone c, and at least 8-c frames after it
        min_t = c - 1
        max_t = N - (8 - c) - 1
        
        # We can optimize the max search over t' < t by keeping track of the running maximum
        best_val = dp[c-1, min_t - 1]
        best_t_prime = min_t - 1
        
        for t in range(min_t, max_t + 1):
            # As t increases, we consider one new candidate for t' (which is t - 1)
            candidate_val = dp[c-1, t - 1]
            if candidate_val > best_val:
                best_val = candidate_val
                best_t_prime = t - 1
                
            dp[c, t] = np.log(probs[t, c] + epsilon) + best_val
            backtrack[c, t] = best_t_prime
            
    # Backtrack from the end to find the optimal sequence
    # The optimal final milestone T8 is the index that maximizes dp[8, t] for eligible t
    min_t8 = 7
    max_t8 = N - 1
    best_t8 = min_t8 + np.argmax(dp[8, min_t8 : max_t8 + 1])
    
    milestones = [0] * 8
    curr_t = best_t8
    milestones[7] = int(curr_t)
    
    for c in range(8, 1, -1):
        curr_t = backtrack[c, curr_t]
        milestones[c-2] = int(curr_t)
        
    return milestones
