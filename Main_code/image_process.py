def preprocess_state(state):
    """
    Check the shape of the state and flatten or reshape it as needed.

    Parameters:
    state (np.array or tensor): The state returned by the Atari game environment.

    Returns:
    np.array: The preprocessed state.
    int: The size of the flattened state.
    """
    if isinstance(state, tuple):
        state = state[0]

    # Check if the state is a tensor and convert to numpy array if needed
    if hasattr(state, 'numpy'):
        state = state.numpy()

    # Check the shape of the state
    state_shape = state.shape
    print(f"Original state shape: {state_shape}")

    # Flatten the state
    state = state.flatten()
    state_size = state.shape[0]
    print(f"Flattened state shape: {state.shape}")

    return state, state_size