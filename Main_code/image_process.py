def preprocess_state(state):
    if isinstance(state, tuple):
        state = state[0]
    else:
        state = state.cpu().numpy()

    state = state.flatten()

    return state
