# ai/Abel/utils/state_encoder.py
import numpy as np

def encode_state(hole_card, round_state):
    community_cards = round_state.get("community_card", [])
    pot = round_state.get("pot", {}).get("main", {}).get("amount", 0)
    player_stack = round_state["seats"][round_state["next_player"]]["stack"]
    opponent_stack = round_state["seats"][1 - round_state["next_player"]]["stack"]
    pot_odds = pot / max(1, player_stack)  # Prevent division by zero

    state = np.array([
        len(hole_card),  # Number of hole cards
        len(community_cards),  # Number of community cards
        pot / 1000,  # Normalize pot size
        player_stack / 1000,  # Normalize player stack
        opponent_stack / 1000,  # Normalize opponent stack
        pot_odds  # Pot odds
    ])

    # Logging for debugging
    with open("encoded_states_log.txt", "a") as f:
        f.write(f"Encoded State: {state.tolist()}, Shape: {state.shape}\n")

    return state
