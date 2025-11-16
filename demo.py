import asyncio
import websockets
import json
import random

# ===============================
#     Navigation Algorithm
# ===============================

def a_star_decision(game_state):
    """
    Finds the shortest path for Rerir to the nearest pellet using the A* algorithm.
    
    Args:
        game_state (dict): Dictionary containing 'map_state' (map), 'pacman_pos' (Rerir's position), etc.
        
    Returns:
        str: Movement command, e.g., 'up', 'down', 'left', 'right'.
    """
    # 🌟 TODO: Implement your A* pathfinding algorithm here
    # Hint: A* requires the following information:
    # 1. Obstacles (0 in map_state)
    # 2. Start point (pacman_pos)
    # 3. Target point (the nearest '2' or '3')
    
    print("【A* Algorithm】Running... (Currently returning random move)")
    moves = ['up', 'down', 'left', 'right']
    return random.choice(moves)

def mcts_decision(game_state):
    """
    Makes a decision based on the MCTS (Monte Carlo Tree Search) algorithm, using simulations of future states.
    
    Args:
        game_state (dict): Dictionary containing the full game state.
        
    Returns:
        str: Movement command.
    """
    # 🌟 TODO: Implement your MCTS algorithm here
    # Hint: MCTS requires:
    # 1. A complete copy of the game state (for simulation)
    # 2. The number of runs/time limit for the simulation
    
    print("【MCTS Algorithm】Running... (Currently returning random move)")
    moves = ['up', 'down', 'left', 'right']
    return random.choice(moves)

def dqn_decision(game_state):
    """
    Makes a decision using the DQN (Deep Q-Network) algorithm with a trained model.
    
    Args:
        game_state (dict): Dictionary containing the full game state.
        
    Returns:
        str: Movement command.
    """
    # 🌟 TODO: Load your DQN model here and convert the game state to model input.
    # Hint: DQN requires:
    # 1. A trained model file
    # 2. State preprocessing/feature engineering
    
    print("【DQN Algorithm】Running... (Currently returning random move)")
    moves = ['up', 'down', 'left', 'right']
    return random.choice(moves)


def pacman_ai_decision(game_state, algorithm):
    """
    Dispatches the decision-making process to the corresponding function based on the selected algorithm.
    """
    if algorithm == 'a_star':
        return a_star_decision(game_state)
    elif algorithm == 'mcts':
        return mcts_decision(game_state)
    elif algorithm == 'dqn':
        return dqn_decision(game_state)
    else:
        # Default or unknown algorithm, return random move
        print(f"Unknown algorithm: {algorithm}. Using random move.")
        moves = ['up', 'down', 'left', 'right']
        return random.choice(moves)

# ======================================================
# WebSocket CONNECTION HANDLER
# ======================================================

async def handle_pacman_connection(websocket, path=None):
    print("New game client connected. Waiting for game state/algorithm selection...")
    
    # Stores the algorithm selected for the current connection
    current_algorithm = "a_star"  # Default to A*
    print(f"Default AI algorithm set to: {current_algorithm.upper()}")

    try:
        # Continuously receive messages from the game client
        async for message in websocket:
            try:
                command_data = json.loads(message)
            except json.JSONDecodeError:
                print(f"Received invalid JSON: {message}")
                continue

            # 1. 【Handle Algorithm Selection Command】(From the Apply button)
            if "command" in command_data and command_data["command"] == "set_algorithm":
                current_algorithm = command_data.get("name", current_algorithm)
                print(f"AI algorithm set to: {current_algorithm.upper()}")
                
                # Optional: If the algorithm is set, skip decision-making for this message and wait for the next game state
                continue 

            # 2. 【Handle Game State】(From the game loop)
            # Assume the remaining message is the game state
            game_state = command_data 
            # print(f"Received game state, Rerir position: {game_state.get('pacman_pos')}")

            # 3. Run the AI algorithm and get the move command
            move = pacman_ai_decision(game_state, current_algorithm)
            
            # 4. Send the command back to the client
            await websocket.send(move)
            # print(f"Sending move: {move}")

    except websockets.ConnectionClosedOK:
        print("Game client connection closed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")


async def main():
    # Start the server's asynchronous task
    start_server = websockets.serve(
        handle_pacman_connection, 
        "localhost", # Server address
        8765         # Server port
    )
    print("AI WebSocket server started, listening on ws://localhost:8765...")
    
    # Keep the main coroutine running
    await start_server
    await asyncio.Future()

if __name__ == "__main__":
    try:
        # Use asyncio.run() to automatically create and manage the event loop (Recommended for Python 3.7+)
        asyncio.run(main()) 
    except KeyboardInterrupt:
        print("\nServer stopped.")