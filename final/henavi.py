import asyncio
import websockets
import json
import random
import copy

# Initialization
moves = ['up', 'down', 'left', 'right']
DIRECTIONS = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}

# Helper functions
def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def extract_pos(pos):
    if isinstance(pos, dict):
        return pos['r'], pos['c']
    elif isinstance(pos, (list, tuple)) and len(pos) >= 2:
        return pos[0], pos[1]
    return None, None

def get_direction(move):
    DIRECTIONS = {
        'up': (-1, 0), 'down': (1, 0),
        'left': (0, -1), 'right': (0, 1)
    }
    return DIRECTIONS[move]

# Navigation algorithms
'''
Agents get game_state from web via websockets. 
We can use information in game_state to create agents.

Rerir: the name of my pacman.
Enemies: ghosts.

According to script.js (getCurrentGameState()), 
game_state contains:
    1. pacman_pos: {r: pacmanCurrentRow, c: pacmanCurrentCol, direction: currentDirection},
    2. is_werewolf_mode: isWerewolfMode,

    3. ghosts: ghostsData,
    const ghostsData = [];
    for (const name in CHARACTERS) {
        const char = CHARACTERS[name];
        ghostsData.push({
            name: name,
            r: char.r,
            c: char.c,
            direction: char.direction,
            state: char.state // States: "escape", "chase", "frightened"
        });
    }

    4. map_state: currentMazeState, 
    WALL = 0; // Rerir cannot move on this tile
    PATH = 1; // Rerir can move on this tile
    PELLET = 2; // Rerir can eat them to earn scores
    HEART_FRAGMENT = 3; // Rerir eats them and will werewolf in 8 seconds
    // During Rerir's werewolf state Rerir can eat enemies. Enemies are in "frighten" state and run away from Rerir.
    GHOST_CAGE = 4; // Enemies' staring points
    PACMAN_START = 5; // Rerir's starting point

    5. score: score, 
    6. lives: lives
    
'''

# MCTS 
def get_moves(game_state):
    """ 
    Get legal moves. 
    Legal moves: won't move outside boundary or hit a wall.
    """
    curr_row, curr_col = extract_pos(game_state['pacman_pos'])

    if curr_row is None:
        print(f"[WARNING] Unknown pacman_pos format: {game_state['pacman_pos']}")
        return ['up', 'down', 'left', 'right'] 

    map_state = game_state['map_state']
    legal_moves = []

    for move, (dr, dc) in DIRECTIONS.items():
        new_row = curr_row + dr
        new_col = curr_col + dc

        # Boundary and wall check
        if (0 <= new_row < len(map_state) and 
            0 <= new_col < len(map_state[0]) and 
            map_state[new_row][new_col] != 0):
            legal_moves.append(move)
    
    return legal_moves

def apply_legal_move(state, move):
    legal_moves = get_moves(state)

    if move not in legal_moves:
        return state
    
    new_state = copy.deepcopy(state)
    dr, dc = DIRECTIONS[move]

    curr_r, curr_c = extract_pos(new_state['pacman_pos'])
    
    if curr_r is None:
        return new_state 

    new_r = curr_r + dr
    new_c = curr_c + dc

    # Update position with the original format 
    if isinstance(new_state['pacman_pos'], dict):
        new_state['pacman_pos'] = {
            'r': new_r, 
            'c': new_c, 
            'direction': move 
        }
    else:
        new_state['pacman_pos'] = [new_r, new_c]
    
    return new_state

def avoid_ghosts_priority(game_state, legal_moves):
    """
    The highest priority is to avoid enemies, especially nearby enemies.
    Nearby enemies: If manhattan distance between Rerir and enemy <= 5,
    the enemy is nearby and Rerir should move away from it.
    """

    curr_r, curr_c = extract_pos(game_state['pacman_pos'])
    if curr_r is None: 
        return None
    
    # Check nearby enemies
    nearby_ghosts = []
    for ghost in game_state.get('ghosts', []):
        g_r, g_c = extract_pos(ghost)
        if g_r is None: 
            continue
        
        g_state = ghost.get('state', 'chase') if isinstance(ghost, dict) else 'chase'
        
        dist = manhattan_distance((curr_r, curr_c), (g_r, g_c))
        if dist <= 5:
            nearby_ghosts.append(((g_r, g_c), dist, g_state))
    
    if not nearby_ghosts:
        return None
    
    # Evaluate safety scores based on Rerir's current distance to enemies
    move_safety = {}
    for move in legal_moves:
        safety_score = 0
        
        dr, dc = get_direction(move)
        new_r = curr_r + dr
        new_c = curr_c + dc
        
        # If next move can help Rerir get away from enemies,
        # then it's a safe move and we will add scores to this move.
        # Otherwise it's not a safe move.
        for (ghost_r, ghost_c), current_distance, ghost_state in nearby_ghosts:
            new_distance = manhattan_distance((new_r, new_c), (ghost_r, ghost_c))
            
            if new_distance > current_distance:
                safety_score += 20 * (new_distance - current_distance)
            else:
                safety_score -= 30 * (current_distance - new_distance)
            
            # Increase weights when enemies are nearby
            if current_distance <= 5 and ghost_state != 'frightened':
                if new_distance > current_distance:
                    safety_score += 100
                else:
                    safety_score -= 200
        
        move_safety[move] = safety_score
    
    if move_safety:
        safest_move = max(move_safety, key = move_safety.get)
        if move_safety[safest_move] > 0:
            return safest_move
    
    return None

def is_oscillating_move(game_state, move):
    pacman_pos = game_state['pacman_pos']
    
    if isinstance(pacman_pos, dict) and 'direction' in pacman_pos:
        current_direction = pacman_pos['direction']
        
        if (current_direction == 'left' and move == 'right') or \
           (current_direction == 'right' and move == 'left'):
            return True
        
        if (current_direction == 'up' and move == 'down') or \
           (current_direction == 'down' and move == 'up'):
            return True
    
    return False

def is_exploration_move_smart(game_state, move):
    '''
    Smart exploration move: 
    1. Leave the bottom row of the maze. 
    2. Move in the middle part of the maze. There are some small paths, whcih are easily missed by Rerir.
    '''
    if move is None:
        return False
    
    curr_r, curr_c = extract_pos(game_state['pacman_pos'])
    dr, dc = get_direction(move)
    new_r = curr_r + dr
    
    if curr_r >= 27 and new_r < curr_r:
        return True
    
    if 10 < new_r < 20:
        return True

    return False

def select_smart_move_in_simulation(state, legal_moves):
    # 50% random move, 50% best_move
    if random.random() < 0.5:
        return random.choice(legal_moves)
    
    best_move = None
    max_score = -float('inf')
    curr_r, curr_c = extract_pos(state['pacman_pos'])
    
    for move in legal_moves:
        dr, dc = DIRECTIONS[move]
        nr, nc = curr_r + dr, curr_c + dc
        
        score = get_heuristic_score(state, nr, nc, curr_r, move)
        
        if score > max_score:
            max_score = score
            best_move = move
            
    return best_move if best_move else random.choice(legal_moves)


def mcts_selection_expansion(game_state):
    legal_moves = get_moves(game_state) 
    if not legal_moves:
        return None, []
        
    expanded_states = []
    for move in legal_moves:
        simulated_state = apply_legal_move(game_state, move) 
        expanded_states.append((move, simulated_state))
        
    return legal_moves, expanded_states

def get_heuristic_score(game_state, curr_r, curr_c, start_r, move=None):
    score = 0
    
    # Eat pellets/ heart-fragments
    tile_value = game_state['map_state'][curr_r][curr_c]
    if tile_value == 2: score += 60    # pellets
    elif tile_value == 3: score += 120 # heart-fragments
    
    # Enemy risk
    for ghost in game_state.get('ghosts', []):
        g_r, g_c = extract_pos(ghost)
        if g_r is None: continue
        
        dist = manhattan_distance((curr_r, curr_c), (g_r, g_c))
        g_state = ghost.get('state', 'chase') if isinstance(ghost, dict) else 'chase'
        
        if dist <= 2:
            score += 50 if g_state == 'frightened' else -150 
        elif dist <= 8:
            score -= 30 

    # Exploration bonus
    pacman_state = {'pacman_pos': [curr_r, curr_c], 'map_state': game_state['map_state']}
    if is_exploration_move_smart(pacman_state, move):
        score += 80

    # Moving-up bonus
    if curr_r < start_r:
        score += 40 + (start_r - curr_r) * 10

    # Oscillation Penalty
    if is_oscillating_move(game_state, move):
            score -= 500
        
    return score

def mcts_playout(game_state, depth = 8):
    curr_r, curr_c = extract_pos(game_state['pacman_pos'])
    if curr_r is None:
        return 0
        
    start_r = curr_r
    total_reward = 0
    map_ref = game_state['map_state'] 

    for step in range(depth):
        temp_state = {
            'pacman_pos': [curr_r, curr_c], 
            'map_state': map_ref
        }
        
        legal_moves = get_moves(temp_state)
        if not legal_moves:
            break
            
        move = select_smart_move_in_simulation(temp_state, legal_moves)
        
        dr, dc = DIRECTIONS[move]
        curr_r += dr
        curr_c += dc
        
        step_reward = get_heuristic_score(game_state, curr_r, curr_c, start_r, move = move)
        total_reward += step_reward * (0.85 ** step)
    
    return total_reward

def mcts_navi(game_state, iterations = 30):
    try:
        # Selection & Expansion
        legal_moves, expanded_nodes = mcts_selection_expansion(game_state)
        
        if not legal_moves or not expanded_nodes:
            return 'up'
        
        if len(legal_moves) == 1:
            return legal_moves[0]
        
        safe_move = avoid_ghosts_priority(game_state, legal_moves)
        if safe_move:
            return safe_move

        curr_r, curr_c = extract_pos(game_state['pacman_pos'])
        move_scores = {move: 0 for move in legal_moves}
        move_visits = {move: 0 for move in legal_moves}

        # Simulation 
        for _ in range(iterations):
            for move, simulated_state in expanded_nodes:
                reward = mcts_playout(simulated_state, depth = 8)
                
                # Backpropagation 
                move_scores[move] += reward
                move_visits[move] += 1

        # Final Decision 
        best_move = None
        max_final_score = -float('inf')

        for move in legal_moves:
            avg_score = move_scores[move] / move_visits[move] if move_visits[move] > 0 else 0
            
            # Exploration bonus
            exploration_bonus = 0
            if is_exploration_move_smart(game_state, move):
                exploration_bonus += 80
            
            if curr_r == 27 and move != 'down':
                exploration_bonus += 50

            final_score = avg_score + exploration_bonus
            
            if final_score > max_final_score:
                max_final_score = final_score
                best_move = move

        print(f"[MCTS] Decision: {best_move} (Score: {max_final_score:.2f})")
        return best_move if best_move else random.choice(legal_moves)

    except Exception as e:
        print(f"[MCTS Error] {e}")
        import traceback
        traceback.print_exc() 
        return random.choice(get_moves(game_state)) if get_moves(game_state) else 'up'



# A*
def a_star_navi(game_state):
    # print("[A* Algorithm] Running... (Currently returning random move)")
    print("[A* Algorithm] Unuseable ... (Currently returning random move)")
    return random.choice(moves)

# DQN
def dqn_navi(game_state):
    # print("[DQN Algorithm] Running... (Currently returning random move)")
    print("[DQN Algorithm] Unuseable ... (Currently returning random move)")
    return random.choice(moves)

# Select navigation algorithm
def ai_navi(game_state, algorithm):
    if algorithm == 'mcts':
        return mcts_navi(game_state)
    elif algorithm == 'a_star':
        return a_star_navi(game_state)
    elif algorithm == 'dqn':
        return dqn_navi(game_state)
    else:
        print(f"Unknown algorithm: {algorithm}. Using random move.")
        moves = ['up', 'down', 'left', 'right']
        return random.choice(moves)

# Web connection
async def web_connection(websocket):
    global current_algorithm
    valid_algorithms = ['a_star', 'mcts', 'dqn']
    current_algorithm = "mcts"
    
    try:
        print("New game client connected. Waiting for messages...")
        
        async for message in websocket:
            try:
                data = json.loads(message)
                print(f"Received message, algorithm: {current_algorithm}")
            except json.JSONDecodeError as e:
                print(f"[ERROR] JSON decode error: {e}")
                continue

            if isinstance(data, dict) and data.get('command') == 'set_algorithm':
                new_algorithm = data.get('name')
                if new_algorithm in valid_algorithms:
                    current_algorithm = new_algorithm
                    print(f"Algorithm set to: {current_algorithm.upper()}")
                continue

            if isinstance(data, dict) and 'pacman_pos' in data:
                print(f"Pacman position: {data['pacman_pos']}")
                print(f"Processing game state with {current_algorithm}")
                move = ai_navi(data, current_algorithm)
                print(f"Sending move: {move}")
                await websocket.send(move)
            else:
                print(f"Unknown message format")

    except websockets.ConnectionClosed:
        print("Game client connection closed.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
     

async def main():
    start_server = websockets.serve(
        web_connection, 
        "localhost", # Server address
        8765         # Server port
    )
    print("AI WebSocket server started, listening on ws://localhost:8765...")
    
    async with start_server:
        await asyncio.Future()

if __name__ == "__main__":
    try:
        # Use asyncio.run() to automatically create and manage the event loop
        asyncio.run(main()) 
    except KeyboardInterrupt:
        print("\nServer stopped.")