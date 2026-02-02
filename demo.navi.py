import asyncio
import websockets
import json
import math
import random
import copy
import time

# Global state to track which algorithm the UI wants to use
current_algorithm = 'mcts'

# --- WebSocket Server Logic (From navi.py) ---

async def web_connection(websocket, path):
    global current_algorithm
    print("New Client Connected")
    try:
        async for message in websocket:
            # 1. Parsing Message
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                print("Invalid JSON received")
                continue
            
            # 2. Check if it's a control command (e.g., switching algorithms)
            if isinstance(data, dict) and 'command' in data:
                if data['command'] == 'set_algorithm':
                    current_algorithm = data.get('name', 'mcts')
                    print(f"Algorithm switched to: {current_algorithm}")
                continue 

            # 3. Handle Game State
            # If it's not a command, we assume it's the game state object
            game_state = data 
            
            # 4. Run the AI algorithm
            move = ai_navi(game_state, current_algorithm)
            
            # 5. Send the move back to the client
            await websocket.send(move)
            
    except websockets.exceptions.ConnectionClosedOK:
        print("Client Disconnected")
    except Exception as e:
        print(f"Connection Error: {e}")

async def main():
    # Start the server
    async with websockets.serve(web_connection, "localhost", 8765):
        print("AI WebSocket server started on ws://localhost:8765...")
        print("Waiting for game connection...")
        await asyncio.Future()  # Run forever

# --- AI Dispatcher ---

def ai_navi(game_state, algorithm):
    """
    Dispatches the decision-making process based on the selected algorithm.
    """
    if algorithm == 'a_star':
        return a_star_navi(game_state)
    elif algorithm == 'mcts':
        return mcts_navi(game_state)
    elif algorithm == 'dqn':
        return dqn_navi(game_state)
    else:
        # Fallback for unknown algorithms
        # print(f"Unknown algorithm: {algorithm}")
        return random.choice(['up', 'down', 'left', 'right'])

# --- Algorithm Wrappers ---

def mcts_navi(game_state):
    """
    Wrapper for MCTS Agent.
    """
    try:
        # 1. Adapt Data: Convert HTML dicts to Python tuples
        internal_state = parse_game_state(game_state)
        
        # 2. Run Agent
        # Iterations: 200 is a good balance for real-time WebSocket performance
        agent = MCTS_Agent(iterations=200, max_simulation_depth=15)
        move = agent.search(internal_state)
        
        return move if move else 'up'
    except Exception as e:
        print(f"MCTS Error: {e}")
        return 'up'

def a_star_navi(game_state):
    # Placeholder for A*
    print("A* logic triggered (not implemented)")
    return random.choice(['up', 'down', 'left', 'right'])

def dqn_navi(game_state):
    # Placeholder for DQN
    print("DQN logic triggered (not implemented)")
    return random.choice(['up', 'down', 'left', 'right'])

# --- Data Adapters ---

def parse_game_state(external_state):
    """
    Normalizes input from HTML/Game engine into clean internal format.
    """
    # 1. Map
    grid = external_state.get('map_state', external_state.get('map', []))
    
    # 2. Pacman Position
    p_pos = external_state.get('pacman_pos')
    if isinstance(p_pos, dict):
        pacman_pos = (p_pos.get('r', 0), p_pos.get('c', 0))
    elif isinstance(p_pos, (list, tuple)) and len(p_pos) >= 2:
        pacman_pos = tuple(p_pos[:2])
    else:
        pacman_pos = (1, 1)

    # 3. Ghosts
    ghosts = []
    raw_ghosts = external_state.get('ghosts', [])
    for g in raw_ghosts:
        g_internal = {}
        if isinstance(g, dict):
             g_internal['pos'] = (g.get('r', 0), g.get('c', 0))
             g_internal['state'] = g.get('state', 'chase')
        elif isinstance(g, (list, tuple)) and len(g) >= 2:
             g_internal['pos'] = tuple(g[:2])
             g_internal['state'] = 'chase'
        else:
            continue
        ghosts.append(g_internal)

    return {
        'map': grid,
        'pacman_pos': pacman_pos,
        'ghosts': ghosts
    }

# --- MCTS Core Logic ---

class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.score = 0
        self.untried_moves = get_moves(state)

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def is_terminal_node(self):
        return is_game_terminal(self.state)

    def best_child(self, c_param=1.414):
        choices_weights = [
            (child.score / child.visits) + c_param * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

class MCTS_Agent:
    def __init__(self, iterations=100, max_simulation_depth=10):
        self.iterations = iterations
        self.max_depth = max_simulation_depth

    def search(self, initial_state):
        root = MCTSNode(state=initial_state)

        # 0. Safety Check (Heuristic Pruning)
        legal_moves = get_moves(initial_state)
        safe_moves = self.filter_unsafe_moves(initial_state, legal_moves)
        
        if not safe_moves:
             return legal_moves[0] if legal_moves else 'up'
             
        if len(safe_moves) == 1:
            return safe_moves[0]
        
        # 1. Build Tree
        start_time = time.time()
        for _ in range(self.iterations):
            if time.time() - start_time > 0.4: # Strict time limit for WebSocket latency
                break
                
            node = root
            temp_state = copy.deepcopy(initial_state)

            # Selection
            while node.is_fully_expanded() and not node.is_terminal_node():
                node = node.best_child()
                temp_state = apply_move(temp_state, node.move)

            # Expansion
            if not node.is_terminal_node() and not node.is_fully_expanded():
                move = node.untried_moves.pop()
                temp_state = apply_move(temp_state, move)
                new_node = MCTSNode(temp_state, parent=node, move=move)
                node.children.append(new_node)
                node = new_node

            # Simulation
            reward = self.simulate(temp_state)

            # Backpropagation
            while node is not None:
                node.visits += 1
                node.score += reward
                node = node.parent

        # 2. Select Move
        best_child = None
        most_visits = -1
        
        for child in root.children:
            if child.move not in safe_moves: continue
            if child.visits > most_visits:
                most_visits = child.visits
                best_child = child
        
        return best_child.move if best_child else random.choice(safe_moves)

    def simulate(self, state):
        current_state = copy.deepcopy(state)
        total_reward = 0
        depth = 0
        last_move = None
        reverse_map = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}
        
        while not is_game_terminal(current_state) and depth < self.max_depth:
            legal = get_moves(current_state)
            if not legal: break
            
            # Prevent immediate reversing for realistic pathing
            if last_move and len(legal) > 1:
                filtered = [m for m in legal if m != reverse_map.get(last_move)]
                if filtered: legal = filtered
            
            move = random.choice(legal)
            last_move = move
            current_state = apply_move(current_state, move)
            
            r, c = current_state['pacman_pos']
            try: tile = state['map'][r][c] 
            except: tile = 0
            
            min_ghost_dist = float('inf')
            near_scared = False
            for g in current_state['ghosts']:
                d = manhattan_distance(current_state['pacman_pos'], g['pos'])
                min_ghost_dist = min(min_ghost_dist, d)
                if d == 0 and g['state'] in ['scared', 'frightened']: total_reward += 200
                elif d <= 1 and g['state'] in ['scared', 'frightened']: near_scared = True

            if min_ghost_dist <= 1 and not near_scared:
                total_reward -= 100
                break
            
            if tile == 2: total_reward += 10
            if tile == 3: total_reward += 50
            total_reward += 1
            depth += 1
            
        return total_reward

    def filter_unsafe_moves(self, state, moves):
        safe = []
        for m in moves:
            sim_state = apply_move(state, m)
            pac_pos = sim_state['pacman_pos']
            is_dangerous = False
            for ghost in sim_state['ghosts']:
                dist = manhattan_distance(pac_pos, ghost['pos'])
                if dist <= 1 and ghost['state'] not in ['scared', 'frightened']:
                    is_dangerous = True
                    break
            if not is_dangerous: safe.append(m)
        return safe

# --- Helper Functions ---

def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def get_moves(state):
    directions = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
    if not state['map']: return []
    rows, cols = len(state['map']), len(state['map'][0])
    r, c = state['pacman_pos']
    legal = []
    for m, (dr, dc) in directions.items():
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            tile = state['map'][nr][nc]
            # 0=Wall, 4=Ghost Cage (Treat as wall)
            if tile != 0 and tile != 4: 
                legal.append(m)
    return legal

def apply_move(state, move):
    new_state = copy.deepcopy(state)
    dr, dc = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}[move]
    pr, pc = new_state['pacman_pos']
    new_state['pacman_pos'] = (pr + dr, pc + dc)
    return new_state

def is_game_terminal(state):
    pac_pos = state['pacman_pos']
    for ghost in state['ghosts']:
        if manhattan_distance(pac_pos, ghost['pos']) == 0:
            if ghost['state'] not in ['scared', 'frightened']:
                return True 
    return False

# --- Entry Point ---

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped.")