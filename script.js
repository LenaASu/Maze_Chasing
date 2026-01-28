// ======================================================
//                 PAC-MAN GAME ENGINE
//              Version (2025), Xiang Luo
// ======================================================

// ======================================================
//             AI SERVER INTEGRATION VARIABLES
// ======================================================

const AI_SERVER_URL = "ws://localhost:8765";
let aiWebSocket = null;
let isAIControlled = false; 

// ======================================================
//                GLOBAL GAME VARIABLES
// ======================================================

// ----- Maze Data ----- 
const WALL = 0; // Rerir cannot move on this tile
const PATH = 1; // Rerir can move on this tile
const PELLET = 2; // Rerir can eat them to earn scores
const HEART_FRAGMENT = 3; // Rerir eats them and will werewolf in 8 seconds
// During Rerir's werewolf state Rerir can eat enemies. Enemies are in "frighten" state and run away from Rerir.
const GHOST_CAGE = 4; // Enemies' staring points
const PACMAN_START = 5; // Rerir's starting point

const PACMAN_MAZE = [
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,2,2,2,2,2,2,2,2,2,2,2,2,0,0,2,2,2,2,2,2,2,2,2,2,2,2,0],
    [0,2,0,0,0,0,2,0,0,0,0,0,2,0,0,2,0,0,0,0,0,2,0,0,0,0,2,0],
    [0,3,0,0,0,0,2,0,0,0,0,0,2,0,0,2,0,0,0,0,0,2,0,0,0,0,3,0],
    [0,2,0,0,0,0,2,0,0,0,0,0,2,0,0,2,0,0,0,0,0,2,0,0,0,0,2,0],
    [0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,0],
    [0,2,0,0,0,0,2,0,0,2,0,0,0,0,0,0,0,0,2,0,0,2,0,0,0,0,2,0],
    [0,2,0,0,0,0,2,0,0,2,0,0,0,0,0,0,0,0,2,0,0,2,0,0,0,0,2,0],
    [0,2,2,2,2,2,2,0,0,2,2,2,2,0,0,2,2,2,2,0,0,2,2,2,2,2,2,0],
    [0,0,0,0,0,0,2,0,0,0,0,0,1,0,0,1,0,0,0,0,0,2,0,0,0,0,0,0],
    [0,0,0,0,0,0,2,0,0,0,0,0,1,0,0,1,0,0,0,0,0,2,0,0,0,0,0,0],
    [0,0,0,0,0,0,2,0,0,1,1,1,1,1,1,1,1,1,1,0,0,2,0,0,0,0,0,0],
    [0,0,0,0,0,0,2,0,0,1,0,0,0,4,4,0,0,0,1,0,0,2,0,0,0,0,0,0],
    [0,0,0,0,0,0,2,0,0,1,0,4,4,4,4,4,4,0,1,0,0,2,0,0,0,0,0,0],
    [1,1,1,1,1,1,2,1,1,1,0,4,4,4,4,4,4,0,1,1,1,2,1,1,1,1,1,1],
    [0,0,0,0,0,0,2,0,0,1,0,0,0,0,0,0,0,0,1,0,0,2,0,0,0,0,0,0],
    [0,0,0,0,0,0,2,0,0,1,0,0,0,0,0,0,0,0,1,0,0,2,0,0,0,0,0,0],
    [0,0,0,0,0,0,2,0,0,1,1,1,1,1,1,1,1,1,1,0,0,2,0,0,0,0,0,0],
    [0,2,2,2,2,2,2,2,2,2,2,2,2,0,0,2,2,2,2,2,2,2,2,2,2,2,2,0],
    [0,2,0,0,0,0,2,0,0,0,0,0,2,0,0,2,0,0,0,0,0,2,0,0,0,0,2,0],
    [0,2,0,0,0,0,2,0,0,0,0,0,2,0,0,2,0,0,0,0,0,2,0,0,0,0,2,0],
    [0,3,2,2,0,0,2,2,2,2,2,2,2,1,1,2,2,2,2,2,2,2,0,0,2,2,3,0],
    [0,0,0,2,0,0,2,0,0,2,0,0,0,0,0,0,0,0,2,0,0,2,0,0,2,0,0,0],
    [0,0,0,2,0,0,2,0,0,2,0,0,0,0,0,0,0,0,2,0,0,2,0,0,2,0,0,0],
    [0,2,2,2,2,2,2,0,0,2,2,2,2,0,0,2,2,2,2,0,0,2,2,2,2,2,2,0],
    [0,2,0,0,0,0,0,0,0,0,0,0,2,0,0,2,0,0,0,0,0,0,0,0,0,0,2,0],
    [0,2,0,0,0,0,0,0,0,0,0,0,2,2,2,2,0,0,0,0,0,0,0,0,0,0,2,0],
    [0,2,2,2,2,2,2,2,2,2,2,2,2,5,1,2,2,2,2,2,2,2,2,2,2,2,2,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
];

let currentMazeState = [];

// ----- BGM -----
let audioBGM = null;

// ----- Characters (Rerir) -----
const GAME_SPEED = 250;
const GAME_SPEED_SECONDS = GAME_SPEED / 1000;
const WEREWOLF_DURATION = 8000; //8 seconds (8000 ms)

let isWerewolfMode = false;
let werewolfTimer = null;

let pacmanCurrentRow = 27;
let pacmanCurrentCol = 13;

// Store initial position for resetting after death
const PACMAN_START_R = 27; 
const PACMAN_START_C = 13;

// ----- Characters (Enemies) -----
const CHARACTERS = {
    // Use speedMultiplier to change the speed of enemies 
    traveler: { r:14, c:13, direction:'up',  class:'traveler-graphic', element:null, speedMultiplier:0.7, tick:0, role:"blinky", state: "escape" },
    albedo:   { r:14, c:14, direction:'up',  class:'albedo-graphic',   element:null, speedMultiplier:0.7, tick:0, role:"pinky",  state: "escape" },
    aino:     { r:14, c:12, direction:'up',  class:'aino-graphic',     element:null, speedMultiplier:0.7, tick:0, role:"inky",   state: "escape" },
    flins:    { r:14, c:15, direction:'up',  class:'flins-graphic',    element:null, speedMultiplier:0.7, tick:0, role:"clyde",  state: "escape" }
};

// ----- Initialize Basic Settings -----
let score = 0;
let lives = 3;

let cellSize = 0;
let totalCollectables = 0;
let remainingCollectables = 0;

let isPaused = false;
let currentDirection = 'right';
let nextDirection = 'right';
let pacmanGraphicElement = null;

let ghostsActivated = true;
let isAutoStart = false; 

let hasGhostActivationScheduled = false;
let gameLoopInterval = null;


// ======================================================
//                HELPER FUNCTIONS
// ======================================================

// Return DOM cell
function getGridCell(r, c) {
    const grid = document.getElementById("maze-grid");
    const idx = r * PACMAN_MAZE[0].length + c;
    return grid.children[idx] || null;
}

function updateScoreboard() {
    document.getElementById("current-score").textContent = score;
    document.getElementById("lives").textContent = lives;
}

// ======================================================
//             WEBSOCKET & AI AGENT COMMUNICATION
// ======================================================

// 1. Create WebSocket & AI Agent connection
function connectToAI() {
    aiWebSocket = new WebSocket(AI_SERVER_URL);

    aiWebSocket.onopen = () => {
        console.log("Successfully connect to Python AI Agent server!");
    };

    aiWebSocket.onmessage = (event) => {
        const command = event.data;
        // console.log("Received AI command:", command); // Test if agent can receive commands 
        applyAICommand(command);
    };

    aiWebSocket.onerror = (error) => {
        console.error("WebSocket error:", error);
    };

    aiWebSocket.onclose = () => {
        console.log("Connection with AI Agent closed.");
        // Optional: add re-connect logic
    };
}

// 2. Web collect all the data AI needs
function getCurrentGameState() {
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
    
    return {
        // Information for AI agents to make decisions 
        pacman_pos: {r: pacmanCurrentRow, c: pacmanCurrentCol, direction: currentDirection},
        ghosts: ghostsData,
        map_state: currentMazeState, 
        score: score,
        lives: lives,
        is_werewolf_mode: isWerewolfMode
    };
}

// 3. Web sends information to AI agents
function sendGameState() {
    if (aiWebSocket && aiWebSocket.readyState === WebSocket.OPEN) {
        const gameState = getCurrentGameState();
        aiWebSocket.send(JSON.stringify(gameState));
    }
}

// 4. Ai agents analyse information
function applyAICommand(command) {
    // If the next command is valid, run this command in next frame
    const validDirs = ['up', 'down', 'left', 'right', 'stop'];
    if (validDirs.includes(command)) {
        nextDirection = command;

        // If pause the game (press "space"), resume when receiving the first valid command (press "space" again)
        if (!gameLoopInterval) {
            startGameLoop();
            ghostsActivated = true;
        }
    }
}



/**
 * @param {boolean} enable - If true, then controlled by AI
 * @param {string} algorithm - The selected algorithm ('mcts', 'a_star', 'dqn')
 */
function toggleAIControl(enable, algorithm = null) {
    if (enable) {
        if (!isAIControlled) {
            isAIControlled = true;
            console.log(`AI Control ENABLED: ${algorithm} selected.`); // Select algorithm
            connectToAI(); // Sent the name of selected algorithm to AI Agent
        } else {
            // When running a AI Agent, apply another algorithm will switch to another agent. 
            console.log(`AI already ON. Switched algorithm to: ${algorithm}`);
        }
    } else {
        isAIControlled = false;
        console.log("AI Control DISABLED (Back to Keyboard).");
        if (aiWebSocket) {
             // Keep connecting but do not send information
             // Optional: Close the connection 
        }
    }
}

// Initialize/Re-Draw the Maze and Collectables
function initializeMaze() {
    const grid = document.getElementById("maze-grid");
    
    // 1. Clear the entire grid content before redrawing
    grid.innerHTML = ''; 

    // 2. Reset the current game state to a fresh copy of the original maze
    //    Use .map() to create a true deep copy of the 2D array.
    currentMazeState = PACMAN_MAZE.map(row => [...row]); 

    // 3. Reset collectable counts
    totalCollectables = 0;
    
    // 4. Loop through the currentMazeState to draw cells and count items
    currentMazeState.forEach((row,r) => row.forEach((val,c) => {
        if (val === PELLET || val === HEART_FRAGMENT) {
            totalCollectables++;
        }
        
        const cell = document.createElement("div");
        const type = val === PACMAN_START ? PATH : val;

        const mapping = {
            0:"wall",
            1:"path",
            2:"pellet",
            3:"heart-fragment",
            4:"ghost-cage"
        };

        cell.classList.add("cell",mapping[type]);

        // Gate tile
        if (r === 12 && (c === 13 || c === 14)) {
            cell.classList.add("ghost-gate");
        }

        grid.appendChild(cell);
    }));

    remainingCollectables = totalCollectables;
    
    // 5. Re-append Pac-Man and Ghost Graphics to the new grid
    // grid.appendChild(pacmanGraphicElement);
    if (pacmanGraphicElement) grid.appendChild(pacmanGraphicElement);
    for (const name in CHARACTERS) {
        grid.appendChild(CHARACTERS[name].element);
    }
}

// ======================================================
//             DEATH HANDLER 
// ======================================================

let isCurrentlyDying = false; // Prevents re-entry into the death sequence

function handleRerirDeath() {
    if (isCurrentlyDying || lives <= 0) {
        return; // Already handling death or already dead
    }
    
    isCurrentlyDying = true; // Set flag to block subsequent calls
    
    // 1. Immediately stop the game loop
    stopGameLoop();

    // 2. Decrement life and update scoreboard
    lives--;
    updateScoreboard(); 

    // 3. Determine next action
    if (lives <= 0) {
        // Handle Game Over
        setTimeout(gameOver, 500); 
    } else {
        // Reset characters and restart level/life
        setTimeout(resetGame, 500); 
    }
}

// ======================================================
//                GAME STATE RESET/COLLISION
// ======================================================

function gameOver() {
    stopGameLoop(); 
    
    const message = "⛓️‍💥 YOU ARE ARRESTED! (Rerir has run out of lives.)";
    
    if (aiWebSocket && aiWebSocket.readyState === WebSocket.OPEN) {
        aiWebSocket.send(JSON.stringify({
            command: "game_summary",
            status: "lost",
            final_score: score || 0
        }));
    }

    setTimeout(() => {
        alert(message);
        fullGameReset();
    }, 100);


    // const status = lives <= 0 ? "lost" : "won";

    // if (aiWebSocket && aiWebSocket.readyState === WebSocket.OPEN) {
    //         aiWebSocket.send(JSON.stringify({
    //             command: "game_summary",
    //             status: lives <= 0 ? "lost" : "won",
    //             final_score: score || 0,
    //             duration: typeof gameDuration !== 'undefined' ? gameDuration : null,
    //             iterations: typeof gameIterations !== 'undefined' ? gameIterations : null,
    //         }));

    //         setTimeout(() => {
    //             alert("⛓️‍💥 YOU ARE ARRESTED! (Rerir has run out of lives.)");
    //             // location.reload(); // Reload the page to restart the game
    //             fullGameReset();
    //         }, 100);
    //     } else {
    //         alert("⛓️‍💥 YOU ARE ARRESTED! (Rerir has run out of lives.)");
    //         // location.reload();
    //         fullGameReset();
    //     }
        
    // stopGameLoop();
    // alert("⛓️‍💥 YOU ARE ARRESTED! (Rerir has run out of lives.)");
    // fullGameReset();

}

function checkWinCondition() {
    if (remainingCollectables <= 0) {
        stopGameLoop();
        alert("🎉🐺 Rerir Werewolf!");
        // fullGameReset();
    }
}

// Send Game Over Stats to AI Agent
function sendGameOverStats() {
    if (aiWebSocket && aiWebSocket.readyState === WebSockets.OPEN) {
        aiWebSocket.send(JSON.stringify({
            command: "game_summary",
            status: "lost",
            // status: lives <= 0 ? "lost" : "won"
            final_score: score,
            lives: lives,
            duration: gameDuration,
            
        }));
    }
}

// Full Game Reset
function fullGameReset() {
    // 1. Reset all persistent state variables
    score = 0;
    lives = 3;
    isPaused = false;
    isCurrentlyDying = false;
    ghostsActivated = false;
    
    // Cancel Werewolf mode if active
    if (werewolfTimer) {
        clearTimeout(werewolfTimer);
        werewolfTimer = null;
        isWerewolfMode = false;
    }

    // 2. Redraw the maze and reset collectables
    initializeMaze(); 

    // 3. Use resetGame() to place characters and start the game loop
    // ResetGame already calls startGameLoop() and updates visuals.
    resetGame();
    
    // Since this is a full reset, we need to ensure the ghost delay is active
    delayGhostActivation(); 
}

function resetGame() {
    // Reset Pac-Man position
    pacmanCurrentRow = PACMAN_START_R;
    pacmanCurrentCol = PACMAN_START_C;
    currentDirection = null;
    nextDirection = null;

    // if (isAIControlled){
    //     currentDirection = 'right';
    //     nextDirection = 'right';
    // } else {
    //     currentDirection = null;
    //     nextDirection = null;
    // }
    
    // Reset Ghost positions and state
    for (const name in CHARACTERS) {
        const char = CHARACTERS[name];
        // Reset to initial positions defined in CHARACTERS
        const initial = {
            traveler: { r:14, c:13, direction:'up', state:"escape" },
            albedo:   { r:14, c:14, direction:'up', state:"escape" },
            aino:     { r:14, c:12, direction:'up', state:"escape" },
            flins:    { r:14, c:15, direction:'up', state:"escape" }
        }[name];

        char.r = initial.r;
        char.c = initial.c;
        char.direction = initial.direction;
        char.state = initial.state;
    }

    //
    stopGameLoop();

    // Deactivate ghosts until Pac-Man moves again
    ghostsActivated = false;
    isPaused = false; 
    isCurrentlyDying = false; // Reset the death flag

    //
    currentDirection = null;
    nextDirection = null;

    updatePacmanVisualPosition();
    updateCharacterVisualPositions();
    updateScoreboard();

    // Automatically restart the game loop to resume movement
    // startGameLoop(); 

    // Delay Ghost Activation by 1 second 
    // delayGhostActivation();
    
    // // 1000ms = 1 second delay
    // setTimeout(() => {
    //     // Only activate if the game hasn't been paused or stopped/over
    //     if (gameLoopInterval) {
    //         ghostsActivated = true;
    //     }
    // }, 1000); 
}

// ======================================================
//                COLLISION LOGIC
// ======================================================

function checkCollision() {
    let fatalCollisionDetected = false;

    for (const name in CHARACTERS) {
        const char = CHARACTERS[name];
        
        // Collision detected
        if (char.r === pacmanCurrentRow && char.c === pacmanCurrentCol) {
            
            if (isWerewolfMode && char.state === "frightened") {
                // Rerir EATS the enemy (Ghost-eat logic remains the same)
                
                score += 200; // Bonous scores for eating one enemy
                updateScoreboard();

                // Teleport the eaten ghost back to the starting cage
                const initial = {
                    traveler: { r:14, c:13, direction:'up', state:"escape" },
                    albedo:   { r:14, c:14, direction:'up', state:"escape" },
                    aino:     { r:14, c:12, direction:'up', state:"escape" },
                    flins:    { r:14, c:15, direction:'up', state:"escape" }
                }[name];

                char.r = initial.r;
                char.c = initial.c;
                char.direction = initial.direction;
                char.state = initial.state;
                
                warpCharacter(char.element, char.r, char.c);
                
                continue; // Continue checking other ghosts to see if Rerir eats them too

            } 
            // else {
            //     // Rerir is EATEN by the enemy (Normal collision)
                
            //     lives--;
            //     updateScoreboard();

            //     if (lives <= 0) {
            //         gameStarted = false;
            //         ghostsActivated = false;
            //         gameOver();
            //         return true;
            // } 
            else {
                handleRerirDeath();
                return true
                // fatalCollisionDetected = true;
            }
        }
    }

    // FINAL CHECK: If ANY fatal collision was detected, trigger the death handler once.
    // if (fatalCollisionDetected) {
    //     handleRerirDeath();
    //     return true;
    // }
    
    return false; // No fatal collision detected
}

// ======================================================
//       CREATE GHOST GRAPHICS + UPDATE POSITIONS
// ======================================================

function createCharacterGraphics() {
    const grid = document.getElementById("maze-grid");
    for (const name in CHARACTERS) {
        const char = CHARACTERS[name];
        const elem = document.createElement("div");
        elem.classList.add("ghost-graphic", char.class);
        grid.appendChild(elem);
        char.element = elem;
    }
    updateCharacterVisualPositions();
}

function updateCharacterVisualPositions() {
    if (cellSize === 0) return;
    for (const name in CHARACTERS) {
        const char = CHARACTERS[name];
        const elem = char.element;
        elem.style.setProperty('--game-speed', `${GAME_SPEED_SECONDS}s`);
        // Use a small transition time if the ghost just left the cage to ensure smooth movement out of the gate
        // This is a stylistic choice and can be adjusted
        const transitionSpeed = (char.r <= 12 && char.state === "escape") ? `${GAME_SPEED_SECONDS/2}s` : `${GAME_SPEED_SECONDS}s`;
        elem.style.transition = `transform ${transitionSpeed} linear`;
        elem.style.transform = `translate(${char.c*cellSize}px, ${char.r*cellSize}px)`;
    }
}

// ======================================================
//                PAC-MAN VISUAL UPDATE
// ======================================================

function updatePacmanVisualPosition() {
    if (!pacmanGraphicElement || cellSize === 0) return;

    pacmanGraphicElement.style.setProperty('--game-speed', `${GAME_SPEED_SECONDS}s`);
    pacmanGraphicElement.style.transform =
        `translate(${pacmanCurrentCol*cellSize}px, ${pacmanCurrentRow*cellSize}px)`;

    pacmanGraphicElement.classList.remove("pacman-up","pacman-down","pacman-left","pacman-right");
    pacmanGraphicElement.classList.add(`pacman-${currentDirection}`);
}

// ======================================================
//                MOVEMENT VALIDATION
// ======================================================
function checkNextMove(direction, r=pacmanCurrentRow, c=pacmanCurrentCol, entity="player") {
    let nr = r, nc = c;
    const maxCols = PACMAN_MAZE[0].length;

    switch(direction) {
        case "up": nr--; break;
        case "down": nr++; break;
        case "left": nc--; break;
        case "right": nc++; break;
    }

    // --- WARP TUNNEL LOGIC: Calculates the post-warp coordinates ---
    let didWarp = false;
    if (nr === 14) {
        if (nc < 0) {
            nc = maxCols - 1;
            didWarp = true;
        }
        if (nc >= maxCols) {
            nc = 0;
            didWarp = true;
        }
    }
    
    // If the move resulted in a warp, it is always valid and should bypass all other checks
    if (didWarp) {
        return { valid:true, nextR:nr, nextC:nc, didWarp:true };
    }

    // --- STANDARD VALIDATION (after potential warp check) ---

    // Out of bounds (non-warp)
    if (nr < 0 || nr >= PACMAN_MAZE.length) return { valid:false, nextR:nr, nextC:nc };
    if (nc < 0 || nc >= maxCols) return { valid:false, nextR:nr, nextC:nc };

    // const tile = PACMAN_MAZE[nr][nc];
    // Use the mutable array for checking walls/path
    const tile = currentMazeState[nr][nc];

    // Walls always block
    if (tile === WALL) return { valid:false, nextR:nr, nextC:nc };

    // GHOST LOGIC: CAGE ENTRY/EXIT
    if (entity === "ghost") {
        const isCurrentGateTile = (r === 12 && (c === 13 || c === 14));
        const isTargetGateTile = (nr === 12 && (nc === 13 || nc === 14));
        const isStartingInCage = (currentMazeState[r][c] === GHOST_CAGE);
        const isTargetInCage = (currentMazeState[nr][nc] === GHOST_CAGE);
        if (currentMazeState[nr][nc] === WALL) return { valid:false, nextR:nr, nextC:nc, didWarp:false };

        // 2. Ghosts can move from the cage (r=13, r=14) to the gate tiles (r=12)
        if (isStartingInCage && isTargetGateTile && direction === "up") {
            return { valid:true, nextR:nr, nextC:nc, didWarp:false };
        }

        // 3. Ghosts can move from the gate tiles (r=12) out to the path (r=11)
        if (isCurrentGateTile && direction === "up" && PACMAN_MAZE[nr][nc] === PATH) {
            return { valid:true, nextR:nr, nextC:nc, didWarp:false };
        }

        // 4. They cannot move through walls, unless they are at the gate moving up (handled by 2/3)
        if (PACMAN_MAZE[nr][nc] === WALL) return { valid:false, nextR:nr, nextC:nc, didWarp:false };
    }
    
    // Fallthrough for Pacman and valid ghost moves outside cage/gate
    return { valid:true, nextR:nr, nextC:nc, didWarp:false };
}

// ======================================================
//                    GAME LOOP
// ======================================================

function startGameLoop() {
    if (gameLoopInterval) return;
    gameLoopInterval = setInterval(() => {
        if (!isPaused) {
            if (isAIControlled){
                sendGameState();
            }

            movePacman();
            moveCharacters();
        }
    }, GAME_SPEED);
}

function stopGameLoop() {
    if (gameLoopInterval) {
        clearInterval(gameLoopInterval);
        gameLoopInterval = null;
    }
}


function togglePause() {
    isPaused = !isPaused;
    
    if (isPaused) {
        stopGameLoop(); 
    } else {
        startGameLoop();
        // ghostsActivated = true;
        tryPlayBGM();
        // startGameLoop(); 
        // ghostsActivated = true;
        
        // play bgm
        // if (audioBGM && audioBGM.paused) {
        //     audioBGM.volume = 0;
        //     audioBGM.play();
        //     let fadeIn = setInterval(() => {
        //         if (audioBGM.volume < 0.5) {
        //             audioBGM.volume += 0.05;
        //         } else {
        //             clearInterval(fadeIn);
        //         }
        //     }, 100);
        // }
    }
}

// ======================================================
//             GHOST ACTIVATION DELAY 
// ======================================================

function delayGhostActivation() {
    // Prevent scheduling the delay multiple times
    if (hasGhostActivationScheduled) return;

    hasGhostActivationScheduled = true;
    ghostsActivated = false; // Ensure ghosts stay off during the delay

    setTimeout(() => {
        // Only activate if the game is still running (not stopped/game over)
        if (gameLoopInterval) {
            ghostsActivated = true;
        }
        hasGhostActivationScheduled = false; // Reset the flag for the next death/reset
    }, 1000); // 1000ms = 1 second delay
}

// ======================================================
//          GHOST BEHAVIOR: AI TARGETING
// ======================================================

function manhattan(r1,c1,r2,c2) {
    return Math.abs(r1-r2)+Math.abs(c1-c2);
}

function getBlinkyTarget() {
    return { r:pacmanCurrentRow, c:pacmanCurrentCol };
}

function getPinkyTarget() {
    let r=pacmanCurrentRow, c=pacmanCurrentCol;
    switch(currentDirection) {
        case "up": r-=4; break;
        case "down": r+=4; break;
        case "left": c-=4; break;
        case "right": c+=4; break;
    }
    return {r,c};
}

function getInkyTarget() {
    const blinky = CHARACTERS.traveler;
    let r=pacmanCurrentRow, c=pacmanCurrentCol;

    switch(currentDirection) {
        case "up": r-=2; break;
        case "down": r+=2; break;
        case "left": c-=2; break;
        case "right": c+=2; break;
    }

    const vr = r - blinky.r;
    const vc = c - blinky.c;

    return { r: blinky.r + 2*vr, c: blinky.c + 2*vc };
}

function getClydeTarget(char) {
    const dist = manhattan(char.r, char.c, pacmanCurrentRow, pacmanCurrentCol);
    if (dist < 8) return { r: 27, c:0 }; 
    return { r: pacmanCurrentRow, c: pacmanCurrentCol };
}

function hasLineOfSight(r1,c1,r2,c2) {
    if (r1 === r2) {
        const [minC,maxC]=[Math.min(c1,c2),Math.max(c1,c2)];
        for (let c=minC+1;c<maxC;c++) if (PACMAN_MAZE[r1][c]===WALL) return false;
        return true;
    }
    if (c1 === c2) {
        const [minR,maxR]=[Math.min(r1,r2),Math.max(r1,r2)];
        for (let r=minR+1;r<maxR;r++) if (PACMAN_MAZE[r][c1]===WALL) return false;
        return true;
    }
    return false;
}

// Simplified no-U-turn logic to prevent oscillation.
function getPossibleMoves(r,c,currentDir,entity="player") {
    const dirs = ["up","down","left","right"];
    const out = [];

    for (const d of dirs) {
        const m = checkNextMove(d,r,c,entity);
        if (m.valid) {
            // No U-turn logic
            const reverse =
                (d==="up" && currentDir==="down") ||
                (d==="down" && currentDir==="up") ||
                (d==="left" && currentDir==="right") ||
                (d==="right" && currentDir==="left");

            // Only allow moves that are NOT a direct reversal
            if (!reverse) out.push({direction:d,r:m.nextR,c:m.nextC});
        }
    }
    
    // If we're trapped and the only non-reversing move is a reversal, allow it.
    if (out.length === 0) {
        // Find the reverse direction
        let reverseDir = "";
        if (currentDir === "up") reverseDir = "down";
        else if (currentDir === "down") reverseDir = "up";
        else if (currentDir === "left") reverseDir = "right";
        else if (currentDir === "right") reverseDir = "left";
        
        const m = checkNextMove(reverseDir,r,c,entity);
        if (m.valid) out.push({direction:reverseDir,r:m.nextR,c:m.nextC});
    }

    return out;
}

function chooseDirectionTowardsTarget(char,target) {
    const moves = getPossibleMoves(char.r,char.c,char.direction,"ghost");
    let best=null, bestDist=99999;

    for (const m of moves) {
        const dist = manhattan(m.r,m.c,target.r,target.c);
        if (dist < bestDist) {
            bestDist = dist;
            best = m;
        }
    }

    // If no best move is found, fall back to the first valid move
    return best || moves[0];
}

// ======================================================
//        MAIN GHOST MOVEMENT + ESCAPE LOGIC
// ======================================================

function moveCharacters() {
    if (!ghostsActivated || isPaused || lives <= 0) return;

    // Use this to determine if we need a standard visual update at the end
    let didWarpOccur = false;
    let collisionDetected = false;  // Flag to stop movement loop on collision

    for (const name in CHARACTERS) {
        const char = CHARACTERS[name];

        if (char.r === pacmanCurrentRow && char.c === pacmanCurrentCol) {
            collisionDetected = checkCollision();
            if (collisionDetected) return;
        }

        // Apply speed multiplier
        char.tick += char.speedMultiplier;
        if (char.tick < 1) continue;
        char.tick -= 1;

        let mv = null; // Stores the chosen next move object
        let nextMoveCheckResult = null;
        let target = null;

        // ================================
        //       ESCAPE/CAGE LOGIC (NEW)
        // ================================
        
        // Ghost is currently in the cage (r=13 or r=14, or at gate r=12)
        const isInsideCageArea = (PACMAN_MAZE[char.r][char.c] === GHOST_CAGE || (char.r === 12 && (char.c === 13 || char.c === 14)));
        
        if (char.state === "escape" || isInsideCageArea) {
            
            let preferredDirection = "up";
            
            // If the ghost is inside the cage body (r=13 or r=14) and needs to move horizontally to align with the gate
            if (char.r >= 13 && (char.c < 13 || char.c > 14)) {
                
                // Determine the target column based on the ghost's role for better pathing
                const targetCol = (char.c < 13) ? 13 : 14; 
                preferredDirection = (char.c < targetCol) ? "right" : "left";
            }

            // Attempt the preferred move (UP, or horizontal toward center)
            nextMoveCheckResult = checkNextMove(preferredDirection, char.r, char.c, "ghost");
            
            if (nextMoveCheckResult.valid) {
                mv = nextMoveCheckResult;
                mv.direction = preferredDirection;
            } else {
                // If the preferred move is blocked (e.g., UP is blocked, or horizontal is blocked),
                // try the opposite horizontal direction if applicable (only inside r=13/14)
                if (char.r >= 13) {
                     const oppositeDir = (preferredDirection === "right") ? "left" : "right";
                     nextMoveCheckResult = checkNextMove(oppositeDir, char.r, char.c, "ghost");
                     if (nextMoveCheckResult.valid) {
                         mv = nextMoveCheckResult;
                         mv.direction = oppositeDir;
                     }
                }
            }


            // If a move was successfully chosen during escape/cage, update position
            if (mv && mv.valid) {
                char.r = mv.nextR;
                char.c = mv.nextC;
                char.direction = mv.direction || char.direction;
                
                // If they've reached the path outside the cage (row 11 or higher), switch to chase
                if (char.r < 12 && char.state === "escape") {
                    char.state = "chase";
                    // Fall through to chase logic to immediately pick a real target
                } else {
                    // Execute visual update and continue to next character
                    if (mv.didWarp) {
                         warpCharacter(char.element, char.r, char.c);
                         didWarpOccur = true;
                    }
                    // This ghost has moved one step and is still in the escape process
                    continue; 
                }
            }
        }

        // ================================
        //       CHASE/SCATTER/FRIGHTENED MODE
        // ================================
        
        if (char.state === "chase" || char.state === "frightened") { 
            if (char.state === "frightened"){
                const fleeCorners = [
                    {r: 1, c: 1},
                    {r: 1, c: 26},
                    {r: 27, c: 1},
                    {r: 27, c: 26}
                ];
                // Select a random corner
                target = fleeCorners[Math.floor(Math.random() * fleeCorners.length)];
            }
            else {
                // Choose target based on role
                if (char.role === "blinky")      target = getBlinkyTarget();
                else if (char.role === "pinky")  target = getPinkyTarget();
                else if (char.role === "inky")   target = getInkyTarget();
                else if (char.role === "clyde")  target = getClydeTarget(char);

                // Line-of-sight override
                if (hasLineOfSight(char.r,char.c,pacmanCurrentRow,pacmanCurrentCol)) {
                    target = { r:pacmanCurrentRow, c:pacmanCurrentCol };
                }
            }

            // Move one step toward target
            const chosenMove = chooseDirectionTowardsTarget(char,target);
            
            if (chosenMove) {
                // Check move validity/warp status before committing coordinates
                nextMoveCheckResult = checkNextMove(chosenMove.direction, char.r, char.c, "ghost");
                
                char.r = nextMoveCheckResult.nextR;
                char.c = nextMoveCheckResult.nextC;
                char.direction = chosenMove.direction; // Direction comes from chosen move

                mv = nextMoveCheckResult;
                mv.direction = chosenMove.direction;
            }
        }

        // ================================
        //     FINAL POSITION UPDATE
        // ================================

        if (mv && mv.valid) {
            if (mv.didWarp) {
                warpCharacter(char.element, char.r, char.c);
                didWarpOccur = true;
            }
        }
    }

    // Call standard visual update for all non-warping ghosts.
    if (!didWarpOccur) {
        updateCharacterVisualPositions();
    }

    checkCollision();
}

// ======================================================
//             WEREWOLF MODE (HEART FRAGMENT)
// ======================================================

function activateWerewolfMode() {
    // 1. Clear any existing timer to allow stacking/refreshing the duration
    if (werewolfTimer) {
        clearTimeout(werewolfTimer);
    }

    isWerewolfMode = true;

    // 2. Set all ghosts to the "frightened" state and reverse their direction
    for (const name in CHARACTERS) {
        const char = CHARACTERS[name];
        
        // Only reverse direction if the ghost is currently in "chase" mode
        if (char.state === "chase") {
            // Reverse the ghost's current direction
            if (char.direction === 'up') char.direction = 'down';
            else if (char.direction === 'down') char.direction = 'up';
            else if (char.direction === 'left') char.direction = 'right';
            else if (char.direction === 'right') char.direction = 'left';
            
            // The next movement tick will execute the reversal
        }
        
        char.state = "frightened";
    }

    // 3. Set a timer to end the mode
    werewolfTimer = setTimeout(endWerewolfMode, WEREWOLF_DURATION);
}

function endWerewolfMode() {
    isWerewolfMode = false;
    werewolfTimer = null;

    // Transition ghosts back to 'chase' mode
    for (const name in CHARACTERS) {
        const char = CHARACTERS[name];
        // Only transition ghosts that are frightened back to chase
        if (char.state === "frightened") {
            char.state = "chase";
        }
    }
}

// =======================================================
//             UTILITY FUNCTION FOR WARP
// =======================================================

function warpCharacter(element, newR, newC) {
    // 1. Disable the CSS transition
    element.style.transition = 'none';

    // 2. Immediately update the visual position (teleport)
    element.style.transform = `translate(${newC * cellSize}px, ${newR * cellSize}px)`;

    // 3. Force the browser to apply the style change before re-enabling transition
    element.offsetHeight; 

    // 4. Re-enable the CSS transition
    element.style.transition = 'transform var(--game-speed) linear';
}

// ======================================================
//                PAC-MAN MOVEMENT
// ======================================================

function movePacman() {
    if (isPaused || !currentDirection) return;

    let moveDirection = currentDirection; // Use this for the actual move
    let step = { valid: false };

    // 1. Attempt turn (Turn Buffering Logic)
    if (nextDirection !== currentDirection) {
        step = checkNextMove(nextDirection);
        if (step.valid) {
            currentDirection = nextDirection;
            moveDirection = nextDirection; // Update the move direction to the new direction
        }
    }

    // 2. Attempt movement with the final (or buffered) direction
    step = checkNextMove(moveDirection); 
    if (!step.valid) return;

    // --- EXECUTE MOVE ---
    
    // Update coordinates to the new (potentially warped) position
    if (!ghostsActivated) {
        ghostsActivated = true;
    }

    pacmanCurrentRow = step.nextR;
    pacmanCurrentCol = step.nextC;

    // Eating & Collection Check
    const tile = currentMazeState[pacmanCurrentRow][pacmanCurrentCol];
    
    if (tile === PELLET || tile === HEART_FRAGMENT) { 
        currentMazeState[pacmanCurrentRow][pacmanCurrentCol] = PATH;

        // --- Werewolf Mode Logic ---
        if (tile === HEART_FRAGMENT){
            score += 50; 
            activateWerewolfMode();
        }
        else{
            score += 10;
        }

        // --- WIN/COLLECTION LOGIC ---
        remainingCollectables--; 
        checkWinCondition();

        const cell = getGridCell(pacmanCurrentRow,pacmanCurrentCol);
        if (cell) {
             cell.classList.remove("pellet","heart-fragment");
        }
    }

    // VISUAL UPDATE (Warp or standard)
    if (step.didWarp) {
        warpCharacter(pacmanGraphicElement, pacmanCurrentRow, pacmanCurrentCol);
    } else {
        updatePacmanVisualPosition();
    }
    
    updateScoreboard();
    checkCollision();
}

// ======================================================
//                KEYBOARD INPUT
// ======================================================

function handleKeyDown(e) {
    const key = e.key;
    if (key === " ") { 
        e.preventDefault(); 
        togglePause(); 
        return;
    } 

    if (isAIControlled) return;

    const dirs = { 
        "w":"up","W":"up", "s":"down","S":"down", 
        "a":"left","A":"left", "d":"right","D":"right" 
    };

    if (dirs[key]) {
        e.preventDefault();
        if (isPaused) isPaused = false; // Use new direction to resume the game
        nextDirection = dirs[key];

        // If no movement yet, start moving in this direction
        if (!gameLoopInterval) {
            currentDirection = dirs[key]; 
            tryPlayBGM();
            startGameLoop();

            delayGhostActivation();
        }
    }
}

// function handleKeyDown(e) {
//     const key = e.key;
//     if (key === " ") { 
//         e.preventDefault(); 
//         togglePause(); 
//         return;
//     } 

//     if (isAIControlled) return;

//     // Movement keys
//     const dirs = { 
//         "w":"up","W":"up", 
//         "s":"down","S":"down", 
//         "a":"left","A":"left", 
//         "d":"right","D":"right" 
//     };

//     if (dirs[key]) {
//         e.preventDefault();

//         // Stop pausing on movement key press
//         isPaused = false;
//         nextDirection = dirs[key];
//         // If no movement yet, start moving in this direction
//         if (!currentDirection || !gameLoopInterval) {
//             currentDirection = dirs[key];
//             tryPlayBGM();
//             startGameLoop();
//         }

//         // if (!gameLoopInterval) {
//         //     const { valid } = checkNextMove(nextDirection);
//         //     currentDirection = nextDirection;
//         //     tryPlayBGM();
//         //     startGameLoop();
//         //     // Immediate ghost activation for first movement-key to start
//         //     // ghostsActivated = true; 
//         // }
//     }
//     // if (isPaused) return;

//     if (dirs[key]) {
//         if (isPaused) {
//             isPaused = false;
//         }

//         nextDirection = dirs[key];

//         // This block handles the very first movement key press
//         if (!gameLoopInterval) {
//             const { valid } = checkNextMove(nextDirection);
//             if (valid) {
//                 currentDirection = nextDirection;
//                 tryPlayBGM();
//                 startGameLoop();
      
//                 // Immediate ghost activation for first movement-key to start
//                 // ghostsActivated = true; 
//             }
//         }
//     }
// }

// ======================================================
//                TOUCH INPUT
// ======================================================

let touchStartX = 0, touchStartY = 0;
const SWIPE_THRESHOLD = 50;

function handleTouchSwipe() {
    const grid = document.getElementById("maze-grid");

    grid.addEventListener("touchstart", e => {
        e.preventDefault();
        const t = e.touches[0];
        touchStartX = t.clientX;
        touchStartY = t.clientY;
    }, { passive:false });

    grid.addEventListener("touchend", e => {
        const t = e.changedTouches[0];
        const dx = t.clientX - touchStartX;
        const dy = t.clientY - touchStartY;

        const dist = Math.sqrt(dx*dx + dy*dy);

        if (dist < SWIPE_THRESHOLD) {
            togglePause();
            return;
        }

        if (isPaused) return;

        let dir = null;
        if (Math.abs(dx) > Math.abs(dy)) dir = dx > 0 ? "right" : "left";
        else dir = dy > 0 ? "down" : "up";

        nextDirection = dir;

        if (!gameLoopInterval) {
            const {valid} = checkNextMove(dir);
            if (valid) {
                currentDirection = dir;
                startGameLoop();
                delayGhostActivation();
            }
        }
    }, { passive:false });

    grid.addEventListener("touchmove", e => e.preventDefault(), { passive:false });
}

// ======================================================
//               SCREEN RESIZING
// ======================================================

function resizeMaze() {
    const grid = document.getElementById("maze-grid");
    const card = document.querySelector(".maze-card");
    // const container = grid ? grid.parentElement: null;
    // if (!grid || !container) return;
    if (!grid || !card) return;

    const style = window.getComputedStyle(card);
    const paddingX = parseFloat(style.paddingLeft) + parseFloat(style.paddingRight);
    const availableWidth = card.clientWidth - paddingX - 8; // 8 是 grid 左右各 4px 的 border

    // 2. 严谨计算 cellSize（向下取整避免像素溢出）
    cellSize = Math.floor(availableWidth / 28);

    // 3. 强制网格保持这个精确的像素大小
    grid.style.gridTemplateColumns = `repeat(28, ${cellSize}px)`;
    grid.style.gridAutoRows = `${cellSize}px`;
    grid.style.width = `${cellSize * 28}px`; // 确保网格总宽度等于格子总和

    // const availableWidth = container.clientWidth;
    // // const containerWidth = grid.clientWidth;

    // cellSize = availableWidth/28;

    // grid.style.gridTemplateColumns = `repeat(28, ${cellSize}px)`;
    // grid.style.gridAutoRows = `${cellSize}px`;

    if (grid.offsetWidth > 0) {
        cellSize = grid.offsetWidth / PACMAN_MAZE[0].length;
        pacmanGraphicElement.style.width = `${cellSize}px`;
        pacmanGraphicElement.style.height = `${cellSize}px`;

        for (const n in CHARACTERS) {
            const el = CHARACTERS[n].element;
            el.style.width = `${cellSize}px`;
            el.style.height = `${cellSize}px`;
        }
    }

    updatePacmanVisualPosition();
    updateCharacterVisualPositions();
}

// ======================================================
//                INITIALIZATION
// ======================================================
function tryPlayBGM() {
    if (audioBGM && audioBGM.paused) {
        audioBGM.volume = 0.6;
        audioBGM.play().catch(error => {
            console.log("Waiting for user interaction to play BGM:", error);
        });
    }
}

document.addEventListener("DOMContentLoaded", () => {
    const grid = document.getElementById("maze-grid");

    remainingCollectables = totalCollectables;

    // BGM
    audioBGM = document.getElementById("bgm");

    // Create Pac-Man
    pacmanGraphicElement = document.createElement("div");
    pacmanGraphicElement.classList.add("pacman-graphic","pacman-right");
    grid.appendChild(pacmanGraphicElement);

    // Create ghosts
    createCharacterGraphics();

    initializeMaze();

    // Connect to AI
    connectToAI();

    // AI button
    const aiControlBtns = document.querySelectorAll(".ai-control-btn");
    aiControlBtns.forEach(button => {
        button.addEventListener("click", (event) => {
            const algorithm = event.target.getAttribute("data-algorithm");
            
            // Play BGM on AI start
            tryPlayBGM();

            // Apply AI-Control
            toggleAIControl(true, algorithm);
            gameStarted = true;
            isPaused = false;
            if (currentDirection === null) currentDirection = "left";

            startGameLoop();

            // Set a initial direction to start movement
            // if (!gameStarted) {
            //     currentDirection = "left";
            //     startGameLoop();
            //     // gameStarted = true;
            // }

            // Select specific algorithm to Python server
            if (aiWebSocket && aiWebSocket.readyState === WebSocket.OPEN){
                aiWebSocket.send(JSON.stringify({command: "set_algorithm", name:algorithm}));
                console.log('Sent algorithm selection: ${algorithm}' );
            }

            // Option: Ban other buttons, emphasis present selected button
            aiControlBtns.forEach(btn => btn.classList.remove('active-algo'));
            event.target.classList.add('active-algo');
        })
    })

    // Input handlers
    document.addEventListener("keydown", handleKeyDown);
    handleTouchSwipe();

    resizeMaze();
    updateScoreboard();
});

window.addEventListener("resize", resizeMaze);
setTimeout(resizeMaze, 300);