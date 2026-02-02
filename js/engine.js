if (!window.gameState) {
    window.gameState = {};
}

Object.assign(window.gameState, {
    score: 0,
    lives: 3,
    totalFragments: 0,
    isPaused: false,
    enemyActivated: false,
    currentDirection: null,
    nextDirection: null,
    gameLoopInterval: null,
    isWerewolfMode: false,
    currR: RERIR_START_R,
    currC: RERIR_START_C,
    enemies: {
        enemy1: { r: 14, c: 11, dir: "up", type: "en1" },
        enemy2: { r: 14, c: 12, dir: "up", type: "en2" },
        enemy3: { r: 14, c: 13, dir: "up", type: "en3" },
        enemy4: { r: 14, c: 14, dir: "up", type: "en4" },
    }
});

import { INITIAL_CHARACTERS, RERIR_START_R, RERIR_START_C, RERIR_MAZE, WALL, FRAGMENT, HEART_FRAGMENT } from "./config.js";
import { cellSize, updateScoreboard, updateCharacterPosition } from "./render.js";

let isBgmPlaying = false;

export function startGameLoop() {
    if (window.gameState.gameLoopInterval) return;

    // Calculate fragments
    window.gameState.totalFragments = RERIR_MAZE.flat().filter(cell => 
        cell === FRAGMENT || cell === HEART_FRAGMENT
    ).length;
    // console.log("Original # of total fragments is: ", window.gameState.totalFragments);

    window.gameState.gameLoopInterval = setInterval(gameTick, 200);
}

export function countInitialItems() {
    // Use flat() to calculate # of elements
    window.gameState.totalItems = RERIR_MAZE.flat().filter(cell => 
        cell === FRAGMENT || cell === HEART_FRAGMENT
    ).length;
    
    console.log("Total fragments to collect:", window.gameState.totalItems);
}

function moveEnemies() {
    if (!window.gameState.enemyActivated) return;

    const grid = document.getElementById("maze-grid");
    const cellSize = grid ? grid.offsetWidth / RERIR_MAZE[0].length : 26;

    for (const name in window.gameState.enemies) {
        const enemy = window.gameState.enemies[name];
        const element = document.getElementById(name);

        // Check wall/boundary
        let next = getNextPosition(enemy.r, enemy.c, enemy.dir);

        if (RERIR_MAZE[next.r][next.c] === WALL || Math.random() < 0.1) {
            const dirs = ["up", "down", "left", "right"];
            const validDirs = dirs.filter(d => {
                const p = getNextPosition(enemy.r, enemy.c, d);
                return RERIR_MAZE[p.r] && RERIR_MAZE[p.r][p.c] !== WALL;
            });
            
            if (validDirs.length > 0) {
                enemy.dir = validDirs[Math.floor(Math.random() * validDirs.length)];
                next = getNextPosition(enemy.r, enemy.c, enemy.dir); 
            }
        }

        // Move
        const isWarping = Math.abs(next.c - enemy.c) > 1;
        enemy.r = next.r;
        enemy.c = next.c;

        if (element) {
            if (isWarping) element.style.transition = 'none';
            updateCharacterPosition(element, enemy.r, enemy.c, cellSize);
            if (isWarping) {
                setTimeout(() => {
                    element.style.transition = 'transform 0.2s linear';
                }, 50);
            }
        }
    }
}

function respawnEnemy(id){
    const initialPos = {
        enemy1: {r: 14, c: 11},
        enemy2: {r: 14, c: 12},
        enemy3: {r: 14, c: 13},
        enemy4: {r: 14, c: 14},
    };

    const pos = initialPos[id];
    if (pos) {
        window.gameState.enemies[id].r = pos.r;
        window.gameState.enemies[id].c = pos.c;
        window.gameState.enemies[id].dir = "up"; // Reset the respawned initial move

        const element = document.getElementById(id);
        const grid = document.getElementById("maze-grid");
        const cellSize = grid ? parseInt(grid.style.gridAutoRows) : 26;

        updateCharacterPosition(element, pos.r, pos.c, cellSize);
    }
}

function respawnRerir() {
    // Reset position
    window.gameState.currR = RERIR_START_R;
    window.gameState.currC = RERIR_START_C;
    
    // Clear moving status
    window.gameState.currentDirection = null;
    window.gameState.nextDirection = null;

    // Update visual position
    const rerirEl = document.getElementById("rerir");
    if (rerirEl) {
        // Calculate current grid size for resizing maze
        const grid = document.getElementById("maze-grid");
        const currentSize = grid ? grid.offsetWidth / RERIR_MAZE[0].length : 26;
        
        // Render
        updateCharacterPosition(rerirEl, RERIR_START_R, RERIR_START_C, currentSize);
    }
    
    // console.log("Rerir has respawned.");
}

function checkCollision() {
    const state = window.gameState;
    
    for (const name in state.enemies) {
        const enemy = state.enemies[name];
        
        const distR = Math.abs(enemy.r - state.currR);
        const distC = Math.abs(enemy.c - state.currC);

        if (distR < 0.8 && distC < 0.8) {
            if (state.isWerewolfMode) {
                console.log("Rerir ate enemy:", name);
                state.score += 200;
                respawnEnemy(name);
                updateScoreboard(state.score, state.lives);
            } else {
                console.log("Caught by:", name);
                handleRerirDeath();
                return; 
            }
        }
    }
}

function handleRerirDeath() {
    window.gameState.lives--;
    updateScoreboard(window.gameState.score, window.gameState.lives);

    if (window.gameState.lives <= 0) {
        stopGameLoop();
        setTimeout(() => {
            alert("⛓️‍💥 YOU ARE ARRESTED! Final score: " + window.gameState.score);
            location.reload();
        }, 100);
        return;
    } else {
        window.gameState.currR = RERIR_START_R;
        window.gameState.currC = RERIR_START_C;
        window.gameState.currentDirection = null;
        window.gameState.nextDirection = null;

        Object.keys(window.gameState.enemies).forEach(id => {
            respawnEnemy(id);
        });

        window.gameState.enemyActivated = false;
        delayEnemyActivation();
        // console.log("Rerir respawned. Lives left:" + window.gameState.lives);
    }
    const grid = document.getElementById("maze-grid");
    const cellSize = grid ? grid.offsetWidth / RERIR_MAZE[0].length : 26;
    
    updateCharacterPosition(document.getElementById("rerir"), window.gameState.currR, window.gameState.currC, cellSize);
}

function handleRerirVictory() {
    stopGameLoop(); 

    setTimeout(() => {
        alert("🎉🐺 Rerir Werewolf! You collected all fragments! \nFinal Score: " + window.gameState.score);
        location.reload(); 
    }, 300);
}

function gameTick() {
    if (window.gameState.isPaused) return;
    const state = window.gameState;

    const grid = document.getElementById("maze-grid");
    const currentSize = grid ? grid.offsetWidth / RERIR_MAZE[0].length : 26; 
    
    checkCollision();

    // Move Rerir
    if (state.nextDirection && canMove(state.currR, state.currC, state.nextDirection)) {
        state.currentDirection = state.nextDirection;
        state.nextDirection = null;
    }

    if (state.currentDirection && canMove(state.currR, state.currC, state.currentDirection)) {
        const next = getNextPosition(state.currR, state.currC, state.currentDirection);
        
        // Check if using tunnel
        const isWarping = Math.abs(next.c - state.currC) > 1;

        if (RERIR_MAZE[next.r][next.c] !== WALL) {
            const rerirEl = document.getElementById("rerir");
            
            if (isWarping && rerirEl) {
                rerirEl.style.transition = 'none'; 
                rerirEl.offsetHeight; 
            }

            state.currR = next.r;
            state.currC = next.c;

            updateCharacterPosition(rerirEl, state.currR, state.currC, currentSize);

            if (isWarping && rerirEl) {
                // 
                requestAnimationFrame(() => {
                    rerirEl.style.transition = 'transform 0.2s linear';
                });
            }
        }
        checkEat(state.currR, state.currC);
    }

    checkCollision();

    // Move enemies
    moveEnemies();
    checkCollision();
}

function canMove(r, c, dir) {
    const next = getNextPosition(r, c, dir);
    // Check if out of boundary or hit wall
    if (next.r < 0 || next.r >= RERIR_MAZE.length || next.c < 0 || next.c >= RERIR_MAZE[0].length) return false;
    return RERIR_MAZE[next.r][next.c] !== WALL;
}

function getNextPosition(r, c, dir) {
    let nextR = r;
    let nextC = c;
    if (dir === "up") nextR--;
    if (dir === "down") nextR++;
    if (dir === "left") nextC--;
    if (dir === "right") nextC++;

    // Tunnel
    const mazeWidth = RERIR_MAZE[0].length;
    if (nextC < 0) {
        nextC = mazeWidth - 1;
    }
    else if (nextC >= mazeWidth) {
        nextC = 0;
    }

   return { r: nextR, c: nextC };
}

function getOppositeDir(dir) {
    const opposites = {
        "up": "down",
        "down": "up",
        "left": "right",
        "right": "left"
    };
    return opposites[dir] || null;
}

function checkEat(r, c) {
    const cellValue = RERIR_MAZE[r][c];
    
    // Set score = 0 if NaN or undefined
    if (isNaN(window.gameState.score) || window.gameState.score === undefined) {
        window.gameState.score = 0;
    }

    if (cellValue === FRAGMENT || cellValue === HEART_FRAGMENT) {
        const points = (cellValue === FRAGMENT ? 10 : 50);
        
        // Add scores
        window.gameState.score += points;
        // console.log("Eaten:", points, "Current score:", window.gameState.score);

        // Elements => Path
        RERIR_MAZE[r][c] = 1; 

        // Remove elements
        const tile = document.getElementById(`tile-${r}-${c}`);
        if (tile) {
            tile.classList.remove("fragment", "heart-fragment");
        }

        // Update score
        updateScoreboard(window.gameState.score, window.gameState.lives);
        
        if (cellValue === HEART_FRAGMENT) {
            activateWerewolfMode();
        }

        // Win the game
        window.gameState.totalFragments--;
        if (window.gameState.totalFragments === 0) {
            handleRerirVictory();
        }

    }
}

function activateWerewolfMode() {
    window.gameState.isWerewolfMode = true;
    console.log("Werewolf Mode Active!");
    setTimeout(() => {
        window.gameState.isWerewolfMode = false;
    }, 8000); // Recover after 8 seconds
}

export function stopGameLoop() {
    if (window.gameState.gameLoopInterval) {
        clearInterval(window.gameState.gameLoopInterval);
        window.gameState.gameLoopInterval = null;
    }
}

export function togglePause() { 
    window.gameState.isPaused = !window.gameState.isPaused;
    console.log(window.gameState.isPaused ? "Paused" : "Resumed");  
}

export function toggleAIControl(enable, algorithm = "basic") {
    window.gameState.aiControl = enable ? algorithm : null;
    console.log(enable ? `AI Control Enabled: ${algorithm}` : "AI Control Disabled");
}

export function delayEnemyActivation() {
    setTimeout(() => {
        window.gameState.enemyActivated = true;
        console.log("Enemies Activated");
    }, 500); // Activate enemies after 0.5 seconds
}

export function tryPlayBGM() {
    const bgm = document.getElementById("bgm");
    if (bgm && !isBgmPlaying) {
        bgm.volume = 0.3;

        bgm.play().then(() => {
            isBgmPlaying = true;
            console.log("BGM started playing.");
        }).catch(err => {
            console.warn("Autoplay blocked by browser. Waiting for interaction.", err);
        });
    }
}


