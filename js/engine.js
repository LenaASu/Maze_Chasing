// ======================================================
//                    engine.js
// ======================================================

import { INITIAL_CHARACTERS, RERIR_START_R, RERIR_START_C, RERIR_MAZE, WALL, FRAGMENT, HEART_FRAGMENT } from "./config.js";
import { cellSize, updateScoreboard, updateCharacterPosition } from "./render.js";

if (!window.gameState) {
    window.gameState = {};
}

Object.assign(window.gameState, {
    score: 0,
    lives: 3,
    isPaused: false,
    enemyActivated: false,
    currentDirection: null,
    nextDirection: null,
    gameLoopInterval: null,
    isWerewolfMode: false,
    currR: RERIR_START_R,
    currC: RERIR_START_C,
    enemies: {
        enemy1: { r: 14, c: 12, dir: "up", type: "en1" },
        enemy2: { r: 14, c: 13, dir: "up", type: "en2" },
        enemy3: { r: 14, c: 14, dir: "up", type: "en3" },
        enemy4: { r: 14, c: 15, dir: "up", type: "en4" },
    }
});

let isBgmPlaying = false;

export function startGameLoop() {
    if (window.gameState.gameLoopInterval) return;
    window.gameState.gameLoopInterval = setInterval(gameTick, 250); // Remain the same with game.css
}

function moveEnemies() {
    if (!window.gameState.enemyActivated) return;

    for (const name in window.gameState.enemies) {
        const enemy = window.gameState.enemies[name];
        
        // Get all possible moves
        const possibleDirs = ["up", "down", "left", "right"].filter(d => {
            if (!canMove(enemy.r, enemy.c, d)) return false;
            const opposites = { "up": "down", "down": "up", "left": "right", "right": "left" };
            return d !== opposites[enemy.dir];
        });

        // If not possible move, choose a random move
        if (possibleDirs.length > 1 || !canMove(enemy.r, enemy.c, enemy.dir)) {
            if (possibleDirs.length > 0) {
                enemy.dir = possibleDirs[Math.floor(Math.random() * possibleDirs.length)];
            }
        }

        // Execuate move
        const next = getNextPosition(enemy.r, enemy.c, enemy.dir);
        enemy.r = next.r;
        enemy.c = next.c;

        // Update DOM
        const element = document.getElementById(name);
        const cellSize = parseInt(document.getElementById("maze-grid").style.gridAutoRows);
        updateCharacterPosition(element, enemy.r, enemy.c, cellSize);
    }
}

function checkGhostCollision() {
    const state = window.gameState;
    for (const name in state.enemies) {
        const enemy = state.enemies[name];
        if (enemy.r === state.currR && enemy.c === state.currC) {
            if (state.isWerewolfMode) {
                // 
                console.log("Enemy eaten!");
                respawnEnemy(name);
            } else {
                //
                handleRerirDeath();
            }
        }
    }
}

function gameTick() {
    if (window.gameState.isPaused) return;
    const state = window.gameState;
    
    // Move Rerir
    if (state.nextDirection && canMove(state.currR, state.currC, state.nextDirection)) {
        state.currentDirection = state.nextDirection;
        state.nextDirection = null;
    }

    if (state.currentDirection && canMove(state.currR, state.currC, state.currentDirection)) {
        const next = getNextPosition(state.currR, state.currC, state.currentDirection);
        state.currR = next.r;
        state.currC = next.c;
        
        // Check eaten 
        checkEat(state.currR, state.currC);

        // Update Rerir
        const rerirElement = document.getElementById("rerir");
        const currentSize = cellSize || 26; 

        if (rerirElement) {
            updateCharacterPosition(rerirElement, state.currR, state.currC, currentSize);
        }
    }

    if (typeof updateScoreboard === 'function') {
        updateScoreboard(window.gameState.score, window.gameState.lives);
    }
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
    else if (dir === "down") nextR++;
    else if (dir === "left") nextC--;
    else if (dir === "right") nextC++;
    return { r: nextR, c: nextC };
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
    }, 3000); // Activate enemies after 3 seconds
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


