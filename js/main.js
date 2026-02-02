<<<<<<< HEAD
import { INITIAL_CHARACTERS, RERIR_START_R, RERIR_START_C } from './config.js';
import { resizeMaze, initializeMaze, updateCharacterPosition } from './render.js';
import { setupInputHandles } from './input.js';

document.addEventListener("DOMContentLoaded", () => {
    // Make sure gameState is available
    if (!window.gameState) window.gameState = {}; 
    
    // Initialize maze DOM
    initializeMaze(INITIAL_CHARACTERS); 
    const size = resizeMaze(INITIAL_CHARACTERS);

    // Initialize Rerir
    const rerir = document.getElementById("rerir");
    updateCharacterPosition(rerir, window.gameState.currR, window.gameState.currC, size);

    // Initialize enemies from gameState
    for (const id in window.gameState.enemies) {
        window.gameState.enemies[id].el = document.getElementById(id);
        const enemy = window.gameState.enemies[id];
        const element = document.getElementById(id);
        if (element) {
            updateCharacterPosition(element, enemy.r, enemy.c, size);
        }
    }

    setupInputHandles();

    // When resize the window, update all characters
    window.addEventListener("resize", () => {
        const newSize = resizeMaze(INITIAL_CHARACTERS); 
        
        // Rerir
        updateCharacterPosition(rerir, window.gameState.currR, window.gameState.currC, newSize);

        // Enemies
        for (const id in window.gameState.enemies) {
            const enemy = window.gameState.enemies[id];
            const element = document.getElementById(id);
            if (element) {
                updateCharacterPosition(element, enemy.r, enemy.c, newSize);
            }
        }
    });
=======
import { INITIAL_CHARACTERS, RERIR_START_R, RERIR_START_C } from './config.js';
import { resizeMaze, initializeMaze, updateCharacterPosition } from './render.js';
import { setupInputHandles } from './input.js';

document.addEventListener("DOMContentLoaded", () => {
    // Make sure gameState is available
    if (!window.gameState) window.gameState = {}; 
    
    // Initialize maze DOM
    initializeMaze(INITIAL_CHARACTERS); 
    const size = resizeMaze(INITIAL_CHARACTERS);

    // Initialize Rerir
    const rerir = document.getElementById("rerir");
    updateCharacterPosition(rerir, window.gameState.currR, window.gameState.currC, size);

    // Initialize enemies from gameState
    for (const id in window.gameState.enemies) {
        window.gameState.enemies[id].el = document.getElementById(id);
        const enemy = window.gameState.enemies[id];
        const element = document.getElementById(id);
        if (element) {
            updateCharacterPosition(element, enemy.r, enemy.c, size);
        }
    }

    setupInputHandles();

    // When resize the window, update all characters
    window.addEventListener("resize", () => {
        const newSize = resizeMaze(INITIAL_CHARACTERS); 
        
        // Rerir
        updateCharacterPosition(rerir, window.gameState.currR, window.gameState.currC, newSize);

        // Enemies
        for (const id in window.gameState.enemies) {
            const enemy = window.gameState.enemies[id];
            const element = document.getElementById(id);
            if (element) {
                updateCharacterPosition(element, enemy.r, enemy.c, newSize);
            }
        }
    });
>>>>>>> 3eb84a6 (chore: sync local resource files and fix engine logic)
});