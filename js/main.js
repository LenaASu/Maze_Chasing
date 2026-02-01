// ======================================================
//                    main.js
// ======================================================

import { INITIAL_CHARACTERS, RERIR_START_R, RERIR_START_C } from './config.js';
// import { startGameLoop } from './engine.js';
import { resizeMaze, initializeMaze, updateCharacterPosition } from './render.js';
import { setupInputHandles } from './input.js';

// Initialize maze and start game loop
document.addEventListener("DOMContentLoaded", () => {
    // Initialize
    window.gameState = {
        currR: RERIR_START_R,
        currC: RERIR_START_C,
        nextDirection: null,
        currentDirection: null
    }

    initializeMaze(INITIAL_CHARACTERS);
    const size = resizeMaze(INITIAL_CHARACTERS);

    const rerir = document.getElementById("rerir");
    updateCharacterPosition(rerir, RERIR_START_R, RERIR_START_C, size);

    for (const n in INITIAL_CHARACTERS) {
        const charData = INITIAL_CHARACTERS[n];
        updateCharacterPosition(charData.element, charData.r, charData.c, size);
    }

    setupInputHandles();

    window.addEventListener("resize", () => {
        const newSize = resizeMaze(INITIAL_CHARACTERS); 
        const rerir = document.getElementById("rerir");

        const r = window.gameState ? window.gameState.currR : RERIR_START_R;
        const c = window.gameState ? window.gameState.currC : RERIR_START_C;

        updateCharacterPosition(rerir, r, c, newSize);

        for (const n in INITIAL_CHARACTERS) {
            const charData = INITIAL_CHARACTERS[n];
            if (charData.element) {
                updateCharacterPosition(charData.element, charData.r, charData.c, newSize);
            }
        }
    });
});
