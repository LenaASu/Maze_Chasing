// ======================================================
//                    render.js
// ======================================================

import { RERIR_MAZE, WALL, PATH, FRAGMENT, HEART_FRAGMENT, ENEMY_START, RERIR_START } from './config.js';
import { syncInitialPositions } from './engine.js';

// DOM element references
let rerirGraphicElement = null;
let cellSize = 0;

/**
 * Initialize the maze grid in the DOM
 */
export function initializeMaze(character) {
    const grid = document.getElementById("maze-grid");
    if (!grid) return;
    
    grid.innerHTML = ""; // Clear existing grid if any
    
    // Create grid tiles based on RERIR_MAZE configuration
    RERIR_MAZE.forEach((row, r) => {
        row.forEach((cell, c) => {
            const tile = document.createElement("div");
            tile.classList.add("tile");
            
            // Name the tile based on its type
            if (cell === WALL) tile.classList.add("wall");
            else if (cell === FRAGMENT) tile.classList.add("fragment");
            else if (cell === HEART_FRAGMENT) tile.classList.add("heart-fragment");
            else if (cell === ENEMY_START) tile.classList.add("enemy-start");
            
            tile.id = `tile-${r}-${c}`;
            grid.appendChild(tile);
        });
    });

    // Get reference to Rerir and enemies graphic element
    let rerir = document.getElementById("rerir");
    if (!rerir) {
        rerir = document.createElement("div");
        rerir.id = "rerir";
        // grid.appendChild(rerir);
    }
    rerir.className = "rerir-graphic";
    rerirGraphicElement = rerir;
    grid.appendChild(rerir);

    for (const n in character) {
        const charData = character[n];
        let el = document.getElementById(n);
        if (!el) {
            el = document.createElement("div");
            el.id = n;
            el.classList.add("enemy-graphic", `${n}-graphic`);
            grid.appendChild(el);
        }
        charData.element = el;
    }

    rerirGraphicElement = rerir;
    requestAnimationFrame(() => {
        syncInitialPositions();
    });
}

/**
 * Resize the maze and character elements based on the container size
 * @param {Object} CHARACTERS - The characters object containing enemy elements
 * @returns {number} - The calculated cell size in pixels
 */
export function resizeMaze(CHARACTERS) {
    const grid = document.getElementById("maze-grid");
    const card = document.querySelector(".game-area-container");
    // const container = document.querySelector(".game-area-container");
    if (!grid || !card) return;

    const maxWidth = 500;

    // 1. Get available width inside the card, accounting for padding
    let availableWidth = Math.min(card.clientWidth - 100, maxWidth); 

    // 2. Get cell size by dividing available width by number of columns (26)
    cellSize = Math.floor(availableWidth / 26);
    // cellSize = Math.floor(baseWidth/26);
    const totalWidth = cellSize * 26;
    grid.style.width = `${totalWidth}px`;

    // 3. Update grid styles
    grid.style.gridTemplateColumns = `repeat(26, ${cellSize}px)`;
    grid.style.gridAutoRows = `${cellSize}px`;
    grid.style.width = `${cellSize * 26}px`; 
    grid.style.margin = "0 auto";  

    // 4. Update Rerir graphic size
    if (rerirGraphicElement) {
        rerirGraphicElement.style.width = `${cellSize}px`;
        rerirGraphicElement.style.height = `${cellSize}px`;
    }

    // Update enemy graphic sizes
    for (const n in CHARACTERS) {
        const el = CHARACTERS[n].element;
        if (el) {
            el.style.width = `${cellSize}px`;
            el.style.height = `${cellSize}px`;
        }
    }

    return cellSize;
}

/**
 * Update character position in the maze
 * @param {HTMLElement} element - The character's DOM element
 * @param {number} row - The row index in the maze
 * @param {number} col - The column index in the maze
 * @param {number} cellSize - The size of each cell in pixels
 */
export function updateCharacterPosition(element, row, col, cellSize) {
    if (!element) return;
    const x = Math.round(col * cellSize);
    const y = Math.round(row * cellSize);
    
    element.style.transform = `translate3d(${x}px, ${y}px, 0)`;
    // element.style.transform = `translate(${col * cellSize}px, ${row * cellSize}px)`;
}

/**
 * Update the scoreboard display
 * @param {number} score - The current score
 * @param {number} lives - The remaining lives
 */
export function updateScoreboard(score, lives) {
    const scoreVal = document.getElementById("score-val");
    const livesVal = document.getElementById("lives-val");
    if (scoreVal) scoreVal.textContent = isNaN(score) ? 0:score;
    if (livesVal) livesVal.textContent = (lives === undefined || lives === null) ? 3 : lives;
    
}
window.updateScoreboard = updateScoreboard;
export { cellSize, rerirGraphicElement };