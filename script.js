// Maze Data (0:Wall, 1:Path, 2:Pellet, 3:Power Pellet, 4:Ghost Cage, 5:Pac-Man Start)
const PACMAN_MAZE = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0],
    [0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0],
    [0, 3, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0],
    [0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0],
    [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0],
    [0, 2, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 2, 0],
    [0, 2, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 2, 0],
    [0, 2, 2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 2, 2, 0],
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 4, 4, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 4, 4, 4, 4, 4, 4, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 0, 4, 4, 4, 4, 4, 4, 0, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0],
    [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0],
    [0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0],
    [0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0],
    [0, 3, 2, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0, 0, 2, 2, 3, 0],
    [0, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 0],
    [0, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 0],
    [0, 2, 2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 2, 2, 0],
    [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
    [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
    [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
];

// --- GAME CONSTANTS AND STATE ---
const WALL = 0;
const PATH = 1;
const PELLET = 2;
const POWER_PELLET = 3;
const GHOST_CAGE = 4;
const PACMAN_START = 5; 

// Initial Pac-Man Position 
let pacmanCurrentRow = 27;
let pacmanCurrentCol = 13;

// Game state variables
let score = 0;
let lives = 3;
let isPaused = false; 

// Direction tracking 
let currentDirection = 'right'; 
let nextDirection = 'right';    

// --- Game Loop Variables ---
let gameLoopInterval;           
const GAME_SPEED = 150;         // Time in milliseconds per step (Pac-Man's actual speed)
const GAME_SPEED_SECONDS = GAME_SPEED / 1000; // 0.15s for CSS transition

// --- Visual State Variables ---
let pacmanGraphicElement = null; // Reference to the single <div> element for Pac-Man
let cellSize = 0;               // Pixel size of a single cell (calculated on resize)
const SWIPE_THRESHOLD = 50;     // Minimum pixels for a recognized swipe
let touchStartX = 0;            // For mobile swipe detection
let touchStartY = 0;            // For mobile swipe detection

// --- HELPER FUNCTIONS ---

/**
 * Retrieves the DOM element for a specific cell in the maze grid.
 */
function getGridCell(row, col) {
    const mazeGrid = document.getElementById('maze-grid');
    const columns = PACMAN_MAZE[0].length; 
    const index = row * columns + col;
    
    if (index >= 0 && index < mazeGrid.children.length) {
        return mazeGrid.children[index];
    }
    return null; 
}

/**
 * The score and lives display in the scoreboard.
 */
function updateScoreboard() {
    document.getElementById('current-score').textContent = score;
    document.getElementById('lives').textContent = lives;
}

/**
 * Calculates Pac-Man's pixel position and updates the CSS transform property 
 * for the smooth movement transition.
 */
function updatePacmanVisualPosition() {
    if (!pacmanGraphicElement || cellSize === 0) return;

    // Calculate the pixel offset for the current row and column
    const translateX = pacmanCurrentCol * cellSize;
    const translateY = pacmanCurrentRow * cellSize;
    
    // Set the CSS variable for the transition duration
    pacmanGraphicElement.style.setProperty('--game-speed', `${GAME_SPEED_SECONDS}s`);
    
    // Apply the transformation for smooth movement
    pacmanGraphicElement.style.transform = `translate(${translateX}px, ${translateY}px)`;
    
    // Update direction class for visual rotation
    pacmanGraphicElement.classList.remove('pacman-up', 'pacman-down', 'pacman-left', 'pacman-right');
    pacmanGraphicElement.classList.add(`pacman-${currentDirection}`);
}

/**
 * Checks if a move in a given direction is valid and returns the new coordinates.
 */
function checkNextMove(direction) {
    let nextR = pacmanCurrentRow;
    let nextC = pacmanCurrentCol;
    const maxCols = PACMAN_MAZE[0].length;
    
    // Calculate new position
    switch (direction) {
        case 'up': nextR -= 1; break;
        case 'down': nextR += 1; break;
        case 'left': nextC -= 1; break;
        case 'right': nextC += 1; break;
    }

    // Handle Tunnel Warping
    if (nextR === 14) { // Row 14 is the tunnel row
        if (nextC < 0) { // Moving off the left side
            nextC = maxCols - 1; 
        }
        else if (nextC >= maxCols) { // Moving off the right side
            nextC = 0; 
        }
    }
    
    // Check array boundaries (Tunnel warping handles the horizontal edges for row 14)
    if (nextR < 0 || nextR >= PACMAN_MAZE.length || (nextR !== 14 && (nextC < 0 || nextC >= maxCols)) ) {
        return { valid: false, nextR: nextR, nextC: nextC };
    }
    
    const nextCellValue = PACMAN_MAZE[nextR][nextC];

    // Check for wall collision
    if (nextCellValue === WALL) {
        return { valid: false, nextR: nextR, nextC: nextC };
    }

    return { valid: true, nextR: nextR, nextC: nextC };
}

// --- GAME LOOP CONTROLS ---

/**
 * Starts the continuous movement game loop.
 */
function startGameLoop() {
    if (gameLoopInterval) return; 

    // The game loop calls movePacman every GAME_SPEED milliseconds
    gameLoopInterval = setInterval(() => {
        movePacman();
    }, GAME_SPEED); 
}

/**
 * Stops the game loop (e.g., when Pac-Man hits a wall or the game pauses).
 */
function stopGameLoop() {
    if (gameLoopInterval) {
        clearInterval(gameLoopInterval);
        gameLoopInterval = null;
    }
}

/**
 * Toggles the game pause state.
 */
function togglePause() {
    if (isPaused) {
        isPaused = false;
        startGameLoop();
        console.log("Game Resumed"); // Add visual cue here later
    } else {
        isPaused = true;
        stopGameLoop();
        console.log("Game Paused"); // Add visual cue here later
    }
}


// --- CORE GAME LOGIC: PAC-MAN MOVEMENT ---

/**
 * The main movement function called by the game loop. 
 */
function movePacman() {
    // Check if paused
    if (isPaused) return; 
    
    let directionToTry = currentDirection;

    // 1. Check Pre-Turn: Attempt to move in the requested direction (nextDirection)
    if (nextDirection !== currentDirection) {
        const { valid } = checkNextMove(nextDirection);

        if (valid) {
            directionToTry = nextDirection;
            currentDirection = nextDirection; 
        }
    }
    
    // 2. Perform Movement Check in the determined direction
    const { valid, nextR, nextC } = checkNextMove(directionToTry);

    if (!valid) {
        // If Pac-Man hits a wall, stop the game loop until a new, valid direction is given.
        stopGameLoop(); 
        return; 
    }

    // --- EXECUTE THE VALID MOVE ---

    // a. Update Pac-Man's position variables
    pacmanCurrentRow = nextR;
    pacmanCurrentCol = nextC;
    
    // b. Handle Eating
    const nextCellValue = PACMAN_MAZE[nextR][nextC];
    if (nextCellValue === PELLET || nextCellValue === POWER_PELLET) {
        
        // Update the PACMAN_MAZE array
        PACMAN_MAZE[nextR][nextC] = PATH;
        
        // Update score
        score += (nextCellValue === PELLET) ? 10 : 50;

        // Clear the pellet from the newly entered cell in the DOM
        const newCell = getGridCell(pacmanCurrentRow, pacmanCurrentCol);
        if (newCell) {
            newCell.classList.remove('pellet', 'power-pellet'); 
        }
    }

    // c. Update the visual position smoothly
    updatePacmanVisualPosition(); 

    // d. Update Scoreboard
    updateScoreboard();
}


// --- INPUT CONTROLS ---

/**
 * Handles keyboard input to set the next direction and start the game loop.
 */
function handleKeyDown(event) {
    let desiredDirection = null;
    const key = event.key;
    
    // 1. Handle Pause/Resume (Spacebar)
    if (key === ' ') {
        event.preventDefault(); 
        togglePause();
        return; // Stop processing any other input if we paused/resumed
    }
    
    // Check if the key is a control key (IJKL or Arrow Keys)
    const isControlKey = 
        key === 'i' || key === 'I' || 
        key === 'k' || key === 'K' || 
        key === 'j' || key === 'J' || 
        key === 'l' || key === 'L' ||
        key === 'ArrowUp' || key === 'ArrowDown' || 
        key === 'ArrowLeft' || key === 'ArrowRight';

    if (isControlKey) {
        // Stop browser default action (e.g., scrolling with arrow keys)
        event.preventDefault(); 
    }
    
    // If game is paused, ignore movement keys
    if (isPaused) return; 
    
    // Assign direction only for the IJKL keys
    switch (key) {
        case 'i':
        case 'I':
            desiredDirection = 'up';
            break;
        case 'k':
        case 'K':
            desiredDirection = 'down';
            break;
        case 'j':
        case 'J':
            desiredDirection = 'left';
            break;
        case 'l':
        case 'L':
            desiredDirection = 'right';
            break;
    }

    if (desiredDirection) {
        nextDirection = desiredDirection; 
        
        // If the game is stopped (e.g., hit a wall), restart movement in the new direction
        if (!gameLoopInterval) {
            // Check if the new direction immediately results in a valid move
            const { valid } = checkNextMove(desiredDirection);
            if (valid) {
                currentDirection = desiredDirection;
                startGameLoop();
            }
        }
    }
}

/**
 * Handles mobile swipe/tap input.
 */
function handleTouchSwipe() {
    const mazeGrid = document.getElementById('maze-grid');

    // 1. Touch Start: Record initial position
    mazeGrid.addEventListener('touchstart', (e) => {
        e.preventDefault(); 
        const touch = e.touches[0];
        touchStartX = touch.clientX;
        touchStartY = touch.clientY;
    }, { passive: false }); 

    // 2. Touch End: Calculate and process the swipe/tap
    mazeGrid.addEventListener('touchend', (e) => {
        const touch = e.changedTouches[0];
        const touchEndX = touch.clientX;
        const touchEndY = touch.clientY;

        const diffX = touchEndX - touchStartX;
        const diffY = touchEndY - touchStartY;

        const distance = Math.sqrt(diffX * diffX + diffY * diffY);

        if (distance < SWIPE_THRESHOLD) {
            // IT'S A TAP - PAUSE/RESUME
            togglePause();
            return; 
        } 
        
        // IT'S A SWIPE - MOVEMENT (only if not paused)
        if (isPaused) return;
        
        let desiredDirection;

        if (Math.abs(diffX) > Math.abs(diffY)) {
            desiredDirection = (diffX > 0) ? 'right' : 'left';
        } else {
            desiredDirection = (diffY > 0) ? 'down' : 'up';
        }
        
        nextDirection = desiredDirection; 

        // If the game is stopped, restart movement in the new direction
        if (!gameLoopInterval) {
             const { valid } = checkNextMove(desiredDirection);
            if (valid) {
                currentDirection = desiredDirection;
                startGameLoop();
            }
        }
    });

    // Optional: Prevent default scrolling behavior during touch move
    mazeGrid.addEventListener('touchmove', (e) => {
        e.preventDefault(); 
    }, { passive: false }); 
}


// --- SCREEN RESIZING LOGIC (Mobile fix) ---
function resizeMaze() {
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;
    const mazeCard = document.querySelector('.maze-card');
    const header = document.querySelector('.header-container');
    const scoreboard = document.querySelector('.scoreboard');
    const mazeGrid = document.getElementById('maze-grid');

    // PC/Medium Screen Logic (Width > 600px)
    if (viewportWidth > 600) { 
        mazeCard.style.maxWidth = ''; 
        mazeCard.style.width = '';
    } else {
         // --- Small/Mobile Screen Scaling Logic (Width <= 600px) ---
        const headerHeight = header ? header.offsetHeight : 0; 
        const scoreboardHeight = scoreboard ? scoreboard.offsetHeight : 0;
        
        const bodyComputedStyle = window.getComputedStyle(document.body);
        const bodyPaddingTop = parseFloat(bodyComputedStyle.paddingTop) || 0;
        const bodyPaddingBottom = parseFloat(bodyComputedStyle.paddingBottom) || 0;
        const bodyGap = parseFloat(bodyComputedStyle.gap) || 0; 
        
        const gameAreaContainer = document.querySelector('.game-area-container');
        const gameAreaComputedStyle = window.getComputedStyle(gameAreaContainer);
        const gameAreaGap = parseFloat(gameAreaComputedStyle.gap) || 0; 
        
        const consumedVerticalSpace = 
            bodyPaddingTop + 
            headerHeight + 
            bodyGap + 
            gameAreaGap + 
            scoreboardHeight + 
            bodyPaddingBottom;

        const availableMazeHeight = viewportHeight - consumedVerticalSpace;
        
        const mazeAspect = 28 / 29;
        const max_width_constraint = viewportWidth * 0.95; 
        const max_height_constraint = availableMazeHeight * mazeAspect;

        const finalMazeSize = Math.min(max_width_constraint, max_height_constraint);

        mazeCard.style.maxWidth = `${finalMazeSize}px`;
        mazeCard.style.width = `${finalMazeSize}px`; 
    }
    
    // --- Calculate and store the cell size (in pixels) for visual movement ---
    if (mazeGrid.offsetWidth > 0) {
        cellSize = mazeGrid.offsetWidth / PACMAN_MAZE[0].length;
        
        // Also update the size of the Pac-Man graphic element
        if (pacmanGraphicElement) {
            pacmanGraphicElement.style.width = `${cellSize}px`;
            pacmanGraphicElement.style.height = `${cellSize}px`;
        }
    }
    
    // Snap Pac-Man to the correct position after resizing (no visible movement)
    updatePacmanVisualPosition(); 
}


// --- INITIALIZATION ---

document.addEventListener('DOMContentLoaded', () => {
    // --- MAZE DRAWING LOGIC ---
    const mazeGrid = document.getElementById('maze-grid');

    const cellClasses = {
        0: 'wall', 1: 'path', 2: 'pellet', 3: 'power-pellet', 4: 'ghost-cage'
        // PACMAN_START (5) is now treated as a regular PATH (1) in the grid background
    };

    PACMAN_MAZE.forEach((row, rowIndex) => {
        row.forEach((cellValue, colIndex) => {
            const cell = document.createElement('div');
            
            // Map the cell value to a class (treat 5 as 1 for drawing purposes)
            const className = cellClasses[cellValue === PACMAN_START ? PATH : cellValue];
            cell.classList.add('cell', className);

            if (rowIndex === 11 && (colIndex === 13 || colIndex === 14)) {
                cell.classList.remove('ghost-cage', 'path', 'pellet', 'power-pellet'); 
                cell.classList.add('path', 'ghost-gate');
            }
            
            mazeGrid.appendChild(cell);
        });
    });
    // --- END MAZE DRAWING LOGIC ---

    // --- NEW: CREATE PAC-MAN GRAPHIC ELEMENT (Only once) ---
    pacmanGraphicElement = document.createElement('div');
    pacmanGraphicElement.classList.add('pacman-graphic');
    pacmanGraphicElement.classList.add(`pacman-${currentDirection}`);
    mazeGrid.appendChild(pacmanGraphicElement);
    
    // Set up Input Handlers
    document.addEventListener('keydown', handleKeyDown);
    handleTouchSwipe(); 

    // Call initialization functions
    resizeMaze(); 
    updateScoreboard();
    
    // NOTE: The game is paused/started by user input (IJKL/Swipe) or the Spacebar/Tap.
});

// Run on window resize events (for responsiveness)
window.addEventListener('resize', resizeMaze);

// Run on a timer for a quick initial sizing when browser elements might be loading
setTimeout(resizeMaze, 300);