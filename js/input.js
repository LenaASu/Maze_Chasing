// ======================================================
//                    input.js
// ======================================================

import { RERIR_MAZE, WALL } from './config.js';
import { updateCharacterPosition } from './render.js';
import { togglePause, startGameLoop, delayEnemyActivation, tryPlayBGM, toggleAIControl} from './engine.js';


export function setupInputHandles() {
    // Initialize
    if (!window.gameState) window.gameState = { score: 0, lives: 3 };
    if (window.gameState.score === undefined) window.gameState.score = 0;

    // Keyboard Controls
    document.addEventListener("keydown", (e) => {
        const key = e.key;

        // Pause
        if (key === " ") {
            e.preventDefault();
            togglePause();
            return;
        }

        // Move Rerir
        const dirs = {
            "w": "up", "W": "up",
            "s": "down", "S": "down",
            "a": "left", "A": "left",
            "d": "right", "D": "right"
        };

        if (key in dirs) {   
            e.preventDefault();
            window.gameState.nextDirection = dirs[key];
       
            if (!window.gameState.gameLoopInterval && !window.gameState.isPaused) {
                window.gameState.currentDirection = dirs[key];
                tryPlayBGM();
                startGameLoop();
                delayEnemyActivation();
            }
        }
    });
    
    let touchStartX = 0;
    let touchStartY = 0;

    // Touch controls
    document.addEventListener('touchstart', (e) => {
        touchStartX = e.changedTouches[0].screenX;
        touchStartY = e.changedTouches[0].screenY;
    }, { passive: true });

    document.addEventListener('touchend', (e) => {
        const touchEndX = e.changedTouches[0].screenX;
        const touchEndY = e.changedTouches[0].screenY;
        handleSwipe(touchStartX, touchStartY, touchEndX, touchEndY);

        if (!window.gameState.gameLoopInterval && !window.gameState.isPaused) {
            tryPlayBGM();
            startGameLoop();
            delayEnemyActivation();
        }
    }, { passive: true });

    function handleSwipe(startX, startY, endX, endY) {
        const dx = endX - startX;
        const dy = endY - startY;
        const absX = Math.abs(dx);
        const absY = Math.abs(dy);
        const threshold = 30; // Prevent untended touch

        if (Math.max(absX, absY) > threshold) {
            if (absX > absY) {
                // Horizontal scroll
                window.gameState.nextDirection = dx > 0 ? "right" : "left";
            } else {
                // Vertical scroll
                window.gameState.nextDirection = dy > 0 ? "down" : "up";
            }
            console.log("Swipe Direction:", window.gameState.nextDirection);
        }
    }

    // AI Control Buttons
    const aiBtns = document.querySelectorAll(".ai-control-btn");
    aiBtns.forEach(btn => {
        btn.addEventListener("click", (e) => {
            const algo = e.target.getAttribute("data-algo");
            tryPlayBGM();
            toggleAIControl(true, algo);

            if (!window.gameState.currentDirection) window.gameState.currentDirection = "left";
            startGameLoop();
        });
    });
}

export function updateGlobalRerirPos(r,c) {
    currR = r;
    currC = c;

}