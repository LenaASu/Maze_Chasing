// ======================================================
//                    logic.js
// ======================================================

import { RERIR_MAZE, WALL } from "./config.js";

export function canMove(r, c) {
    if (r < 0 || r > RERIR_MAZE.length - 1 || c < 0 || c > RERIR_MAZE[0].length - 1) return false;
    return RERIR_MAZE[r][c] !== WALL;
}

export function checkCollision(rerirPos, enemyPos) {
    for (const name in enemyPos) {
        const enemy = enemyPos[name];
        if (rerirPos.r === enemy.r && rerirPos.c === enemy.c) {
            if (window.gameState.isWerewolf) {
                resetSingleEnemy(enemy);
                window.gameState.score += 200;
                return "enemyDefeated";
            } else {
                return "rerirDefeated";
            }
        }
    }
    return null;
}

function resetSingleEnemy(enemy) {
    enemy.r = 14;
    enemy.c = 13;
    enemy.state = "escape";
}