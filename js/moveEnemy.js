import { INITIAL_CHARACTERS } from "./config";
import { rerirGraphicElement } from "./render";
import { RERIR_MAZE } from "./config";
// import { INITIAL_CHARACTERS, RERIR_START_R, RERIR_START_C, RERIR_MAZE, WALL, FRAGMENT, HEART_FRAGMENT } from "./config.js";
// import { cellSize, updateScoreboard, updateCharacterPosition } from "./render.js";


function manhattan(r1,c1,r2,c2) {
    return Math.abs(r1-r2) + Math.abs(c1-c2);
}

function getOpposite(dir) {
    const opposites = { 'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'};
    return opposites[dir] || null;
}

function getNextEnemyStep(enemy, target, maze) {
    const dirs = [
        {r:-1, c:0, name:'up'}, 
        {r:1, c:0, name:'down'},
        {r:0, c:-1, name:'left'},
        {r:0, c:1, name:'right'},   
    ];

    const moves = dirs.filter(d=> {
        const nr = enemy.r + d.r, nc = enemy.c + d.c;
        return maze[nr] && maze[nr][nc] !== 0 && d.name !== getOpposite(enemy.lastDir);
    });

    if (moves.length === 0) return null;

    return moves.sort((a,b) => {
        const d1 = manhattan(enemy.r + a.r, enemy.c + a.c, target.r, target.c);
        const d2 = manhattan(enemy.r + b.r, enemy.c + b.c, target.r, target.c);
        return d1 - d2;
    })[0];
}

export function moveEnemies(id, char, rerir, currentRerirDir) {
    let r = rerir.r, c = rerir.c;
    switch(id) {
        case 'enemy1':
            return { r, c };

        case 'enemy2': {
            if (currentRerirDir === 'up') r-=4;
            else if (currentRerirDir === 'down') r+=4;
            else if (currentRerirDir === 'left') c-=4;
            else if (currentRerirDir === 'right') c+=4;
            return { r, c };
        }

        case 'enemy3': {
            const vr = r - char.r;
            const vc = c - char.c;
            return { r: char.r + 2*vr, c: char.c + 2*vc};
        }

        case 'enemy4': {
            const d = Math.abs(char.r - r) + Math.abs(char.c - c);
            return d < 8 ? { r: 27, c: 0} : { r, c };
        }

        default:
            return { r, c };
    }
}
