(function () {
  "use strict";

  const LOGICAL_WIDTH = 1280;
  const LOGICAL_HEIGHT = 720;
  const FLOOR_RATIO = 0.79;
  const MATRIX_CHARS = "01ABCDEFGHIJKLMNOPQRSTUVWXYZ$%#@[]{}<>+-=*";
  const STORAGE_KEY = "matrix_dash_best_score";

  const canvas = document.getElementById("gameCanvas");
  const messageOverlay = document.getElementById("messageOverlay");
  const scoreValue = document.getElementById("scoreValue");
  const bestValue = document.getElementById("bestValue");
  const jumpButton = document.getElementById("jumpButton");

  const ctx = canvas.getContext("2d", { alpha: false });

  let dpr = 1;
  let gameState = "ready";
  let worldSpeed = 420;
  let spawnTimer = 0;
  let score = 0;
  let bestScore = Number.parseInt(localStorage.getItem(STORAGE_KEY) || "0", 10) || 0;
  let gameTime = 0;
  let flashTimer = 0;
  let lastTime = performance.now();

  bestValue.textContent = String(bestScore);

  const player = {
    x: LOGICAL_WIDTH * 0.2,
    y: 0,
    size: 58,
    vy: 0,
    grounded: true,
    rotation: 0
  };

  const obstacles = [];
  const particles = [];
  const matrixColumns = [];
  const trail = [];

  function floorY() {
    return LOGICAL_HEIGHT * FLOOR_RATIO;
  }

  function updateOverlay(html, hidden) {
    messageOverlay.innerHTML = html;
    messageOverlay.classList.toggle("hidden", hidden);
  }

  function setReadyOverlay() {
    updateOverlay(
      "PRESS JUMP TO JACK IN<br><span class='hint'>SPACE / ARROW UP / TAP</span>",
      false
    );
  }

  function setGameOverOverlay() {
    updateOverlay(
      `SIGNAL LOST<br><span class='hint'>SCORE ${score} | BEST ${bestScore}</span><span class='hint'>PRESS JUMP TO REBOOT</span>`,
      false
    );
  }

  function clearOverlay() {
    updateOverlay("", true);
  }

  function resizeCanvas() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.floor(LOGICAL_WIDTH * dpr);
    canvas.height = Math.floor(LOGICAL_HEIGHT * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.imageSmoothingEnabled = true;
  }

  function randomBetween(min, max) {
    return min + Math.random() * (max - min);
  }

  function initializeColumns() {
    matrixColumns.length = 0;
    const spacing = 20;
    const count = Math.ceil(LOGICAL_WIDTH / spacing) + 2;
    for (let i = 0; i < count; i += 1) {
      matrixColumns.push({
        x: i * spacing + randomBetween(-7, 7),
        y: randomBetween(-LOGICAL_HEIGHT, LOGICAL_HEIGHT),
        speed: randomBetween(150, 480),
        length: Math.floor(randomBetween(7, 24))
      });
    }
  }

  function resetGame() {
    gameState = "ready";
    worldSpeed = 420;
    spawnTimer = 0.9;
    score = 0;
    gameTime = 0;
    flashTimer = 0;
    obstacles.length = 0;
    particles.length = 0;
    trail.length = 0;

    player.size = 58;
    player.x = LOGICAL_WIDTH * 0.2;
    player.y = floorY() - player.size;
    player.vy = 0;
    player.grounded = true;
    player.rotation = 0;

    scoreValue.textContent = "0";
    setReadyOverlay();
  }

  function startRun() {
    if (gameState === "playing") {
      return;
    }
    if (gameState === "dead") {
      resetGame();
    }
    gameState = "playing";
    clearOverlay();
    triggerJump();
  }

  function triggerJump() {
    if (gameState === "ready") {
      startRun();
      return;
    }

    if (gameState === "dead") {
      startRun();
      return;
    }

    if (!player.grounded) {
      return;
    }

    player.grounded = false;
    player.vy = -1020;
    for (let i = 0; i < 8; i += 1) {
      particles.push({
        x: player.x + player.size * randomBetween(0.25, 0.8),
        y: player.y + player.size * 0.88,
        vx: randomBetween(-140, 40),
        vy: randomBetween(-80, -12),
        life: randomBetween(0.23, 0.42),
        size: randomBetween(2, 5),
        color: "rgba(130, 255, 185, 0.9)"
      });
    }
  }

  function spawnObstacle(type, offsetX) {
    if (type === "spike") {
      obstacles.push({
        type,
        x: LOGICAL_WIDTH + offsetX,
        width: 52,
        height: 72
      });
      return;
    }
    obstacles.push({
      type: "pillar",
      x: LOGICAL_WIDTH + offsetX,
      width: randomBetween(56, 74),
      height: randomBetween(88, 146)
    });
  }

  function spawnPattern() {
    const r = Math.random();
    if (r < 0.34) {
      spawnObstacle("spike", 0);
      return;
    }
    if (r < 0.6) {
      spawnObstacle("spike", 0);
      spawnObstacle("spike", 78);
      return;
    }
    if (r < 0.82) {
      spawnObstacle("pillar", 0);
      return;
    }
    spawnObstacle("spike", 0);
    spawnObstacle("pillar", 140);
  }

  function toRect(obstacle) {
    const yBase = floorY();
    if (obstacle.type === "spike") {
      return {
        x: obstacle.x + obstacle.width * 0.2,
        y: yBase - obstacle.height * 0.68,
        width: obstacle.width * 0.6,
        height: obstacle.height * 0.68
      };
    }
    return {
      x: obstacle.x + 2,
      y: yBase - obstacle.height,
      width: obstacle.width - 4,
      height: obstacle.height
    };
  }

  function intersects(a, b) {
    return !(
      a.x + a.width < b.x ||
      b.x + b.width < a.x ||
      a.y + a.height < b.y ||
      b.y + b.height < a.y
    );
  }

  function crash() {
    if (gameState !== "playing") {
      return;
    }
    gameState = "dead";
    flashTimer = 0.2;
    if (score > bestScore) {
      bestScore = score;
      localStorage.setItem(STORAGE_KEY, String(bestScore));
      bestValue.textContent = String(bestScore);
    }

    for (let i = 0; i < 28; i += 1) {
      particles.push({
        x: player.x + player.size * 0.5,
        y: player.y + player.size * 0.5,
        vx: randomBetween(-320, 240),
        vy: randomBetween(-260, 170),
        life: randomBetween(0.4, 0.95),
        size: randomBetween(2, 8),
        color: i % 2 === 0 ? "rgba(130, 255, 185, 1)" : "rgba(255, 72, 104, 0.92)"
      });
    }
    setGameOverOverlay();
  }

  function updatePlayer(dt) {
    const gravity = 2920;
    player.vy += gravity * dt;
    player.y += player.vy * dt;
    const floor = floorY();

    if (player.y + player.size >= floor) {
      player.y = floor - player.size;
      if (!player.grounded && player.vy > 0) {
        for (let i = 0; i < 4; i += 1) {
          particles.push({
            x: player.x + player.size * randomBetween(0.25, 0.8),
            y: floor - 2,
            vx: randomBetween(-50, 110),
            vy: randomBetween(-70, -20),
            life: randomBetween(0.14, 0.28),
            size: randomBetween(2, 4),
            color: "rgba(93, 255, 162, 0.78)"
          });
        }
      }
      player.vy = 0;
      player.grounded = true;
    }

    if (player.grounded) {
      player.rotation += (0 - player.rotation) * Math.min(dt * 14, 1);
    } else {
      player.rotation += dt * 8.5;
    }

    trail.push({
      x: player.x + player.size * 0.45,
      y: player.y + player.size * 0.45,
      life: randomBetween(0.12, 0.22)
    });
    if (trail.length > 34) {
      trail.shift();
    }
  }

  function updateObstacles(dt) {
    spawnTimer -= dt;
    if (spawnTimer <= 0) {
      spawnPattern();
      const speedFactor = Math.min((worldSpeed - 420) / 260, 1);
      spawnTimer = randomBetween(0.84, 1.45) - speedFactor * 0.18;
    }

    for (let i = obstacles.length - 1; i >= 0; i -= 1) {
      const obstacle = obstacles[i];
      obstacle.x -= worldSpeed * dt;
      if (obstacle.x + obstacle.width < -140) {
        obstacles.splice(i, 1);
      }
    }
  }

  function updateParticles(dt) {
    for (let i = particles.length - 1; i >= 0; i -= 1) {
      const p = particles[i];
      p.life -= dt;
      p.x += p.vx * dt;
      p.y += p.vy * dt;
      p.vy += 560 * dt;
      if (p.life <= 0) {
        particles.splice(i, 1);
      }
    }

    for (let i = trail.length - 1; i >= 0; i -= 1) {
      trail[i].life -= dt;
      if (trail[i].life <= 0) {
        trail.splice(i, 1);
      }
    }
  }

  function checkCollisions() {
    const hitbox = {
      x: player.x + player.size * 0.18,
      y: player.y + player.size * 0.1,
      width: player.size * 0.64,
      height: player.size * 0.82
    };

    for (let i = 0; i < obstacles.length; i += 1) {
      if (intersects(hitbox, toRect(obstacles[i]))) {
        crash();
        return;
      }
    }
  }

  function update(dt) {
    for (let i = 0; i < matrixColumns.length; i += 1) {
      const col = matrixColumns[i];
      col.y += col.speed * dt;
      if (col.y - col.length * 20 > LOGICAL_HEIGHT + 60) {
        col.y = randomBetween(-350, -10);
        col.speed = randomBetween(150, 500);
        col.length = Math.floor(randomBetween(8, 24));
      }
    }

    if (gameState !== "playing") {
      updateParticles(dt);
      if (flashTimer > 0) {
        flashTimer -= dt;
      }
      return;
    }

    gameTime += dt;
    worldSpeed = Math.min(680, 420 + gameTime * 11.5);

    updatePlayer(dt);
    updateObstacles(dt);
    updateParticles(dt);
    checkCollisions();

    score = Math.floor(gameTime * 100 + worldSpeed * 0.05);
    scoreValue.textContent = String(score);
  }

  function drawBackground() {
    const gradient = ctx.createLinearGradient(0, 0, 0, LOGICAL_HEIGHT);
    gradient.addColorStop(0, "#071b12");
    gradient.addColorStop(0.48, "#04120b");
    gradient.addColorStop(1, "#030805");
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, LOGICAL_WIDTH, LOGICAL_HEIGHT);

    ctx.font = "16px 'Share Tech Mono', monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (let i = 0; i < matrixColumns.length; i += 1) {
      const col = matrixColumns[i];
      for (let j = 0; j < col.length; j += 1) {
        const y = col.y - j * 20;
        if (y < -30 || y > LOGICAL_HEIGHT + 30) {
          continue;
        }
        const alpha = Math.max(0, 1 - j / col.length);
        const char = MATRIX_CHARS[(Math.floor(y + i * 11 + gameTime * 100) + j) % MATRIX_CHARS.length];
        const isHead = j === 0;
        ctx.fillStyle = isHead
          ? `rgba(198, 255, 223, ${Math.min(0.95, alpha + 0.35)})`
          : `rgba(86, 255, 154, ${alpha * 0.68})`;
        ctx.fillText(char, col.x, y);
      }
    }

    ctx.strokeStyle = "rgba(104, 255, 173, 0.08)";
    ctx.lineWidth = 1;
    for (let i = 0; i < 10; i += 1) {
      const y = (LOGICAL_HEIGHT * 0.14) + i * 52 + ((gameTime * worldSpeed * 0.03) % 52);
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(LOGICAL_WIDTH, y);
      ctx.stroke();
    }
  }

  function drawFloor() {
    const y = floorY();
    ctx.fillStyle = "rgba(12, 30, 18, 0.95)";
    ctx.fillRect(0, y, LOGICAL_WIDTH, LOGICAL_HEIGHT - y);

    ctx.strokeStyle = "rgba(125, 255, 182, 0.85)";
    ctx.lineWidth = 2;
    ctx.shadowColor = "rgba(99, 255, 167, 0.8)";
    ctx.shadowBlur = 12;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(LOGICAL_WIDTH, y);
    ctx.stroke();
    ctx.shadowBlur = 0;

    const markerGap = 76;
    const offset = (gameTime * worldSpeed) % markerGap;
    for (let x = -markerGap; x < LOGICAL_WIDTH + markerGap; x += markerGap) {
      const px = x - offset;
      ctx.strokeStyle = "rgba(93, 255, 162, 0.32)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(px, y + 4);
      ctx.lineTo(px + 26, y + 4);
      ctx.stroke();
    }
  }

  function drawObstacle(obstacle) {
    const y = floorY();
    if (obstacle.type === "spike") {
      const left = obstacle.x;
      const right = obstacle.x + obstacle.width;
      const top = y - obstacle.height;
      ctx.beginPath();
      ctx.moveTo(left, y);
      ctx.lineTo((left + right) * 0.5, top);
      ctx.lineTo(right, y);
      ctx.closePath();

      const grad = ctx.createLinearGradient(left, top, right, y);
      grad.addColorStop(0, "rgba(46, 255, 149, 0.9)");
      grad.addColorStop(1, "rgba(16, 95, 56, 0.92)");
      ctx.fillStyle = grad;
      ctx.fill();
      ctx.strokeStyle = "rgba(207, 255, 227, 0.8)";
      ctx.lineWidth = 1.7;
      ctx.stroke();
      return;
    }

    const top = y - obstacle.height;
    const grad = ctx.createLinearGradient(obstacle.x, top, obstacle.x, y);
    grad.addColorStop(0, "rgba(81, 255, 169, 0.86)");
    grad.addColorStop(1, "rgba(19, 92, 58, 0.94)");
    ctx.fillStyle = grad;
    ctx.fillRect(obstacle.x, top, obstacle.width, obstacle.height);
    ctx.strokeStyle = "rgba(206, 255, 229, 0.6)";
    ctx.lineWidth = 1.6;
    ctx.strokeRect(obstacle.x + 0.8, top + 0.8, obstacle.width - 1.6, obstacle.height - 1.6);

    ctx.fillStyle = "rgba(2, 14, 8, 0.4)";
    for (let i = 0; i < 5; i += 1) {
      const py = top + 8 + i * ((obstacle.height - 14) / 5);
      ctx.fillRect(obstacle.x + 4, py, obstacle.width - 8, 2);
    }
  }

  function drawPlayer() {
    for (let i = 0; i < trail.length; i += 1) {
      const t = trail[i];
      ctx.fillStyle = `rgba(83, 255, 154, ${Math.max(0, t.life * 3.1)})`;
      ctx.beginPath();
      ctx.arc(t.x, t.y, 2 + t.life * 15, 0, Math.PI * 2);
      ctx.fill();
    }

    const centerX = player.x + player.size * 0.5;
    const centerY = player.y + player.size * 0.5;
    ctx.save();
    ctx.translate(centerX, centerY);
    ctx.rotate(player.rotation);

    const pulse = 0.75 + Math.sin(gameTime * 9) * 0.18;
    ctx.shadowColor = "rgba(72, 255, 152, 0.9)";
    ctx.shadowBlur = 22 * pulse;

    ctx.fillStyle = "#06120b";
    ctx.fillRect(-player.size * 0.5, -player.size * 0.5, player.size, player.size);
    ctx.lineWidth = 3;
    ctx.strokeStyle = "rgba(145, 255, 200, 0.95)";
    ctx.strokeRect(-player.size * 0.5, -player.size * 0.5, player.size, player.size);

    ctx.shadowBlur = 0;
    ctx.strokeStyle = "rgba(102, 255, 168, 0.75)";
    ctx.lineWidth = 2;
    const inner = player.size * 0.28;
    ctx.beginPath();
    ctx.moveTo(-inner, -inner);
    ctx.lineTo(inner, inner);
    ctx.moveTo(inner, -inner);
    ctx.lineTo(-inner, inner);
    ctx.stroke();
    ctx.restore();
  }

  function drawParticles() {
    for (let i = 0; i < particles.length; i += 1) {
      const p = particles[i];
      ctx.fillStyle = p.color;
      ctx.globalAlpha = Math.max(0, p.life * 1.45);
      ctx.fillRect(p.x, p.y, p.size, p.size);
    }
    ctx.globalAlpha = 1;
  }

  function drawScanlines() {
    ctx.fillStyle = "rgba(255, 255, 255, 0.035)";
    for (let y = 0; y < LOGICAL_HEIGHT; y += 4) {
      ctx.fillRect(0, y, LOGICAL_WIDTH, 1);
    }
  }

  function render() {
    drawBackground();
    drawFloor();

    for (let i = 0; i < obstacles.length; i += 1) {
      drawObstacle(obstacles[i]);
    }

    drawPlayer();
    drawParticles();
    drawScanlines();

    if (flashTimer > 0) {
      ctx.fillStyle = `rgba(255, 45, 82, ${Math.min(0.32, flashTimer * 1.8)})`;
      ctx.fillRect(0, 0, LOGICAL_WIDTH, LOGICAL_HEIGHT);
    }
  }

  function frame(now) {
    const dt = Math.min(0.033, (now - lastTime) / 1000);
    lastTime = now;

    update(dt);
    render();
    requestAnimationFrame(frame);
  }

  function onInput(evt) {
    evt.preventDefault();
    triggerJump();
  }

  window.addEventListener("resize", resizeCanvas);
  window.addEventListener("keydown", (evt) => {
    const code = evt.code;
    if (code === "Space" || code === "ArrowUp" || code === "KeyW") {
      evt.preventDefault();
      triggerJump();
    }
  }, { passive: false });

  canvas.addEventListener("pointerdown", onInput, { passive: false });
  jumpButton.addEventListener("pointerdown", onInput, { passive: false });

  initializeColumns();
  resizeCanvas();
  resetGame();
  requestAnimationFrame((t) => {
    lastTime = t;
    requestAnimationFrame(frame);
  });
})();
