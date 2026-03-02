QUnit.module('Matrix Dash', function(hooks) {
  hooks.beforeEach(function() {
    // Reset the game state before each test
    MatrixDash.resetGame();
  });

  QUnit.test('Initial game state', function(assert) {
    assert.equal(MatrixDash.gameState, 'ready', 'Game should be in "ready" state initially');
    assert.ok(MatrixDash.player.grounded, 'Player should be on the ground');
    assert.equal(MatrixDash.obstacles.length, 0, 'There should be no obstacles at the start');
  });

  QUnit.test('Player Jump', function(assert) {
    // Start the game and trigger a jump
    MatrixDash.triggerJump();
    
    assert.equal(MatrixDash.gameState, 'playing', 'Game should be in "playing" state after first jump');
    assert.notOk(MatrixDash.player.grounded, 'Player should not be grounded after jumping');
    assert.ok(MatrixDash.player.vy < 0, 'Player should have a negative vertical velocity (moving up)');
  });

  QUnit.test('Player movement and gravity', function(assert) {
    MatrixDash.triggerJump(); // Start playing and jump
    const initialY = MatrixDash.player.y;
    
    // Simulate some time passing
    MatrixDash.updatePlayer(0.1);
    
    assert.ok(MatrixDash.player.y < initialY, 'Player should have moved up after a short time');
    
    // Simulate more time for gravity to take effect
    MatrixDash.updatePlayer(0.5);
    
    assert.ok(MatrixDash.player.vy > 0, 'Player vertical velocity should be positive (moving down)');
  });
  
  QUnit.test('Obstacle Spawning', function(assert) {
    MatrixDash.spawnObstacle('spike', 100);
    
    assert.equal(MatrixDash.obstacles.length, 1, 'Should have one obstacle after spawning');
    const obstacle = MatrixDash.obstacles[0];
    assert.equal(obstacle.type, 'spike', 'Obstacle should be of type "spike"');
    assert.equal(obstacle.x, MatrixDash.LOGICAL_WIDTH + 100, 'Obstacle should be spawned at the correct x position');
  });

  QUnit.test('Collision Detection - Intersects', function(assert) {
    const rectA = { x: 10, y: 10, width: 20, height: 20 };
    const rectB = { x: 15, y: 15, width: 20, height: 20 };
    const rectC = { x: 40, y: 40, width: 10, height: 10 };

    assert.ok(MatrixDash.intersects(rectA, rectB), 'Rects A and B should intersect');
    assert.notOk(MatrixDash.intersects(rectA, rectC), 'Rects A and C should not intersect');
  });

  QUnit.test('Player crash', function(assert) {
    MatrixDash.triggerJump(); // Start the game
    
    // Manually place the player and an obstacle to force a collision
    const player = MatrixDash.player;
    player.x = 100;
    player.y = MatrixDash.floorY() - player.size;
    player.grounded = true;

    MatrixDash.obstacles.push({
      type: 'spike',
      x: player.x,
      width: 52,
      height: 72
    });

    MatrixDash.checkCollisions();

    assert.equal(MatrixDash.gameState, 'dead', 'Game should be in "dead" state after a collision');
  });

  QUnit.test('Game reset after death', function(assert) {
    MatrixDash.triggerJump(); // Start
    MatrixDash.crash(); // Crash
    assert.equal(MatrixDash.gameState, 'dead', 'Game is in "dead" state');
    
    MatrixDash.triggerJump(); // Try to restart
    
    assert.equal(MatrixDash.gameState, 'playing', 'Game should be "playing" after jumping from the dead state');
    assert.ok(MatrixDash.player.vy < 0, 'Player should be jumping after restart');
  });
});
