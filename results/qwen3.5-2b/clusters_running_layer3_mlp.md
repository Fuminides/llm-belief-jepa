# Clusters of the latent model (running, layer 3, mlp encoder)

Belief explains 11% of the latent variance linearly. 8 k-means clusters on 400 test episodes x 8 turns.

## raw latent

NMI with each variable: turn 0.18, belief pattern 0.15, clue attribute 0.07, clue contradicts 0.04, clue phrasing 0.02, clue negations 0.02, clue source 0.00, clue template 0.00, clue level 0.00

### Cluster 0 (428 states)
- mean P(value 0): {'colour': 0.43, 'shape': 0.5, 'size': 0.4, 'material': 0.58}
- over-represented: clue attribute = colour (53% vs 26%); turn = 4 (23% vs 12%); clue phrasing = negated (61% vs 50%)
- example: Alice says the object is not blue. | Bob says the object is square.
- example: Alice says the object is small. | Bob says the object is not blue.
- example: Bob says the object is not wooden. | Bob says the object is round.

### Cluster 1 (239 states)
- mean P(value 0): {'colour': 0.29, 'shape': 0.43, 'size': 0.35, 'material': 0.53}
- over-represented: turn = 8 (30% vs 12%); clue attribute = colour (41% vs 26%); clue contradicts = yes (55% vs 36%)
- example: Alice says the object is not blue. | Bob says the object is not red.
- example: Alice says the object is not round. | Alice says the object is not big.
- example: Bob says the object is not square. | Alice says the object is blue.

### Cluster 2 (608 states)
- mean P(value 0): {'colour': 0.43, 'shape': 0.32, 'size': 0.38, 'material': 0.6}
- over-represented: clue attribute = shape (48% vs 25%); turn = 5 (17% vs 12%); clue contradicts = yes (41% vs 36%)
- example: Alice says the object is not square. | Bob says the object is not red.
- example: Alice says the object is not big. | Alice says the object is square.
- example: Alice says the object is square.

### Cluster 3 (276 states)
- mean P(value 0): {'colour': 0.39, 'shape': 0.43, 'size': 0.47, 'material': 0.55}
- over-represented: turn = 8 (33% vs 12%); clue attribute = size (36% vs 25%); clue contradicts = yes (51% vs 36%)
- example: Alice says the object is big. | Bob says the object is square.
- example: Alice says the object is not red. | Bob says the object is round.
- example: Bob says the object is blue. | Alice says the object is big.

### Cluster 4 (292 states)
- mean P(value 0): {'colour': 0.41, 'shape': 0.46, 'size': 0.4, 'material': 0.55}
- over-represented: turn = 2 (32% vs 12%); clue attribute = material (61% vs 24%); clue contradicts = no (81% vs 64%)
- example: Bob says the object is not red. | Bob says the object is not blue.
- example: Alice says the object is not wooden. | Bob says the object is not wooden.
- example: Alice says the object is not wooden. | Alice says the object is wooden.

### Cluster 5 (517 states)
- mean P(value 0): {'colour': 0.44, 'shape': 0.4, 'size': 0.41, 'material': 0.33}
- over-represented: clue attribute = material (40% vs 24%); clue phrasing = affirmative (70% vs 50%); clue negations = 0 (70% vs 50%)
- example: Alice says the object is blue. | Bob says the object is wooden.
- example: Bob says the object is square. | Alice says the object is round.
- example: Alice says the object is not big. | Bob says the object is metal.

### Cluster 6 (487 states)
- mean P(value 0): {'colour': 0.45, 'shape': 0.53, 'size': 0.43, 'material': 0.64}
- over-represented: turn = 1 (48% vs 12%); clue attribute = size (43% vs 25%); clue contradicts = no (90% vs 64%)
- example: Alice says the object is not round.
- example: Bob says the object is not big. | Alice says the object is red.
- example: Bob says the object is not red. | Alice says the object is round.

### Cluster 7 (353 states)
- mean P(value 0): {'colour': 0.45, 'shape': 0.51, 'size': 0.4, 'material': 0.55}
- over-represented: turn = 8 (30% vs 12%); clue contradicts = yes (54% vs 36%); clue attribute = material (28% vs 24%)
- example: Bob says the object is blue. | Bob says the object is not wooden.
- example: Bob says the object is small. | Bob says the object is round.
- example: Alice says the object is not square. | Alice says the object is not round.

## latent minus belief

NMI with each variable: turn 0.20, belief pattern 0.09, clue attribute 0.06, clue contradicts 0.05, clue phrasing 0.01, clue negations 0.01, clue source 0.00, clue template 0.00, clue level 0.00

### Cluster 0 (276 states)
- mean P(value 0): {'colour': 0.41, 'shape': 0.43, 'size': 0.45, 'material': 0.54}
- over-represented: turn = 8 (30% vs 12%); clue attribute = size (34% vs 25%); clue phrasing = affirmative (68% vs 50%)
- example: Alice says the object is wooden. | Alice says the object is square.
- example: Alice says the object is round. | Bob says the object is small.
- example: Alice says the object is blue. | Bob says the object is big.

### Cluster 1 (649 states)
- mean P(value 0): {'colour': 0.44, 'shape': 0.44, 'size': 0.43, 'material': 0.53}
- over-represented: clue attribute = size (46% vs 25%); turn = 5 (16% vs 12%); clue contradicts = yes (39% vs 36%)
- example: Alice says the object is metal. | Alice says the object is not square.
- example: Bob says the object is not big. | Alice says the object is not big.
- example: Bob says the object is not metal. | Alice says the object is not big.

### Cluster 2 (213 states)
- mean P(value 0): {'colour': 0.32, 'shape': 0.43, 'size': 0.37, 'material': 0.54}
- over-represented: turn = 8 (27% vs 12%); clue attribute = colour (44% vs 26%); clue contradicts = yes (55% vs 36%)
- example: Bob says the object is blue. | Alice says the object is not metal.
- example: Alice says the object is not blue. | Alice says the object is round.
- example: Bob says the object is not blue. | Alice says the object is wooden.

### Cluster 3 (418 states)
- mean P(value 0): {'colour': 0.42, 'shape': 0.42, 'size': 0.38, 'material': 0.54}
- over-represented: turn = 8 (30% vs 12%); clue attribute = shape (43% vs 25%); clue contradicts = yes (60% vs 36%)
- example: Bob says the object is not round. | Bob says the object is blue.
- example: Bob says the object is not big. | Alice says the object is metal.
- example: Alice says the object is red. | Bob says the object is round.

### Cluster 4 (422 states)
- mean P(value 0): {'colour': 0.41, 'shape': 0.44, 'size': 0.4, 'material': 0.55}
- over-represented: turn = 4 (27% vs 12%); clue attribute = colour (53% vs 26%); clue contradicts = no (73% vs 64%)
- example: Bob says the object is blue. | Alice says the object is not metal.
- example: Alice says the object is blue. | Alice says the object is not big.
- example: Bob says the object is metal. | Alice says the object is small.

### Cluster 5 (271 states)
- mean P(value 0): {'colour': 0.43, 'shape': 0.44, 'size': 0.41, 'material': 0.53}
- over-represented: turn = 8 (25% vs 12%); clue attribute = material (35% vs 24%); clue contradicts = yes (48% vs 36%)
- example: Alice says the object is wooden. | Alice says the object is not metal.
- example: Bob says the object is not wooden. | Alice says the object is not wooden.
- example: Bob says the object is not wooden. | Bob says the object is not round.

### Cluster 6 (376 states)
- mean P(value 0): {'colour': 0.42, 'shape': 0.41, 'size': 0.38, 'material': 0.56}
- over-represented: turn = 3 (26% vs 12%); clue attribute = shape (53% vs 25%); clue contradicts = no (73% vs 64%)
- example: Bob says the object is not wooden. | Alice says the object is blue.
- example: Bob says the object is not small. | Bob says the object is not square.
- example: Bob says the object is big. | Alice says the object is not square.

### Cluster 7 (575 states)
- mean P(value 0): {'colour': 0.45, 'shape': 0.49, 'size': 0.4, 'material': 0.55}
- over-represented: turn = 1 (45% vs 12%); clue attribute = material (37% vs 24%); clue contradicts = no (91% vs 64%)
- example: Bob says the object is not wooden.
- example: Bob says the object is not red. | Bob says the object is not wooden.
- example: Alice says the object is not round.
