# Clusters of the latent model (memory, layer 2, mlp encoder)

Belief explains 11% of the latent variance linearly. 8 k-means clusters on 400 test episodes x 8 turns.

## raw latent

NMI with each variable: turn 0.20, belief pattern 0.14, clue source 0.09, clue contradicts 0.05, clue phrasing 0.01, clue negations 0.01, clue attribute 0.01, clue template 0.00, clue level 0.00

### Cluster 0 (338 states)
- mean P(value 0): {'colour': 0.44, 'shape': 0.46, 'size': 0.36, 'material': 0.53}
- over-represented: turn = 8 (22% vs 12%); clue contradicts = yes (45% vs 36%); clue attribute = size (30% vs 25%)
- example: Bob says the object is wooden. | Alice says the object is not big.
- example: Alice says the object is red. | Alice says the object is metal.
- example: Bob says the object is not wooden. | Alice says the object is not round.

### Cluster 1 (953 states)
- mean P(value 0): {'colour': 0.44, 'shape': 0.51, 'size': 0.41, 'material': 0.58}
- over-represented: turn = 3 (20% vs 12%); clue source = Bob (61% vs 49%); clue phrasing = negated (60% vs 50%)
- example: Bob says the object is not square. | Bob says the object is not big.
- example: Alice says the object is not metal. | Bob says the object is round.
- example: Bob says the object is not metal. | Bob says the object is blue.

### Cluster 2 (636 states)
- mean P(value 0): {'colour': 0.4, 'shape': 0.27, 'size': 0.4, 'material': 0.49}
- over-represented: turn = 8 (21% vs 12%); clue attribute = shape (35% vs 25%); clue contradicts = yes (48% vs 36%)
- example: Bob says the object is not big. | Alice says the object is big.
- example: Bob says the object is not square. | Alice says the object is round.
- example: Alice says the object is square. | Bob says the object is not round.

### Cluster 3 (274 states)
- mean P(value 0): {'colour': 0.42, 'shape': 0.41, 'size': 0.44, 'material': 0.4}
- over-represented: turn = 6 (16% vs 12%); clue attribute = size (28% vs 25%); clue contradicts = yes (39% vs 36%)
- example: Bob says the object is not big. | Bob says the object is square.
- example: Alice says the object is metal. | Alice says the object is square.
- example: Bob says the object is not wooden. | Bob says the object is not red.

### Cluster 4 (224 states)
- mean P(value 0): {'colour': 0.34, 'shape': 0.49, 'size': 0.43, 'material': 0.56}
- over-represented: turn = 7 (20% vs 12%); clue contradicts = yes (46% vs 36%); clue attribute = colour (33% vs 26%)
- example: Alice says the object is not blue. | Alice says the object is not wooden.
- example: Alice says the object is not round. | Bob says the object is small.
- example: Alice says the object is not blue. | Alice says the object is round.

### Cluster 5 (332 states)
- mean P(value 0): {'colour': 0.44, 'shape': 0.51, 'size': 0.45, 'material': 0.62}
- over-represented: turn = 1 (70% vs 12%); clue source = Alice (100% vs 51%); clue contradicts = no (96% vs 64%)
- example: Alice says the object is not wooden.
- example: Alice says the object is blue.
- example: Alice says the object is blue. | Alice says the object is big.

### Cluster 6 (276 states)
- mean P(value 0): {'colour': 0.41, 'shape': 0.42, 'size': 0.37, 'material': 0.57}
- over-represented: turn = 6 (17% vs 12%); clue attribute = material (30% vs 24%); clue contradicts = yes (45% vs 36%)
- example: Bob says the object is not red. | Bob says the object is wooden.
- example: Alice says the object is round. | Bob says the object is not round.
- example: Alice says the object is not square. | Alice says the object is square.

### Cluster 7 (167 states)
- mean P(value 0): {'colour': 0.43, 'shape': 0.51, 'size': 0.4, 'material': 0.56}
- over-represented: turn = 1 (100% vs 12%); clue source = Bob (100% vs 49%); clue contradicts = no (100% vs 64%)
- example: Bob says the object is not red.
- example: Bob says the object is not big.
- example: Bob says the object is not blue.

## latent minus belief

NMI with each variable: turn 0.17, clue source 0.13, belief pattern 0.06, clue contradicts 0.05, clue attribute 0.01, clue phrasing 0.00, clue negations 0.00, clue template 0.00, clue level 0.00

### Cluster 0 (360 states)
- mean P(value 0): {'colour': 0.42, 'shape': 0.44, 'size': 0.39, 'material': 0.54}
- over-represented: turn = 8 (20% vs 12%); clue contradicts = yes (48% vs 36%); clue attribute = material (29% vs 24%)
- example: Bob says the object is metal. | Alice says the object is small.
- example: Alice says the object is not red. | Bob says the object is not blue.
- example: Bob says the object is round. | Bob says the object is not square.

### Cluster 1 (155 states)
- mean P(value 0): {'colour': 0.42, 'shape': 0.49, 'size': 0.4, 'material': 0.56}
- over-represented: turn = 1 (100% vs 12%); clue source = Bob (100% vs 49%); clue contradicts = no (100% vs 64%)
- example: Bob says the object is not blue.
- example: Bob says the object is blue.
- example: Bob says the object is metal.

### Cluster 2 (808 states)
- mean P(value 0): {'colour': 0.42, 'shape': 0.41, 'size': 0.41, 'material': 0.53}
- over-represented: turn = 2 (20% vs 12%); clue source = Bob (77% vs 49%); clue attribute = colour (30% vs 26%)
- example: Bob says the object is small. | Bob says the object is not round.
- example: Alice says the object is not round. | Alice says the object is metal.
- example: Bob says the object is red. | Alice says the object is round.

### Cluster 3 (231 states)
- mean P(value 0): {'colour': 0.41, 'shape': 0.47, 'size': 0.36, 'material': 0.52}
- over-represented: clue attribute = shape (39% vs 25%); clue phrasing = negated (56% vs 50%); clue negations = 1 (56% vs 50%)
- example: Bob says the object is not square. | Bob says the object is not wooden.
- example: Bob says the object is square. | Alice says the object is not metal.
- example: Alice says the object is round. | Alice says the object is square.

### Cluster 4 (333 states)
- mean P(value 0): {'colour': 0.43, 'shape': 0.44, 'size': 0.4, 'material': 0.55}
- over-represented: turn = 8 (24% vs 12%); clue contradicts = yes (49% vs 36%); clue attribute = size (34% vs 25%)
- example: Bob says the object is big. | Bob says the object is not small.
- example: Alice says the object is small. | Alice says the object is blue.
- example: Alice says the object is not small. | Alice says the object is not big.

### Cluster 5 (393 states)
- mean P(value 0): {'colour': 0.43, 'shape': 0.46, 'size': 0.43, 'material': 0.59}
- over-represented: turn = 1 (55% vs 12%); clue source = Alice (100% vs 51%); clue contradicts = no (92% vs 64%)
- example: Alice says the object is blue.
- example: Alice says the object is not big. | Alice says the object is big.
- example: Alice says the object is small. | Alice says the object is metal.

### Cluster 6 (542 states)
- mean P(value 0): {'colour': 0.43, 'shape': 0.46, 'size': 0.41, 'material': 0.54}
- over-represented: turn = 8 (21% vs 12%); clue source = Alice (70% vs 51%); clue contradicts = yes (49% vs 36%)
- example: Alice says the object is red. | Bob says the object is not red.
- example: Alice says the object is not square. | Alice says the object is red.
- example: Bob says the object is not big. | Alice says the object is metal.

### Cluster 7 (378 states)
- mean P(value 0): {'colour': 0.41, 'shape': 0.42, 'size': 0.41, 'material': 0.51}
- over-represented: turn = 3 (19% vs 12%); clue contradicts = no (70% vs 64%); clue attribute = colour (28% vs 26%)
- example: Bob says the object is wooden. | Alice says the object is not square.
- example: Alice says the object is blue. | Alice says the object is not big.
- example: Bob says the object is not big. | Bob says the object is small.
