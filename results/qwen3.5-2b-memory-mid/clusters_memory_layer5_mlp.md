# Clusters of the latent model (memory, layer 5, mlp encoder)

Belief explains 9% of the latent variance linearly. 8 k-means clusters on 400 test episodes x 8 turns.

## raw latent

NMI with each variable: turn 0.28, clue source 0.13, belief pattern 0.06, clue contradicts 0.06, clue attribute 0.01, clue phrasing 0.01, clue negations 0.01, clue template 0.00, clue level 0.00

### Cluster 0 (947 states)
- mean P(value 0): {'colour': 0.42, 'shape': 0.45, 'size': 0.4, 'material': 0.55}
- over-represented: turn = 3 (24% vs 12%); clue source = Bob (53% vs 49%); clue contradicts = no (68% vs 64%)
- example: Alice says the object is blue. | Bob says the object is metal.
- example: Alice says the object is not square. | Bob says the object is wooden.
- example: Alice says the object is not wooden. | Bob says the object is small.

### Cluster 1 (355 states)
- mean P(value 0): {'colour': 0.41, 'shape': 0.42, 'size': 0.45, 'material': 0.5}
- over-represented: turn = 8 (34% vs 12%); clue contradicts = yes (54% vs 36%); clue source = Alice (66% vs 51%)
- example: Bob says the object is not big. | Alice says the object is not wooden.
- example: Alice says the object is small. | Alice says the object is wooden.
- example: Bob says the object is red. | Alice says the object is not red.

### Cluster 2 (407 states)
- mean P(value 0): {'colour': 0.43, 'shape': 0.52, 'size': 0.42, 'material': 0.59}
- over-represented: turn = 1 (98% vs 12%); clue contradicts = no (99% vs 64%); clue source = Alice (58% vs 51%)
- example: Bob says the object is not big.
- example: Alice says the object is not big.
- example: Bob says the object is metal.

### Cluster 3 (448 states)
- mean P(value 0): {'colour': 0.43, 'shape': 0.43, 'size': 0.38, 'material': 0.51}
- over-represented: turn = 8 (32% vs 12%); clue contradicts = yes (60% vs 36%); clue source = Bob (67% vs 49%)
- example: Bob says the object is not blue. | Bob says the object is big.
- example: Alice says the object is not round. | Alice says the object is not big.
- example: Bob says the object is not round. | Bob says the object is not metal.

### Cluster 4 (224 states)
- mean P(value 0): {'colour': 0.43, 'shape': 0.38, 'size': 0.34, 'material': 0.49}
- over-represented: turn = 8 (23% vs 12%); clue attribute = material (33% vs 24%); clue contradicts = yes (46% vs 36%)
- example: Alice says the object is not small. | Bob says the object is not blue.
- example: Alice says the object is not blue. | Alice says the object is not small.
- example: Bob says the object is not blue. | Bob says the object is small.

### Cluster 5 (105 states)
- mean P(value 0): {'colour': 0.44, 'shape': 0.47, 'size': 0.41, 'material': 0.61}
- over-represented: turn = 2 (15% vs 12%); clue attribute = material (27% vs 24%); clue contradicts = no (70% vs 64%)
- example: Bob says the object is not red. | Bob says the object is not wooden.
- example: Alice says the object is not red. | Bob says the object is square.
- example: Alice says the object is not metal. | Alice says the object is round.

### Cluster 6 (333 states)
- mean P(value 0): {'colour': 0.4, 'shape': 0.42, 'size': 0.41, 'material': 0.52}
- over-represented: turn = 3 (25% vs 12%); clue source = Bob (100% vs 49%); clue attribute = colour (33% vs 26%)
- example: Bob says the object is not big. | Bob says the object is not wooden.
- example: Bob says the object is not square. | Bob says the object is metal.
- example: Bob says the object is not red. | Bob says the object is not blue.

### Cluster 7 (381 states)
- mean P(value 0): {'colour': 0.39, 'shape': 0.41, 'size': 0.4, 'material': 0.56}
- over-represented: clue source = Alice (99% vs 51%); turn = 2 (21% vs 12%); clue attribute = shape (31% vs 25%)
- example: Alice says the object is not big. | Alice says the object is not red.
- example: Alice says the object is not square. | Alice says the object is metal.
- example: Alice says the object is not red. | Alice says the object is not small.

## latent minus belief

NMI with each variable: turn 0.29, clue source 0.16, belief pattern 0.06, clue contradicts 0.06, clue phrasing 0.01, clue negations 0.01, clue attribute 0.00, clue template 0.00, clue level 0.00

### Cluster 0 (412 states)
- mean P(value 0): {'colour': 0.43, 'shape': 0.51, 'size': 0.41, 'material': 0.59}
- over-represented: turn = 1 (97% vs 12%); clue contradicts = no (99% vs 64%); clue source = Alice (58% vs 51%)
- example: Alice says the object is red.
- example: Bob says the object is not metal.
- example: Bob says the object is wooden.

### Cluster 1 (204 states)
- mean P(value 0): {'colour': 0.43, 'shape': 0.39, 'size': 0.36, 'material': 0.51}
- over-represented: turn = 7 (24% vs 12%); clue attribute = material (32% vs 24%); clue contradicts = yes (45% vs 36%)
- example: Alice says the object is blue. | Bob says the object is not small.
- example: Alice says the object is not blue. | Alice says the object is round.
- example: Alice says the object is metal. | Bob says the object is blue.

### Cluster 2 (383 states)
- mean P(value 0): {'colour': 0.41, 'shape': 0.43, 'size': 0.43, 'material': 0.52}
- over-represented: turn = 8 (33% vs 12%); clue contradicts = yes (54% vs 36%); clue attribute = size (28% vs 25%)
- example: Bob says the object is round. | Bob says the object is red.
- example: Bob says the object is metal. | Alice says the object is not round.
- example: Alice says the object is big. | Alice says the object is round.

### Cluster 3 (316 states)
- mean P(value 0): {'colour': 0.41, 'shape': 0.42, 'size': 0.41, 'material': 0.56}
- over-represented: turn = 2 (34% vs 12%); clue source = Alice (98% vs 51%); clue attribute = shape (27% vs 25%)
- example: Alice says the object is square. | Alice says the object is blue.
- example: Alice says the object is not big. | Alice says the object is wooden.
- example: Alice says the object is not big. | Alice says the object is not round.

### Cluster 4 (448 states)
- mean P(value 0): {'colour': 0.42, 'shape': 0.42, 'size': 0.38, 'material': 0.52}
- over-represented: turn = 8 (33% vs 12%); clue contradicts = yes (58% vs 36%); clue source = Bob (65% vs 49%)
- example: Alice says the object is not small. | Bob says the object is not round.
- example: Alice says the object is not wooden. | Bob says the object is not red.
- example: Alice says the object is wooden. | Alice says the object is not wooden.

### Cluster 5 (100 states)
- mean P(value 0): {'colour': 0.44, 'shape': 0.47, 'size': 0.4, 'material': 0.6}
- over-represented: turn = 3 (16% vs 12%); clue attribute = material (28% vs 24%); clue contradicts = no (69% vs 64%)
- example: Alice says the object is square. | Alice says the object is square.
- example: Bob says the object is round. | Bob says the object is round.
- example: Alice says the object is not wooden. | Alice says the object is not square.

### Cluster 6 (755 states)
- mean P(value 0): {'colour': 0.42, 'shape': 0.43, 'size': 0.4, 'material': 0.53}
- over-represented: turn = 4 (27% vs 12%); clue source = Alice (71% vs 51%); clue phrasing = affirmative (56% vs 50%)
- example: Alice says the object is metal. | Alice says the object is round.
- example: Bob says the object is round. | Alice says the object is not red.
- example: Bob says the object is not red. | Alice says the object is small.

### Cluster 7 (582 states)
- mean P(value 0): {'colour': 0.41, 'shape': 0.44, 'size': 0.41, 'material': 0.54}
- over-represented: turn = 2 (34% vs 12%); clue source = Bob (100% vs 49%); clue attribute = colour (30% vs 26%)
- example: Alice says the object is not blue. | Bob says the object is round.
- example: Bob says the object is not red. | Bob says the object is not small.
- example: Bob says the object is round. | Bob says the object is not small.
