# Clusters of the latent model (pool, layer 3, attention encoder)

Belief explains 8% of the latent variance linearly. 8 k-means clusters on 400 test episodes x 8 turns.

## raw latent

NMI with each variable: clue attribute 0.22, clue negations 0.18, belief pattern 0.17, turn 0.15, clue phrasing 0.13, clue level 0.04, clue template 0.03, clue contradicts 0.03, clue source 0.00

### Cluster 0 (195 states)
- mean P(value 0): {'colour': 0.32, 'shape': 0.31, 'size': 0.44, 'material': 0.54}
- over-represented: clue negations = 0 (97% vs 37%); clue attribute = shape (58% vs 25%); clue phrasing = affirmative (97% vs 50%)
- example: Bob claims that the object is not big. | According to Alice, the object is square.
- example: Asked whether the object is big, Alice said that this is false. | Alice told us the object is blue.
- example: When we spoke to Alice this morning, the reply was that the object is not small. | Alice told us the object is blue.

### Cluster 1 (382 states)
- mean P(value 0): {'colour': 0.42, 'shape': 0.45, 'size': 0.41, 'material': 0.44}
- over-represented: clue attribute = material (98% vs 24%); turn = 6 (17% vs 12%); clue template = 5: "If you ask {source}, the answer you will..." (11% vs 8%)
- example: Alice denies that the object is wooden. | If you ask Alice, the answer you will get is that the object is metal.
- example: Alice says the object is not big. | Alice claims that the object is not wooden.
- example: Alice says the object is not metal. | Bob claims that the object is not wooden.

### Cluster 2 (1016 states)
- mean P(value 0): {'colour': 0.41, 'shape': 0.47, 'size': 0.41, 'material': 0.58}
- over-represented: turn = 1 (29% vs 12%); clue template = 9: "Asked whether the object {anticlaim}, {s..." (12% vs 8%); clue negations = 2 (17% vs 12%)
- example: Bob says it is false that the object is big. | When we spoke to Bob this morning, the reply was that the object is not metal.
- example: Bob says it is false that the object is wooden. | Alice says the object is not red.
- example: Alice says the object is not big.

### Cluster 3 (305 states)
- mean P(value 0): {'colour': 0.4, 'shape': 0.47, 'size': 0.17, 'material': 0.59}
- over-represented: clue attribute = size (47% vs 25%); turn = 4 (17% vs 12%); clue negations = 0 (50% vs 37%)
- example: Bob says the object is metal. | Alice says the object is small.
- example: If you ask Alice, the answer you will get is that the object is not round. | When we spoke to Alice this morning, the reply was that the object is red.
- example: When we spoke to Bob this morning, the reply was that the object is not metal. | If you ask Bob, the answer you will get is that the object is small.

### Cluster 4 (370 states)
- mean P(value 0): {'colour': 0.41, 'shape': 0.55, 'size': 0.36, 'material': 0.58}
- over-represented: clue attribute = size (51% vs 25%); turn = 6 (22% vs 12%); clue phrasing = negated (82% vs 50%)
- example: If you ask Alice, the answer you will get is that the object is not big. | Bob says the object is not big.
- example: If you ask Bob, the answer you will get is that the object is square. | Alice, who had a good look at it earlier, told us that the object is not big.
- example: Bob says the object is not blue. | Bob says the object is not square.

### Cluster 5 (133 states)
- mean P(value 0): {'colour': 0.4, 'shape': 0.71, 'size': 0.42, 'material': 0.59}
- over-represented: clue attribute = shape (100% vs 25%); clue negations = 0 (98% vs 37%); clue phrasing = affirmative (98% vs 50%)
- example: When we spoke to Bob this morning, the reply was that the object is not small. | Alice, who had a good look at it earlier, told us that the object is round.
- example: Alice says it is false that the object is wooden. | If you ask Alice, the answer you will get is that the object is round.
- example: According to Bob, the object is square. | If you ask Alice, the answer you will get is that the object is round.

### Cluster 6 (521 states)
- mean P(value 0): {'colour': 0.38, 'shape': 0.41, 'size': 0.45, 'material': 0.55}
- over-represented: turn = 8 (27% vs 12%); clue negations = 2 (21% vs 12%); clue template = 9: "Asked whether the object {anticlaim}, {s..." (13% vs 8%)
- example: Bob says the object is not metal. | Alice says the object is not metal.
- example: Alice says the object is not small. | When we spoke to Alice this morning, the reply was that the object is not small.
- example: When we spoke to Alice this morning, the reply was that the object is not blue. | Bob, who had a good look at it earlier, told us that the object is not metal.

### Cluster 7 (278 states)
- mean P(value 0): {'colour': 0.55, 'shape': 0.49, 'size': 0.65, 'material': 0.55}
- over-represented: clue negations = 0 (99% vs 37%); clue attribute = size (53% vs 25%); clue phrasing = affirmative (100% vs 50%)
- example: Alice says the object is small. | According to Alice, the object is big.
- example: Bob says the object is metal. | Alice, who had a good look at it earlier, told us that the object is big.
- example: Alice denies that the object is not wooden. | Alice claims that the object is big.

## latent minus belief

NMI with each variable: clue attribute 0.40, turn 0.16, belief pattern 0.07, clue negations 0.04, clue contradicts 0.04, clue phrasing 0.03, clue template 0.01, clue level 0.01, clue source 0.00

### Cluster 0 (264 states)
- mean P(value 0): {'colour': 0.41, 'shape': 0.49, 'size': 0.4, 'material': 0.57}
- over-represented: clue attribute = shape (100% vs 25%); turn = 7 (19% vs 12%); clue template = 5: "If you ask {source}, the answer you will..." (11% vs 8%)
- example: When we spoke to Alice this morning, the reply was that the object is not square. | If you ask Bob, the answer you will get is that the object is round.
- example: Alice denies that the object is not big. | According to Bob, the object is not round.
- example: Bob says the object is not small. | Asked whether the object is not round, Alice said that this is false.

### Cluster 1 (108 states)
- mean P(value 0): {'colour': 0.61, 'shape': 0.47, 'size': 0.45, 'material': 0.54}
- over-represented: clue attribute = colour (100% vs 26%); clue negations = 0 (98% vs 37%); clue template = 3: "{source} told us the object {claim}...." (19% vs 8%)
- example: Alice says the object is square. | According to Alice, the object is red.
- example: According to Alice, the object is not red. | If you ask Bob, the answer you will get is that the object is red.
- example: According to Bob, the object is not big. | Bob says the object is red.

### Cluster 2 (548 states)
- mean P(value 0): {'colour': 0.4, 'shape': 0.48, 'size': 0.39, 'material': 0.55}
- over-represented: clue attribute = colour (51% vs 26%); turn = 6 (21% vs 12%); clue negations = 2 (16% vs 12%)
- example: Alice says it is false that the object is small. | Bob, who had a good look at it earlier, told us that the object is not blue.
- example: Alice says the object is metal. | Alice claims that the object is small.
- example: Bob told us the object is metal. | When we spoke to Bob this morning, the reply was that the object is not small.

### Cluster 3 (263 states)
- mean P(value 0): {'colour': 0.42, 'shape': 0.46, 'size': 0.4, 'material': 0.56}
- over-represented: clue attribute = material (99% vs 24%); turn = 6 (21% vs 12%); clue template = 5: "If you ask {source}, the answer you will..." (11% vs 8%)
- example: Bob says it is false that the object is big. | Bob says the object is not wooden.
- example: Alice says the object is square. | Alice says the object is wooden.
- example: According to Bob, the object is not red. | Bob, who had a good look at it earlier, told us that the object is not wooden.

### Cluster 4 (283 states)
- mean P(value 0): {'colour': 0.39, 'shape': 0.39, 'size': 0.42, 'material': 0.57}
- over-represented: clue attribute = shape (100% vs 25%); turn = 7 (21% vs 12%); clue template = 3: "{source} told us the object {claim}...." (12% vs 8%)
- example: Bob claims that the object is blue. | Alice told us the object is not square.
- example: Alice told us the object is wooden. | Bob says the object is not square.
- example: Alice says it is false that the object is blue. | Bob says the object is square.

### Cluster 5 (498 states)
- mean P(value 0): {'colour': 0.4, 'shape': 0.46, 'size': 0.39, 'material': 0.54}
- over-represented: clue attribute = material (54% vs 24%); turn = 8 (18% vs 12%); clue template = 9: "Asked whether the object {anticlaim}, {s..." (12% vs 8%)
- example: Alice says the object is round. | Bob says the object is not red.
- example: Bob claims that the object is not round. | According to Alice, the object is not red.
- example: When we spoke to Alice this morning, the reply was that the object is not blue. | When we spoke to Alice this morning, the reply was that the object is not red.

### Cluster 6 (310 states)
- mean P(value 0): {'colour': 0.41, 'shape': 0.47, 'size': 0.46, 'material': 0.55}
- over-represented: clue attribute = size (99% vs 25%); turn = 5 (20% vs 12%); clue template = 1: "According to {source}, the object {claim..." (12% vs 9%)
- example: Alice says the object is big. | Alice claims that the object is not big.
- example: If you ask Alice, the answer you will get is that the object is wooden. | Bob, who had a good look at it earlier, told us that the object is not big.
- example: Alice, who had a good look at it earlier, told us that the object is not small. | According to Bob, the object is big.

### Cluster 7 (926 states)
- mean P(value 0): {'colour': 0.41, 'shape': 0.48, 'size': 0.42, 'material': 0.55}
- over-represented: turn = 1 (36% vs 12%); clue template = 9: "Asked whether the object {anticlaim}, {s..." (12% vs 8%); clue contradicts = no (87% vs 64%)
- example: Alice says the object is red. | Alice says it is false that the object is square.
- example: Alice denies that the object is not metal. | Alice denies that the object is big.
- example: Asked whether the object is not round, Bob said that this is false. | Bob says the object is not small.
