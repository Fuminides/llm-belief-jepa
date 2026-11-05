# Clusters of the latent model (memory, layer 2, mlp encoder)

Belief explains 8% of the latent variance linearly. 8 k-means clusters on 400 test episodes x 8 turns.

## raw latent

NMI with each variable: belief pattern 0.18, turn 0.13, clue attribute 0.04, clue contradicts 0.04, clue template 0.02, clue negations 0.01, clue phrasing 0.01, clue level 0.00, clue source 0.00

### Cluster 0 (490 states)
- mean P(value 0): {'colour': 0.43, 'shape': 0.36, 'size': 0.34, 'material': 0.57}
- over-represented: clue template = 5: "If you ask {source}, the answer you will..." (16% vs 8%); turn = 8 (19% vs 12%); clue contradicts = yes (48% vs 36%)
- example: Bob, who had a good look at it earlier, told us that the object is not blue. | Alice says the object is not round.
- example: Bob, who had a good look at it earlier, told us that the object is metal. | When we spoke to Alice this morning, the reply was that the object is not metal.
- example: According to Bob, the object is not small. | Bob, who had a good look at it earlier, told us that the object is not blue.

### Cluster 1 (331 states)
- mean P(value 0): {'colour': 0.45, 'shape': 0.53, 'size': 0.71, 'material': 0.58}
- over-represented: clue attribute = size (57% vs 25%); turn = 3 (17% vs 12%); clue negations = 0 (47% vs 37%)
- example: According to Bob, the object is small. | Alice says the object is not small.
- example: Alice claims that the object is big.
- example: Bob, who had a good look at it earlier, told us that the object is not wooden. | Alice says it is false that the object is not red.

### Cluster 2 (485 states)
- mean P(value 0): {'colour': 0.42, 'shape': 0.46, 'size': 0.39, 'material': 0.57}
- over-represented: turn = 8 (26% vs 12%); clue template = 4: "{source}, who had a good look at it earl..." (14% vs 9%); clue contradicts = yes (52% vs 36%)
- example: Alice says the object is not small. | If you ask Bob, the answer you will get is that the object is not blue.
- example: Asked whether the object is not small, Bob said that this is false. | Alice says the object is red.
- example: Bob, who had a good look at it earlier, told us that the object is not square. | Bob, who had a good look at it earlier, told us that the object is wooden.

### Cluster 3 (258 states)
- mean P(value 0): {'colour': 0.45, 'shape': 0.54, 'size': 0.41, 'material': 0.58}
- over-represented: clue template = 6: "When we spoke to {source} this morning, ..." (19% vs 8%); turn = 7 (21% vs 12%); clue contradicts = yes (50% vs 36%)
- example: When we spoke to Alice this morning, the reply was that the object is not blue. | Alice says the object is not metal.
- example: When we spoke to Bob this morning, the reply was that the object is not small. | According to Bob, the object is blue.
- example: According to Alice, the object is big. | Bob claims that the object is big.

### Cluster 4 (374 states)
- mean P(value 0): {'colour': 0.21, 'shape': 0.47, 'size': 0.37, 'material': 0.56}
- over-represented: clue attribute = colour (47% vs 26%); turn = 5 (18% vs 12%); clue template = 6: "When we spoke to {source} this morning, ..." (10% vs 8%)
- example: Alice denies that the object is not blue. | Asked whether the object is not small, Alice said that this is false.
- example: Alice says it is false that the object is blue. | Bob says the object is square.
- example: Bob told us the object is not blue. | Alice says the object is not wooden.

### Cluster 5 (186 states)
- mean P(value 0): {'colour': 0.42, 'shape': 0.4, 'size': 0.45, 'material': 0.59}
- over-represented: clue template = 1: "According to {source}, the object {claim..." (19% vs 9%); turn = 8 (18% vs 12%); clue level = 1 (32% vs 26%)
- example: Bob says it is false that the object is round. | When we spoke to Alice this morning, the reply was that the object is small.
- example: Alice, who had a good look at it earlier, told us that the object is round. | Alice, who had a good look at it earlier, told us that the object is not big.
- example: Bob says the object is not wooden. | Alice claims that the object is not metal.

### Cluster 6 (731 states)
- mean P(value 0): {'colour': 0.46, 'shape': 0.52, 'size': 0.35, 'material': 0.62}
- over-represented: turn = 1 (40% vs 12%); clue template = 9: "Asked whether the object {anticlaim}, {s..." (11% vs 8%); clue contradicts = no (87% vs 64%)
- example: When we spoke to Bob this morning, the reply was that the object is not square. | Bob says the object is not square.
- example: Alice claims that the object is not wooden. | Asked whether the object is not metal, Alice said that this is false.
- example: Alice says the object is not wooden.

### Cluster 7 (345 states)
- mean P(value 0): {'colour': 0.43, 'shape': 0.44, 'size': 0.4, 'material': 0.3}
- over-represented: clue attribute = material (52% vs 24%); clue negations = 0 (55% vs 37%); clue phrasing = affirmative (70% vs 50%)
- example: Alice told us the object is square. | According to Alice, the object is big.
- example: Bob says it is false that the object is not wooden. | Alice says the object is not blue.
- example: Alice says the object is not big. | When we spoke to Bob this morning, the reply was that the object is metal.

## latent minus belief

NMI with each variable: turn 0.16, belief pattern 0.05, clue contradicts 0.05, clue template 0.02, clue attribute 0.00, clue source 0.00, clue level 0.00, clue negations 0.00, clue phrasing 0.00

### Cluster 0 (287 states)
- mean P(value 0): {'colour': 0.42, 'shape': 0.46, 'size': 0.4, 'material': 0.55}
- over-represented: clue template = 2: "{source} claims that the object {claim}...." (17% vs 9%); turn = 8 (20% vs 12%); clue contradicts = yes (49% vs 36%)
- example: Alice told us the object is not round. | According to Alice, the object is square.
- example: According to Bob, the object is not blue. | Bob denies that the object is red.
- example: Alice says the object is square. | Alice says the object is big.

### Cluster 1 (315 states)
- mean P(value 0): {'colour': 0.4, 'shape': 0.46, 'size': 0.39, 'material': 0.54}
- over-represented: turn = 8 (23% vs 12%); clue template = 4: "{source}, who had a good look at it earl..." (17% vs 9%); clue contradicts = yes (53% vs 36%)
- example: Alice told us the object is blue. | According to Bob, the object is not square.
- example: Bob says it is false that the object is not big. | Alice says the object is round.
- example: Alice told us the object is not square. | Bob says the object is not big.

### Cluster 2 (267 states)
- mean P(value 0): {'colour': 0.42, 'shape': 0.5, 'size': 0.42, 'material': 0.56}
- over-represented: clue template = 6: "When we spoke to {source} this morning, ..." (18% vs 8%); turn = 8 (22% vs 12%); clue contradicts = yes (48% vs 36%)
- example: Alice, who had a good look at it earlier, told us that the object is small. | Asked whether the object is square, Alice said that this is false.
- example: When we spoke to Bob this morning, the reply was that the object is not blue. | Alice says the object is wooden.
- example: Alice told us the object is blue. | According to Alice, the object is metal.

### Cluster 3 (441 states)
- mean P(value 0): {'colour': 0.4, 'shape': 0.46, 'size': 0.42, 'material': 0.55}
- over-represented: turn = 3 (18% vs 12%); clue template = 6: "When we spoke to {source} this morning, ..." (11% vs 8%); clue negations = 2 (15% vs 12%)
- example: According to Bob, the object is red. | Asked whether the object is square, Alice said that this is false.
- example: Bob told us the object is not metal. | If you ask Alice, the answer you will get is that the object is not round.
- example: Alice claims that the object is not big. | When we spoke to Alice this morning, the reply was that the object is not big.

### Cluster 4 (602 states)
- mean P(value 0): {'colour': 0.42, 'shape': 0.5, 'size': 0.42, 'material': 0.57}
- over-represented: turn = 1 (61% vs 12%); clue contradicts = no (94% vs 64%); clue attribute = material (30% vs 24%)
- example: If you ask Bob, the answer you will get is that the object is not square.
- example: Alice, who had a good look at it earlier, told us that the object is not small. | Alice says the object is small.
- example: Bob says it is false that the object is round.

### Cluster 5 (178 states)
- mean P(value 0): {'colour': 0.41, 'shape': 0.42, 'size': 0.48, 'material': 0.57}
- over-represented: clue template = 1: "According to {source}, the object {claim..." (21% vs 9%); clue level = 1 (31% vs 26%); turn = 2 (14% vs 12%)
- example: Alice says it is false that the object is not square. | Bob says the object is blue.
- example: Alice says it is false that the object is not round. | Alice says the object is square.
- example: According to Alice, the object is square.

### Cluster 6 (722 states)
- mean P(value 0): {'colour': 0.41, 'shape': 0.47, 'size': 0.4, 'material': 0.54}
- over-represented: turn = 3 (22% vs 12%); clue template = 5: "If you ask {source}, the answer you will..." (11% vs 8%); clue source = Alice (57% vs 51%)
- example: Alice told us the object is blue. | Bob says the object is metal.
- example: Alice says it is false that the object is not wooden. | Alice says the object is blue.
- example: Asked whether the object is square, Bob said that this is false. | Alice claims that the object is blue.

### Cluster 7 (388 states)
- mean P(value 0): {'colour': 0.42, 'shape': 0.46, 'size': 0.42, 'material': 0.55}
- over-represented: turn = 8 (24% vs 12%); clue template = 4: "{source}, who had a good look at it earl..." (13% vs 9%); clue contradicts = yes (49% vs 36%)
- example: Asked whether the object is not red, Bob said that this is false. | Bob says the object is not red.
- example: According to Alice, the object is red. | Bob denies that the object is not metal.
- example: Bob says the object is not small. | Bob says the object is not blue.
