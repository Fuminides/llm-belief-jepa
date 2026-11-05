# Clusters of the free numbers (anchored JEPA, layer 14, k = 16, seed 0)

8 k-means clusters on 3200 held-out states (after clues 1 to 8).

NMI with each variable: turn 0.16, belief pattern 0.13, contradicted attributes 0.10, clue attribute 0.06, clue contradicts 0.05, clue source 0.04, clue phrasing 0.04, clue negations 0.04

## Cluster 0 (94 states; turns {'1': 25, '2': 16, '3': 10, '4': 13, '5': 12, '6': 6, '7': 7, '8': 5})
- mean P(value 0): {'colour': 0.4000000059604645, 'shape': 0.47999998927116394, 'size': 0.699999988079071, 'material': 0.550000011920929}
- over-represented: belief pattern = 50 (1% vs 0%); clue attribute = size (87% vs 25%); turn = 1 (27% vs 12%); clue phrasing = affirmative (91% vs 50%)
- **Qwen/Qwen3.5-9B says:** The histories in Group A are characterized by a high frequency of **repetitive clues** (the same statement appearing multiple times in a single history) and **direct contradictions** between Alice and Bob regarding the same property, whereas Group B contains no such repetitions or contradictions. Additionally, Group A histories often feature long sequences dominated by a single speaker making the same claim, while Group B histories show more balanced alternation without these specific patterns.
- inside: Bob says the object is red. | Bob says the object is not square. | Bob says the object is not square. | Alice says the object is not wooden. | Bob says the object is blue. | Bob says the object is big.
- inside: Alice says the object is not square. | Bob says the object is wooden. | Alice says the object is square. | Alice says the object is square. | Alice says the object is big.
- inside: Alice says the object is red. | Bob says the object is not blue. | Alice says the object is blue. | Alice says the object is round. | Bob says the object is big. | Bob says the object is not wooden. | Alice says the object is big.
- inside: Alice says the object is not red. | Alice says the object is not blue. | Alice says the object is small. | Alice says the object is big. | Bob says the object is big.
- inside: Bob says the object is not red. | Alice says the object is small. | Alice says the object is not round.
- contrast (same number of clues, other clusters): Bob says the object is metal. | Alice says the object is not red. | Bob says the object is not round. | Bob says the object is not blue. | Alice says the object is wooden. | Alice says the object is not wooden.
- contrast (same number of clues, other clusters): Bob says the object is not blue. | Bob says the object is not square. | Bob says the object is metal. | Bob says the object is not small. | Bob says the object is blue.
- contrast (same number of clues, other clusters): Bob says the object is not metal. | Alice says the object is not round. | Bob says the object is big. | Alice says the object is blue. | Alice says the object is not big. | Alice says the object is square. | Alice says the object is red.

## Cluster 1 (680 states; turns {'2': 19, '3': 67, '4': 104, '5': 116, '6': 118, '7': 126, '8': 130})
- mean P(value 0): {'colour': 0.4300000071525574, 'shape': 0.4699999988079071, 'size': 0.4000000059604645, 'material': 0.550000011920929}
- over-represented: turn = 8 (19% vs 12%); contradicted attributes = 2 (26% vs 17%); clue contradicts = yes (50% vs 36%); clue phrasing = negated (69% vs 50%)
- **Qwen/Qwen3.5-9B says:** The histories in Group A are characterized by a strict alternating turn-taking pattern between Alice and Bob, whereas the histories in Group B frequently feature consecutive clues from the same source. Additionally, Group A contains significantly more instances of direct contradictions (e.g., a source stating a property is true and then false) compared to Group B. Finally, Group A includes several sequences where a single source provides multiple clues in a row, which is absent in the strictly alternating Group B.
- inside: Alice says the object is not red. | Alice says the object is not blue. | Bob says the object is not red. | Alice says the object is not square. | Bob says the object is not red. | Bob says the object is round. | Alice says the object is not big. | Bob says the object is not big.
- inside: Bob says the object is not blue. | Bob says the object is blue. | Alice says the object is red. | Bob says the object is not metal. | Bob says the object is blue. | Bob says the object is not big. | Alice says the object is not big.
- inside: Alice says the object is red. | Bob says the object is not round. | Bob says the object is not red. | Bob says the object is not square.
- inside: Alice says the object is not big. | Alice says the object is wooden. | Alice says the object is small. | Alice says the object is not round. | Bob says the object is round.
- inside: Bob says the object is not red. | Bob says the object is not red. | Alice says the object is square. | Alice says the object is blue.
- contrast (same number of clues, other clusters): Bob says the object is blue. | Bob says the object is not wooden. | Alice says the object is not red. | Bob says the object is small. | Bob says the object is small. | Alice says the object is red. | Bob says the object is not red. | Bob says the object is not round.
- contrast (same number of clues, other clusters): Alice says the object is small. | Alice says the object is metal. | Bob says the object is not blue. | Alice says the object is not blue. | Alice says the object is not red. | Bob says the object is metal. | Alice says the object is not square.
- contrast (same number of clues, other clusters): Alice says the object is red. | Bob says the object is not blue. | Alice says the object is blue. | Alice says the object is round.

## Cluster 2 (700 states; turns {'2': 20, '3': 61, '4': 109, '5': 129, '6': 128, '7': 130, '8': 123})
- mean P(value 0): {'colour': 0.4099999964237213, 'shape': 0.41999998688697815, 'size': 0.3799999952316284, 'material': 0.550000011920929}
- over-represented: contradicted attributes = 2 (27% vs 17%); turn = 7 (19% vs 12%); clue contradicts = yes (49% vs 36%); clue attribute = size (29% vs 25%)
- **Qwen/Qwen3.5-9B says:** The histories in Group A are characterized by a strict alternating turn-taking pattern between Alice and Bob, whereas the histories in Group B frequently feature consecutive clues from the same source. Additionally, Group A contains significantly more instances of direct contradictions between the two sources regarding the same property, while Group B shows a higher frequency of redundant or repetitive statements from a single speaker.
- inside: Alice says the object is small. | Alice says the object is not round. | Alice says the object is not big. | Alice says the object is not square. | Bob says the object is red. | Alice says the object is wooden. | Bob says the object is not wooden. | Bob says the object is not big.
- inside: Alice says the object is not round. | Bob says the object is not square. | Alice says the object is square. | Bob says the object is big. | Alice says the object is not metal. | Bob says the object is not square. | Bob says the object is square.
- inside: Bob says the object is wooden. | Bob says the object is not small. | Bob says the object is not metal. | Alice says the object is wooden. | Bob says the object is wooden. | Bob says the object is not round. | Alice says the object is blue. | Bob says the object is not big.
- inside: Alice says the object is big. | Bob says the object is not big. | Alice says the object is not small. | Alice says the object is not red. | Alice says the object is not small. | Alice says the object is wooden.
- inside: Alice says the object is not round. | Bob says the object is not square. | Alice says the object is blue.
- contrast (same number of clues, other clusters): Alice says the object is wooden. | Alice says the object is not blue. | Alice says the object is not small. | Alice says the object is not blue. | Bob says the object is square. | Alice says the object is not metal. | Bob says the object is red. | Bob says the object is wooden.
- contrast (same number of clues, other clusters): Alice says the object is blue. | Bob says the object is round. | Alice says the object is not small. | Alice says the object is not blue. | Alice says the object is not wooden. | Bob says the object is not small. | Alice says the object is round.
- contrast (same number of clues, other clusters): Bob says the object is big. | Alice says the object is small. | Alice says the object is square. | Alice says the object is not red. | Alice says the object is not small. | Alice says the object is round. | Alice says the object is not small. | Bob says the object is blue.

## Cluster 3 (568 states; turns {'2': 44, '3': 65, '4': 75, '5': 81, '6': 101, '7': 101, '8': 101})
- mean P(value 0): {'colour': 0.4099999964237213, 'shape': 0.44999998807907104, 'size': 0.4099999964237213, 'material': 0.4399999976158142}
- over-represented: turn = 6 (18% vs 12%); clue phrasing = affirmative (67% vs 50%); clue negations = 0 (67% vs 50%); clue contradicts = yes (42% vs 36%)
- **Qwen/Qwen3.5-9B says:** The histories in Group A are characterized by a high frequency of direct contradictions between clues (e.g., a source stating a property is true and later false, or two different sources giving opposite values for the same property within the same sequence), whereas Group B contains no such internal contradictions. Additionally, Group A features a significantly higher density of negative clues (statements like "not blue" or "not square") compared to the predominantly positive assertions found in Group B. Finally, Group A exhibits a more chaotic alternation between Alice and Bob, often with Bob dominating the later turns, while Group B shows more balanced or predictable turn-taking patterns.
- inside: Alice says the object is blue. | Alice says the object is red. | Bob says the object is big.
- inside: Bob says the object is square. | Bob says the object is not square. | Alice says the object is metal. | Bob says the object is not blue. | Bob says the object is not wooden. | Bob says the object is blue. | Alice says the object is red.
- inside: Bob says the object is not round. | Alice says the object is wooden. | Alice says the object is small. | Alice says the object is not red. | Bob says the object is square. | Alice says the object is not metal. | Bob says the object is big.
- inside: Bob says the object is metal. | Bob says the object is round.
- inside: Alice says the object is metal. | Bob says the object is big. | Alice says the object is not wooden. | Bob says the object is not wooden. | Alice says the object is not red. | Bob says the object is not blue. | Alice says the object is metal.
- contrast (same number of clues, other clusters): Alice says the object is not blue. | Alice says the object is not small. | Bob says the object is wooden.
- contrast (same number of clues, other clusters): Alice says the object is not blue. | Alice says the object is not blue. | Bob says the object is red. | Bob says the object is round. | Bob says the object is blue. | Bob says the object is wooden. | Alice says the object is not wooden.
- contrast (same number of clues, other clusters): Alice says the object is not wooden. | Bob says the object is small. | Bob says the object is not blue. | Alice says the object is big. | Alice says the object is red. | Alice says the object is not big. | Alice says the object is not metal.

## Cluster 4 (228 states; turns {'1': 80, '2': 71, '3': 46, '4': 19, '5': 10, '6': 2})
- mean P(value 0): {'colour': 0.4300000071525574, 'shape': 0.4699999988079071, 'size': 0.4000000059604645, 'material': 0.47999998927116394}
- over-represented: belief pattern = 76 (1% vs 0%); clue attribute = material (73% vs 24%); turn = 1 (35% vs 12%); contradicted attributes = 0 (81% vs 44%)
- **Qwen/Qwen3.5-9B says:** The histories in Group A are distinguished by a pervasive and repeated focus on the "wooden" property, which appears in nearly every single clue across all entries, whereas Group B contains no mention of the material property at all. Additionally, Group A features a high frequency of Alice speaking alone or leading the sequence, while Group B includes more instances of Bob speaking first or providing the sole clue in a turn. Finally, Group A contains explicit contradictions regarding the material (e.g., Alice claiming it is metal while Bob claims it is not, or Alice claiming it is not wooden while Bob claims it is), a pattern of conflict absent in Group B.
- inside: Alice says the object is not wooden.
- inside: Alice says the object is big. | Bob says the object is not small. | Bob says the object is not wooden.
- inside: Alice says the object is metal.
- inside: Bob says the object is not wooden.
- inside: Alice says the object is not round. | Alice says the object is wooden.
- contrast (same number of clues, other clusters): Alice says the object is big.
- contrast (same number of clues, other clusters): Alice says the object is big. | Alice says the object is not round. | Bob says the object is not small.
- contrast (same number of clues, other clusters): Alice says the object is small.

## Cluster 5 (293 states; turns {'1': 151, '2': 82, '3': 36, '4': 14, '5': 6, '6': 3, '8': 1})
- mean P(value 0): {'colour': 0.4099999964237213, 'shape': 0.49000000953674316, 'size': 0.3799999952316284, 'material': 0.6399999856948853}
- over-represented: belief pattern = 41 (10% vs 2%); turn = 1 (52% vs 12%); contradicted attributes = 0 (87% vs 44%); clue source = Alice (88% vs 51%)
- **Qwen/Qwen3.5-9B says:** The histories in Group A are exclusively composed of clues provided by Alice, whereas Group B contains a mix of clues from both Alice and Bob. Additionally, Group A features a high frequency of redundant or contradictory statements from Alice (such as claiming an object is both "small" and "not big"), while Group B displays a more balanced distribution of sources and fewer such direct contradictions within single turns.
- inside: Alice says the object is small.
- inside: Alice says the object is not red.
- inside: Alice says the object is not big.
- inside: Alice says the object is blue. | Alice says the object is not big.
- inside: Alice says the object is blue. | Alice says the object is not big. | Alice says the object is small.
- contrast (same number of clues, other clusters): Alice says the object is wooden.
- contrast (same number of clues, other clusters): Bob says the object is not small.
- contrast (same number of clues, other clusters): Bob says the object is red.

## Cluster 6 (284 states; turns {'1': 103, '2': 100, '3': 54, '4': 17, '5': 5, '6': 3, '7': 1, '8': 1})
- mean P(value 0): {'colour': 0.4300000071525574, 'shape': 0.47999998927116394, 'size': 0.38999998569488525, 'material': 0.6000000238418579}
- over-represented: turn = 1 (36% vs 12%); contradicted attributes = 0 (85% vs 44%); clue source = Bob (85% vs 49%); clue attribute = colour (39% vs 26%)
- **Qwen/Qwen3.5-9B says:** The histories in Group A are distinguished by the fact that every single clue is spoken by Bob, whereas Group B contains a mix of clues from both Alice and Bob. Additionally, Group A exclusively features negative or exclusionary statements (e.g., "not red," "not metal"), while Group B includes positive assertions about the object's properties (e.g., "is wooden," "is big"). Finally, Group A never presents a contradiction within a single turn, whereas Group B frequently contains conflicting statements from the same speaker in the same turn (e.g., Bob saying the object is both wooden and blue in B7).
- inside: Bob says the object is not red.
- inside: Bob says the object is small.
- inside: Bob says the object is not big. | Bob says the object is not metal.
- inside: Bob says the object is not square.
- inside: Bob says the object is not blue.
- contrast (same number of clues, other clusters): Alice says the object is big.
- contrast (same number of clues, other clusters): Bob says the object is not wooden.
- contrast (same number of clues, other clusters): Bob says the object is wooden. | Alice says the object is blue.

## Cluster 7 (353 states; turns {'1': 41, '2': 48, '3': 61, '4': 49, '5': 41, '6': 39, '7': 35, '8': 39})
- mean P(value 0): {'colour': 0.4099999964237213, 'shape': 0.27000001072883606, 'size': 0.3700000047683716, 'material': 0.5600000023841858}
- over-represented: clue attribute = shape (52% vs 25%); turn = 3 (17% vs 12%); clue phrasing = affirmative (65% vs 50%); clue negations = 0 (65% vs 50%)
- **Qwen/Qwen3.5-9B says:** The histories in Group A are characterized by a high frequency of immediate contradictions within the same speaker's turn (e.g., Alice stating a property and then immediately denying it in the next clue), whereas Group B contains no such direct self-contradictions. Additionally, Group A features a significantly higher density of clues provided by Alice compared to Bob, while Group B shows a more balanced or Bob-dominant distribution. Finally, Group A includes several instances where a single speaker repeats the exact same clue consecutively, a pattern that is absent in Group B.
- inside: Alice says the object is blue. | Alice says the object is not round. | Bob says the object is square. | Alice says the object is not small.
- inside: Alice says the object is square. | Alice says the object is not blue. | Alice says the object is not square. | Alice says the object is square. | Bob says the object is not blue. | Alice says the object is wooden.
- inside: Alice says the object is square. | Alice says the object is not square. | Bob says the object is not red. | Alice says the object is not metal.
- inside: Bob says the object is metal. | Alice says the object is square. | Bob says the object is blue. | Alice says the object is red. | Alice says the object is not wooden. | Alice says the object is metal. | Alice says the object is red.
- inside: Alice says the object is round. | Alice says the object is round.
- contrast (same number of clues, other clusters): Alice says the object is metal. | Alice says the object is not big. | Alice says the object is not round. | Bob says the object is wooden.
- contrast (same number of clues, other clusters): Alice says the object is not blue. | Alice says the object is not metal. | Alice says the object is not metal. | Alice says the object is big. | Alice says the object is blue. | Alice says the object is not square.
- contrast (same number of clues, other clusters): Bob says the object is not red. | Bob says the object is not red. | Alice says the object is square. | Alice says the object is blue.
