# Sign2Text test progress

Updated 21 Oct 2026 after the photo scan finished and the model was retrained.

## 1. Latest accuracy test

Held-out still photos after scanning every letter folder. 6,218 photos, 95-D model.

Overall: **98.7%** (was 98.2% on the smaller scan).

Weakest still: **N** 95.2% (mostly called M), **M** 95.7% (mostly called N), **I** 96.3%, **U** 96.9%. **J** was 99.6% and **Z** was 99.2% on frozen photos. Live signing of J and Z is still a motion, which this photo test does not cover.

## 2. Earlier accuracy test (before the extra photos)

Held-out still photos from the first landmark file. 1,652 photos.

Overall: **98.2%**.

| Letter | Correct on test | Missed as |
| --- | --- | --- |
| A | 60/62 | M, N |
| B | 60/60 | |
| C | 56/56 | |
| D | 66/67 | F |
| E | 63/63 | |
| F | 76/78 | C, C |
| G | 64/66 | D, Z |
| H | 64/65 | G |
| I | 65/65 | |
| J | 68/69 | E |
| K | 72/73 | G |
| L | 69/69 | |
| M | 45/47 | A, W |
| N | 35/38 | M, M, M |
| O | 62/62 | |
| P | 54/56 | H, Q |
| Q | 56/58 | N, N |
| R | 67/70 | E, G, U |
| S | 68/69 | I |
| T | 63/63 | |
| U | 64/67 | R, R, R |
| V | 69/69 | |
| W | 65/67 | R, V |
| X | 60/61 | S |
| Y | 69/69 | |
| Z | 63/63 | |

Weakest on that photo test: **N** (confused with M), then **U** (confused with R) and **M**. **J** and **Z** score well on frozen photos. They are drawn in the air, and that motion was not part of this test.

## 3. Photo scan

Finished. Every letter folder was scanned. A row is saved only when a hand is found.

Saved rows in `dataset/processed/landmarks.csv`: **62,173**.

| Letter | Rows saved |
| --- | --- |
| A | 2266 |
| B | 2236 |
| C | 2078 |
| D | 2552 |
| E | 2338 |
| F | 2895 |
| G | 2523 |
| H | 2446 |
| I | 2448 |
| J | 2630 |
| K | 2732 |
| L | 2583 |
| M | 1881 |
| N | 1460 |
| O | 2330 |
| P | 2105 |
| Q | 2196 |
| R | 2600 |
| S | 2612 |
| T | 2392 |
| U | 2545 |
| V | 2581 |
| W | 2498 |
| X | 2213 |
| Y | 2623 |
| Z | 2410 |

No letters left to scan.
