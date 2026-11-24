# Project summary: predicting school-age children per building

This document explains the project end to end: the task, the simulated data, the preprocessing, the
models, how they were tuned and evaluated, what the results say, and what to do next. It is written
for a technical reader who is new to the repository.

## 1. Task and settings

The task is to predict, for each residential building, how many children it holds in each of three
school-age cohorts (kindergarten, elementary, high school), from the building's and its
neighborhood's characteristics. The data are simulated, so the generating process is known and
every model can be judged against it. The unit of prediction is the building; the number of
apartments is part of the data and is also the natural exposure (the building's size).

| Setting | Value |
|---|---|
| Unit of prediction | building |
| Targets | `n_kindergarten`, `n_elementary`, `n_highschool` (counts per building; no total column) |
| Neighborhoods | 150  |
| Buildings | 1,529 |
| Apartments | 61,317  |
| Split | random over buildings: 917 train / 306 validation / 306 test (about 60 / 20 / 20 %) |



## 2. Data generation process

The data are simulated, so the process that generates them is known exactly. It follows how
children are housed: children live in apartments, apartments are in buildings, and buildings are in
neighborhoods. Each level has its own role.

- **Neighborhood: the context.** Socioeconomic status (SES), median age, average household size,
  nearby daycares and school status describe the area. Every building in a neighborhood shares
  them. They shape both the apartments built there and the families who live in them.
- **Building: the level we predict.** Each neighborhood has a random number of buildings. Each
  building has a random number of apartments and a mix of apartment sizes (3 to 6 rooms). The mix
  leans toward larger apartments where households are larger or SES is higher. A building's count
  per cohort is the sum over its apartments.
- **Apartment: where the children are generated.** An apartment's rooms and its neighborhood set
  how many children live there and how old they are. Larger apartments hold more children, and
  older ones.

Hidden effects, two per building and one per neighborhood, make apartments in the same building or
neighborhood more alike than their features explain, as unmeasured local factors do in real data.

The whole draw is reproducible from one seed (42). The steps below follow the order of the draws.
Throughout, $j$ indexes neighborhoods, $b$ buildings and $i$ apartments.

### 2.1 Neighborhoods: the context

There are 150 neighborhoods. For each one, five values are drawn, in this order.

**SES**, standardized: 0 is the population average, and one unit is one standard deviation.

$$
S_j = \mathrm{clip}\big(\mathcal N(\mu=0,\ \sigma^2=1^2),\ \min=-2.5,\ \max=2.5\big)
$$

$\mathrm{clip}$ caps a draw outside the bounds at the bound, which removes implausible extremes.

**Median age**, in years, around a typical family-age neighborhood:

$$
M_j = \mathrm{clip}\big(\mathcal N(\mu=37,\ \sigma^2=5^2),\ \min=25,\ \max=52\big)
$$

**Average household size**, in people. It is 2.6 at median age 37 and falls by 0.02 people per year
of median age, so older areas have smaller households. The noise $\varepsilon_j$ lets
neighborhoods of the same age differ.

$$
H_j = \mathrm{clip}\big(2.6 - 0.02\,(M_j - 37) + \varepsilon_j,\ \min=1.8,\ \max=4.2\big),
\qquad
\varepsilon_j \sim \mathcal N(\mu=0,\ \sigma^2=0.25^2)
$$

**Daycares within 500 m**, a count. At average SES and median age 37 the expected count is
$e^{1.0} \approx 2.7$. Each SES unit multiplies it by $e^{0.15} \approx 1.16$. Ten more years of
median age multiply it by $e^{-0.3} \approx 0.74$.

$$
D_j \sim \mathrm{Poisson}\Big(\lambda = \exp\big[1.0 + 0.15\,S_j - 0.03\,(M_j - 37)\big]\Big)
$$

**School status**, one of three categories:

$$
P(Z_j = \text{existing}) = 0.55,
\qquad
P(Z_j = \text{planned}) = 0.20,
\qquad
P(Z_j = \text{none}) = 0.25
$$

Household size depends on age, and daycares on SES and age. Apart from these links, the features
are drawn independently.

### 2.2 Buildings: count, size and room mix

**Buildings per neighborhood**: at least one and ten on average, 1,529 in total.

$$
B_j = 1 + \mathrm{Poisson}(\lambda=9)
$$

**Apartments per building**, rounded to a whole number and then clipped, mostly between 20 and 60:

$$
A_b = \mathrm{clip}\big(\mathrm{round}(\mathcal N(\mu=40,\ \sigma^2=12^2)),\ \min=12,\ \max=80\big)
$$

$A_b$ is drawn independently of every neighborhood feature: a building's size says nothing about
its neighborhood. This matters for the models' exposure (section 4) and for the importance of
`n_apartments` (section 7.3).

**Room mix.** Each building splits its apartments over 3, 4, 5 and 6 rooms. The neighborhood sets
an expected mix, and each building draws its own mix around it. First, household size is put on a
scale of half-persons above 2.6 and combined with SES into one tilt. Section 2.3 uses the same
scale.

$$
\tilde H_j = \frac{H_j - 2.6}{0.5},
\qquad
q_j = 0.35\,\tilde H_j + 0.20\,S_j
$$

The tilt rotates the base shares of 30%, 38%, 22% and 10% for 3, 4, 5 and 6 rooms:

$$
\tilde\omega_{jr} = \pi_r \exp\Big(q_j\,\frac{r - 4}{2}\Big),
\qquad
(\pi_3, \pi_4, \pi_5, \pi_6) = (0.30,\ 0.38,\ 0.22,\ 0.10),
\qquad
\omega_{jr} = \frac{\tilde\omega_{jr}}{\sum_{r'=3}^{6} \tilde\omega_{jr'}}
$$

Here $r$ is the number of rooms. Four rooms is the pivot: its multiplier is always 1. A positive
tilt (larger households, higher SES) moves weight from 3 and 4 rooms toward 5 and 6 rooms.
Dividing by 2 keeps the shift gradual. In the data the tilt runs from about −0.6 to 0.6. At
$q_j = 0.5$ the multipliers are 0.78, 1, 1.28 and 1.65, and the shares become 22%, 36%, 27% and
16%. Then, building by building, a Dirichlet draw gives the building its own mix around its
neighborhood's shares $\omega_{jr}$. A multinomial draw turns that mix into room counts that sum
exactly to $A_b$:

$$
\mathbf g_b \sim \mathrm{Dirichlet}\big(30\,\boldsymbol\omega_j\big),
\qquad
(A_{b3},\ A_{b4},\ A_{b5},\ A_{b6}) \sim \mathrm{Multinomial}(A_b,\ \mathbf g_b)
$$

A concentration of 30 keeps buildings near their neighborhood's mix while letting them differ. A
larger value would make them more alike.

### 2.3 Apartments: children and their ages

Building $b$ holds $A_b$ apartments, $A_{br}$ of them with $r$ rooms. $R_i$ is the number of rooms
of apartment $i$. Both steps below use $\tilde H_j$ from section 2.2 and three more scaled values:

$$
\tilde M_j = \frac{M_j - 37}{10},
\qquad
c_i = R_i - 4,
\qquad
\mathrm{sat}(D_j) = 1 - e^{-D_j/3}
$$

$\tilde M_j$ is median age in decades above 37, and $c_i$ counts rooms above or below 4. The
saturation $\mathrm{sat}$ gives daycares diminishing returns: 3 daycares already reach 63% of the
maximum.

**Total children per apartment.** First comes a hidden effect for every building, then one for
every neighborhood. Each is shared by all the apartments below it and stands for unmeasured local
factors:

$$
u_b \sim \mathcal N(\mu=0,\ \sigma^2=0.25^2),
\qquad
w_j \sim \mathcal N(\mu=0,\ \sigma^2=0.20^2)
$$

Then come the expected number of children $\mu_i$ of apartment $i$, on the log scale, and its
count $N_i$:

$$
\begin{aligned}
\log \mu_i ={}& -1.30
+ 0.45\,\mathbb 1\lbrace R_i = 4\rbrace + 0.80\,\mathbb 1\lbrace R_i = 5\rbrace
+ 0.95\,\mathbb 1\lbrace R_i = 6\rbrace
+ 0.05\,S_j^2 + 0.30\,\tilde H_j + 0.12\,\mathrm{sat}(D_j) \\
&+ 0.10\,\mathbb 1\lbrace Z_j = \text{existing}\rbrace
+ 0.04\,\mathbb 1\lbrace Z_j = \text{planned}\rbrace
- 0.06\,\tilde M_j + 0.10\,c_i\,\tilde H_j + u_b + w_j
\end{aligned}
$$

$$
N_i \sim \mathrm{NB2}\big(\text{mean}=\mu_i,\ \phi=3\big),
\qquad
\mathrm{Var}(N_i) = \mu_i + \frac{\mu_i^2}{3}
$$

The count is a negative binomial: a count whose variance exceeds its mean by a term that grows
with the square of the mean (the "NB2" form). The indicator $\mathbb 1\lbrace\cdot\rbrace$ is 1
when its condition holds and 0 otherwise. Each coefficient $\beta$ multiplies the expected count
by $e^{\beta}$ per unit of its term:

- **−1.30:** a 3-room apartment where every other term is zero expects $e^{-1.30} \approx 0.27$
  children. Every other term is zero at SES 0, household size 2.6, median age 37, no daycares, no
  school and no hidden effects.
- **0.45, 0.80, 0.95:** at household size 2.6, 4, 5 and 6 rooms multiply that by 1.57, 2.23 and
  2.59. The gain flattens after 5 rooms.
- **0.05:** a weak U-shape in SES. SES ±2 adds about 22%.
- **0.30:** in a 4-room apartment, each half-person of household size multiplies the count by
  1.35. This is the strongest neighborhood signal.
- **0.12:** daycares add at most about 13%.
- **0.10, 0.04:** against no school, an existing school multiplies the count by 1.11 and a planned
  one by 1.04. Families settle near schools, and a planned school is a weaker draw.
- **−0.06:** a neighborhood a decade older multiplies the count by 0.94.
- **0.10:** the rooms × household-size interaction lets larger apartments take up more of the
  local demand for space. A half-person of household size multiplies the count by
  $e^{0.20} \approx 1.22$ at 3 rooms and by $e^{0.50} \approx 1.65$ at 6 rooms.
- **$\phi = 3$:** the negative binomial adds variance $\mu^2/3$ beyond a Poisson count.

Two terms are absent by design: there is **no linear SES term** and **no building-size term**. SES
still acts indirectly, through the daycares (section 2.1) and the room mix (section 2.2).

**The cohort split.** First comes a second hidden effect for every building, then a lifecycle score
$\theta_i$ for every apartment. A higher score means an older family stage:

$$
v_b \sim \mathcal N(\mu=0,\ \sigma^2=0.20^2)
$$

$$
\theta_i = 0.30\,c_i - 0.20\,\mathrm{sat}(D_j) + 0.25\,\tilde M_j + 0.15\,\tilde H_j
+ 0.12\,(c_i)_+\,(\tilde M_j)_+ + v_b
$$

Here $(x)_+ = \max(x, 0)$. The score tilts the reference shares of 40%, 35% and 25% (kindergarten,
elementary, high school). A multinomial then splits the apartment's children over the three
cohorts:

$$
p_{i,\mathrm{kg}} \propto 0.40\,e^{-1.0\,\theta_i},
\qquad
p_{i,\mathrm{el}} \propto 0.35,
\qquad
p_{i,\mathrm{hs}} \propto 0.25\,e^{1.1\,\theta_i}
$$

$$
(N_{i,\mathrm{kg}},\ N_{i,\mathrm{el}},\ N_{i,\mathrm{hs}}) \sim \mathrm{Multinomial}(N_i,\ \mathbf p_i)
$$

The three probabilities are normalized to sum to 1 (a softmax). At $\theta_i = 0$ they are exactly
0.40, 0.35 and 0.25. A higher score moves children from kindergarten toward high school:

- More rooms, an older neighborhood and larger households raise the score. Daycares lower it,
  because they draw families with young children.
- The interaction adds a bonus only for large apartments (above 4 rooms) in older neighborhoods
  (above 37). It stands for families that move up into bigger homes. A plain product would also
  give the bonus to small apartments in young neighborhoods, where both factors are negative.
- SES and school status do not enter the score directly. SES acts on it indirectly, through the
  daycares and the room counts.

The multinomial makes the cohorts sum exactly to $N_i$.

### 2.4 Buildings: the sums we predict

The targets are the sums over each building's apartments, for kindergarten (kg), elementary (el)
and high school (hs):

$$
C_{bk} = \sum_{i \in b} N_{ik},
\qquad
k \in \lbrace \mathrm{kg}, \mathrm{el}, \mathrm{hs} \rbrace
$$

The building's total is the sum of the three cohorts, and the data do not store it.

What count does a building expect? All its apartments share the neighborhood's features and the
hidden effects of the building and its neighborhood; they differ only in rooms. Four quantities
enter the formula:

- $s_{br} = A_{br}/A_b$ is the share of $r$-room apartments.
- $\bar\eta_{jr}$ is the log mean $\log\mu_i$ of an $r$-room apartment in neighborhood $j$, without
  $u_b + w_j$.
- $\bar\theta_{jr}$ is that apartment's score, without $v_b$.
- $p_k(\cdot)$ is the cohort probability above.

Let $x_b$ be the building's features: its neighborhood's five values and its room counts
$A_{br}$. Given $x_b$ and the hidden effects:

$$
\mathbb E\big[C_{bk} \mid x_b, u_b, w_j, v_b\big]
= A_b\, e^{u_b + w_j} \sum_{r=3}^{6} s_{br}\, e^{\bar\eta_{jr}}\, p_k(\bar\theta_{jr} + v_b)
$$

The hidden effects are independent of each other and of the features. The average of
$e^{u_b + w_j}$ is therefore the constant $e^{(0.25^2 + 0.20^2)/2} \approx 1.053$. Only the
average over $v_b$ inside the softmax has no closed form:

$$
\mathbb E\big[C_{bk} \mid x_b\big]
= A_b\, e^{(0.25^2 + 0.20^2)/2} \sum_{r=3}^{6} s_{br}\, e^{\bar\eta_{jr}}\,
\mathbb E_{v}\big[p_k(\bar\theta_{jr} + v)\big],
\qquad v \sim \mathcal N(\mu=0,\ \sigma^2=0.20^2)
$$

For the total, the probabilities sum to 1 and drop out:

$$
\mathbb E\Big[\sum_k C_{bk} \;\Big|\; x_b\Big]
= A_b\, e^{(0.25^2 + 0.20^2)/2} \sum_{r=3}^{6} s_{br}\, e^{\bar\eta_{jr}}
\approx 1.053\, A_b \sum_{r=3}^{6} s_{br}\, e^{\bar\eta_{jr}}
$$

With the room shares and the neighborhood fixed, a building's expected count is **proportional to
its number of apartments** $A_b$.

### 2.5 What the models see and what is hidden

The models see one row per building (section 2.6). The following are not in the data:

- **The hidden effects** $u_b$, $w_j$ and $v_b$. The simulator returns them next to the data, for
  checks that recover the coefficients, but they are never columns.
- **The apartments**: each apartment's rooms, children and cohorts. The models see only a
  building's room counts and its cohort sums.
- **Each building's own room mix** $\mathbf g_b$. Its realized room counts are observed, and the
  outcomes depend only on those.

The hidden effects are drawn independently of every feature. They add unexplained variation around
the expected count: they are heterogeneity, not confounders. A building one standard deviation
above average in $u_b$ expects $e^{0.25} \approx 1.28$ times as many children, and no feature
reveals it. Even given the effects, each apartment's count is a negative binomial draw and its
split a multinomial draw. A model that knew the expectation of section 2.4 exactly, effects
included, would therefore still not predict the counts without error. Without the effects, the
error is larger still.

### 2.6 The dataset's columns

One row per building, 1,529 rows, the columns in this order:

| Column | Level | Symbol | Description |
|---|---|---|---|
| `building_id` | building | $b$ | the building's id |
| `neighborhood_id` | neighborhood | $j$ | the neighborhood's id |
| `ses` | neighborhood | $S_j$ | socioeconomic status, standardized |
| `avg_household_size` | neighborhood | $H_j$ | average household size (people) |
| `median_age` | neighborhood | $M_j$ | median age (years) |
| `n_daycares_500m` | neighborhood | $D_j$ | daycares within 500 m |
| `school_status` | neighborhood | $Z_j$ | existing, planned or none |
| `3_rooms` | building | $A_{b3}$ | apartments with 3 rooms |
| `4_rooms` | building | $A_{b4}$ | apartments with 4 rooms |
| `5_rooms` | building | $A_{b5}$ | apartments with 5 rooms |
| `6_rooms` | building | $A_{b6}$ | apartments with 6 rooms |
| `n_apartments` | building | $A_b$ | apartments, the sum of the four room columns |
| `n_kindergarten` | target | $C_{b,\mathrm{kg}}$ | kindergarten children |
| `n_elementary` | target | $C_{b,\mathrm{el}}$ | elementary children |
| `n_highschool` | target | $C_{b,\mathrm{hs}}$ | high-school children |

The neighborhood columns repeat on every building of the neighborhood. The two id columns are not
features. `building_id` names the row. `neighborhood_id` groups the buildings for the bootstrap,
which resamples neighborhoods (section 5).

## 3. Preprocessing

## 4. Models

### 4.1 The shared exposure treatment

### 4.2 Model 1: independent cohorts

### 4.3 Model 2: total × probability

### 4.4 Model 3: Dirichlet-multinomial (placeholder)

## 5. Tuning and evaluation protocol

## 6. Results and model inspection

## 7. Discussion

### 7.1 Which model when

### 7.2 Why the models fail where they do

### 7.3 Why `n_apartments` has low importance

### 7.4 Future steps
