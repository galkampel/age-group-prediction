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
  nearby daycares and school status describe the area. They are linked as in real neighborhoods:
  SES shapes the population's age and household size, the daycares follow the population and its
  purchasing power, and the schools follow the population. Every building in a neighborhood shares them. They shape both the apartments built
  there and the families who live in them.
- **Building: the level we predict.** Each neighborhood has a random number of buildings. A
  building either keeps its neighborhood's usual mid-rise form or is a tower, and that sets its
  typical number of apartments. Its mix of apartment sizes (3 to 6 rooms) leans toward larger
  apartments where SES is higher, slightly toward larger ones where households are larger, and
  toward smaller ones in larger buildings. A building's count per cohort is the sum over its
  apartments.
- **Apartment: where the children are generated.** An apartment's rooms and its neighborhood set
  how many children live there and how old they are. The number of children follows how far
  the neighborhood's household size lies above that of a household without school-age children.
  Larger apartments hold more children, but where households are large the small apartments hold
  many children too. The children are older in larger apartments and in older neighborhoods, and
  younger near daycares and where a school is planned.

Hidden values act at two places, and none of them is in the dataset. A building's type and two
values per neighborhood shape the building sizes (section 2.2). Two effects per building and one
per neighborhood shape the children (section 2.3): they make apartments in the same building or
neighborhood more alike than their features explain, as unmeasured local factors do in real data.

The whole draw is reproducible from one seed (42). The steps below follow the order of the draws.
Throughout, $j$ indexes neighborhoods, $b$ buildings and $i$ apartments.

### 2.1 Neighborhoods: the context

There are 150 neighborhoods, calibrated to urban neighborhoods in Israel. Five values describe each
one, drawn in this order: the population first, then the facilities that follow it. The formulas
are centered on a **reference neighborhood**: SES 0, median age 31 and household size 3.0.

**SES** $S_j$ is the neighborhood's socioeconomic level on a national index, in standard
deviations from the national average; negative values are poorer neighborhoods.

$$
S_j = \mathrm{clip}\big(\mathcal N(\mu=0,\ \sigma^2=1^2),\ \min=-2.5,\ \max=2.5\big)
$$

**Median age** $M_j$ is the median age of all the residents, children included, so a neighborhood
with many children is young. It rises by 3.5 years per SES unit: in Israel the lowest-SES
neighborhoods are largely Haredi and Arab communities with large, young families.

$$
M_j = \mathrm{clip}\big(31 + 3.5\,S_j + \varepsilon^{M}_j,\ \min=16,\ \max=50\big),
\qquad
\varepsilon^{M}_j \sim \mathcal N(\mu=0,\ \sigma^2=4.5^2)
$$

**Average household size** $H_j$ is the mean number of people per home, adults and children
together.

$$
H_j = \mathrm{clip}\big(3.0 - 0.06\,(M_j - 31) - 0.20\,S_j + 0.18\,S_j^2 + \varepsilon^{H}_j,\ \min=1.8,\ \max=6.5\big),
\qquad
\varepsilon^{H}_j \sim \mathcal N(\mu=0,\ \sigma^2=0.30^2)
$$

Household size falls with median age: children both lower the median age and enlarge the
households, and old neighborhoods hold couples whose children have left and people living alone.
Directly and through the median age, SES shifts the household size by
$-0.41\,S_j + 0.18\,S_j^2$, which is lowest near SES 1.1. This is a reverse-J: steep at the bottom
(the large Haredi and Arab families), with a slight rise at the top (well-off family
neighborhoods, against mid-SES ones with more singles and couples).

**Daycares within 500 m** $D_j$:

$$
D_j \sim \mathrm{Poisson}\Big(\lambda = \exp\big[1.02 + 0.30\,(H_j - 3.0) + 0.35\,S_j - 0.03\,(M_j - 31)\big]\Big)
$$

The reference neighborhood expects $e^{1.02} \approx 2.8$ daycares. Demand multiplies the count by
$e^{0.30} \approx 1.35$ per person of household size and by $e^{-0.3} \approx 0.74$ per ten more
years of median age (fewer toddlers); purchasing power multiplies it by $e^{0.35} \approx 1.42$
per SES unit. At the bottom
of the SES scale the two roughly cancel: the largest households have the least purchasing power.

**School status** $Z_j$ is existing, planned or none, with probabilities proportional to three
weights:

$$
P(Z_j = z) = \frac{\psi_{jz}}{\psi_{j,\text{existing}} + \psi_{j,\text{planned}} + \psi_{j,\text{none}}}
$$

$$
\psi_{j,\text{existing}} = 0.55\,e^{0.8\,(H_j - 3.0)},
\qquad
\psi_{j,\text{planned}} = 0.20\,e^{-0.08\,(M_j - 31)},
\qquad
\psi_{j,\text{none}} = 0.25
$$

In the reference neighborhood the probabilities are exactly 55%, 20% and 25%. Schools are built
where the children are: against none, each person of household size multiplies the odds of an
existing school by $e^{0.8} \approx 2.23$, and a median age ten years younger multiplies the odds
of a planned school by the same $e^{0.8} \approx 2.23$, because young neighborhoods are often new
ones that are still growing.
SES does not enter, because schools are public.

The five values are correlated: low-SES neighborhoods are younger and have larger households. A
model therefore separates their effects less easily than if they were independent.

### 2.2 Buildings: count, size and room mix

A building is described by its number of apartments and their split over room counts.

**Buildings per neighborhood** only sets how much data each neighborhood contributes (at least
one, and ten on average):

$$
B_j = 1 + \mathrm{Poisson}(\lambda=9)
$$

**Two hidden neighborhood values.** The tower propensity $\rho_j$ is how far the neighborhood
leans toward high-rise building beyond its features: towers come with renewal or
new-construction plans, so they cluster in some neighborhoods. The texture $\tau_j$ shifts the
size of all the neighborhood's buildings on the log scale, because they were mostly built in the
same period and under the same rules; one standard deviation is a factor of
$e^{0.25} \approx 1.28$.

$$
\rho_j \sim \mathcal N(\mu=0,\ \sigma^2=1.0^2),
\qquad
\tau_j \sim \mathcal N(\mu=0,\ \sigma^2=0.25^2)
$$

**Building type** $T_b$ is 1 for a tower and 0 for a texture building, the neighborhood's usual
mid-rise. The median age enters in decades:

$$
\tilde M_j = \frac{M_j - 31}{10}
$$

$$
P(T_b = 1) = \mathrm{logistic}\big(-2.35 + 0.7\,S_j + 0.3\,\tilde M_j
+ 0.7\,\mathbb 1\lbrace Z_j = \text{planned}\rbrace + \rho_j\big),
\qquad
\mathrm{logistic}(x) = \frac{1}{1 + e^{-x}}
$$

In the reference neighborhood, with no planned school and $\rho_j = 0$, a building is a tower
with probability
$\mathrm{logistic}(-2.35) \approx 9\%$. The odds multiply by $e^{0.7} \approx 2.01$ per SES unit
(towers pay off where land is expensive), by $e^{0.3} \approx 1.35$ per ten years of median age
(urban renewal in established areas), by $e^{0.7} \approx 2.01$ with a planned school (new
neighborhoods are built high) and by $e^{1.0} \approx 2.72$ per standard deviation of $\rho_j$.

**Apartments per building** $A_b$:

$$
A_b = \mathrm{clip}\Big(\mathrm{round}\big(m_b\,e^{\tau_j + \varepsilon^{A}_b}\big),\ \min=8,\ \max=160\Big),
\qquad
m_b = \begin{cases} 30 & \text{if } T_b = 0 \\ 90 & \text{if } T_b = 1 \end{cases},
\qquad
\varepsilon^{A}_b \sim \mathcal N(\mu=0,\ \sigma^2=0.15^2)
$$

A texture building has a median of 30 apartments (6 to 10 floors), and a tower has three times as
many. The deviations multiply the size, because building sizes are right-skewed. The building's
own deviation has a smaller standard deviation than the neighborhood's shift $\tau_j$ (0.15
against 0.25), so buildings of the same type differ more between neighborhoods than inside one. A building's size
therefore says something about its neighborhood: large buildings are more common where SES is
high, where the residents are older and where a school is planned. The size is also the models'
exposure (section 4); section 7.3 returns to its importance as a feature.

**Room mix.** Each building splits its apartments over 3, 4, 5 and 6 rooms. Household size enters
on the scale used again in section 2.3, and the tilt $q_b$ says how far the building leans toward
large apartments:

$$
\tilde H_j = \frac{H_j - 3.0}{0.75},
\qquad
q_b = 0.35\,S_j + 0.10\,\tilde H_j - 0.32\,\log\frac{A_b}{30}
$$

The tilt is zero for a building of 30 apartments in the reference neighborhood. Its three terms:

- **SES** has the widest range, from $-0.875$ to $0.875$ over the SES scale: apartment size is
  bought.
- **Household size** is weak: a household size of 4.5 adds only 0.20. In the poorest
  neighborhoods, which have the largest households, the SES term outweighs it, so large
  households live in small apartments there: crowding.
- **Building size.** Large buildings are planned with small units. A building twice the texture
  median lowers the tilt by $0.32 \log 2 \approx 0.22$, and a typical tower lowers it by
  $0.32 \log 3 \approx 0.35$, as much as one SES unit adds. Towers are more common at high SES,
  so there this term takes back part of the SES term.

The tilt rotates the base shares around 4 rooms:

$$
\tilde\omega_{br} = \pi_r \exp\Big(q_b\,\frac{r - 4}{2}\Big),
\qquad
(\pi_3, \pi_4, \pi_5, \pi_6) = (0.30,\ 0.41,\ 0.24,\ 0.05),
\qquad
\omega_{br} = \frac{\tilde\omega_{br}}{\sum_{r'=3}^{6} \tilde\omega_{br'}}
$$

$\omega_{br}$ is the expected share of $r$-room apartments in building $b$. $e^{q_b}$ multiplies
the 6-room weight and $e^{-q_b/2}$ the 3-room weight. At $q_b = 0.35$, a building of 30 apartments
at SES 1 and household size 3.0, the multipliers are 0.84, 1, 1.19 and 1.42, and the shares become
25%, 40%, 28% and 7%.

A Dirichlet draw gives each building its own mix $\mathbf g_b$ around the expected shares
$\boldsymbol\omega_b$, and a multinomial draw turns it into room counts $A_{br}$ that sum to
$A_b$:

$$
\mathbf g_b \sim \mathrm{Dirichlet}\big(30\,\boldsymbol\omega_b\big),
\qquad
(A_{b3},\ A_{b4},\ A_{b5},\ A_{b6}) \sim \mathrm{Multinomial}(A_b,\ \mathbf g_b)
$$

With the concentration 30, a building's own 4-room share has a standard deviation of about 9
percentage points around the reference building's expected 41%.

### 2.3 Apartments: children and their ages

$R_i$ is the number of rooms of apartment $i$. Children are generated per apartment in two steps,
how many and then how old. Inside one building the apartments differ only in their rooms. Besides
$\tilde M_j$ and $\tilde H_j$ (section 2.2), three values enter:

$$
L_j = \log\frac{H_j - 1.7}{3.0 - 1.7},
\qquad
c_i = R_i - 4,
\qquad
\mathrm{sat}(D_j) = 1 - e^{-D_j/3}
$$

The household size of a neighborhood without children aged 3 to 18 is set at 1.7 (singles,
couples, and families with younger or older children). $H_j - 1.7$ is the excess over such a
household, and $L_j$ is its log relative to the reference neighborhood's $3.0 - 1.7 = 1.3$ people.
The daycare saturation rises from 0 toward 1, and 3 daycares already reach 63% of the maximum.

**Total children per apartment.** A hidden effect per building and one per neighborhood stand for
what attracts families and is not measured, such as a building's prices or a neighborhood's
reputation:

$$
u_b \sim \mathcal N(\mu=0,\ \sigma^2=0.25^2),
\qquad
w_j \sim \mathcal N(\mu=0,\ \sigma^2=0.20^2)
$$

One standard deviation multiplies the expected count of every apartment below it by
$e^{0.25} \approx 1.28$ (the building) and $e^{0.20} \approx 1.22$ (the neighborhood). The
apartment's expected count $\mu_i$ and its count $N_i$:

$$
\begin{aligned}
\log \mu_i ={}& -0.89
+ 0.45\,\mathbb 1\lbrace R_i = 4\rbrace + 0.80\,\mathbb 1\lbrace R_i = 5\rbrace
+ 0.95\,\mathbb 1\lbrace R_i = 6\rbrace
+ 1.0\,L_j - 0.15\,c_i\,L_j \\
&+ 0.12\,\mathrm{sat}(D_j)
+ 0.10\,\mathbb 1\lbrace Z_j = \text{existing}\rbrace
+ 0.04\,\mathbb 1\lbrace Z_j = \text{planned}\rbrace
- 0.06\,\tilde M_j + u_b + w_j
\end{aligned}
$$

$$
N_i = \min\big(N^{*}_i,\ 10\big),
\qquad
N^{*}_i \sim \mathrm{NB2}\big(\text{mean}=\mu_i,\ \phi=3\big),
\qquad
\mathrm{Var}(N^{*}_i) = \mu_i + \frac{\mu_i^2}{3}
$$

- **Baseline.** A 3-room apartment in the reference neighborhood (household size 3.0, median age
  31) with no daycares, no school and no hidden effects expects $e^{-0.89} \approx 0.41$ children.
  Most neighborhoods have daycares and a school, so this is not a typical apartment.
- **Rooms.** At household size 3.0, 4, 5 and 6 rooms multiply a 3-room apartment's count by
  $e^{0.45} \approx 1.57$, $e^{0.80} \approx 2.23$ and $e^{0.95} \approx 2.59$. The step to 6 rooms
  is small: the largest apartments are also bought for comfort.
- **Household size** is the strongest term. With the coefficient 1.0 on $L_j$ and the other terms
  fixed, the expected count of a 4-room apartment is exactly proportional to $H_j - 1.7$: a
  household size of 4.3 doubles the reference neighborhood's count, and one of 2.35 halves it.
- **Crowding.** The product term is a fraction of the household term, so the two combine into one
  power of the household excess, $\big((H_j - 1.7)/1.3\big)^{\,1 - 0.15\,c_i}$: a power of 1.15 at
  3 rooms, 1.0 at 4, 0.85 at 5 and 0.70 at 6. Large apartments follow household size less steeply
  than small ones, in both directions. Where households are small, every apartment expects fewer
  children than at the reference, the small ones most: at household size 2.2, 0.33 times at 3
  rooms and 0.51 times at 6. The product term is positive there ($+0.29$ at 6 rooms), but it
  offsets only 30% of the household term's $-0.96$. Where households are large, every apartment
  expects more, the small ones most. The 6-to-3-room ratio is 2.59 at the reference, about 4.0 at
  household size 2.2 (families with children concentrate in the large apartments) and about 1.8
  at 4.5 (crowded families fill the small ones too).
- **Small terms.** Daycares mark an area that suits families beyond what its household size
  says: many of them multiply the count by at most $e^{0.12} \approx 1.13$. Families settle near
  schools: an existing one multiplies it by $e^{0.10} \approx 1.11$ and a planned one by
  $e^{0.04} \approx 1.04$ (against none). At a fixed household size, an older neighborhood has
  more all-adult households, such as parents with grown children: ten more years of median age
  multiply it by $e^{-0.06} \approx 0.94$.

The dispersion $\phi = 3$ makes apartments with the same expected count differ more than a Poisson
draw: somewhat more of them hold no children, and a few hold a large family. No apartment holds
more than 10 children, about the limit for one family within a 16-year age span.

There is **no direct SES term**: SES acts mainly through the household size, which carries its
reverse-J (section 2.1), and also through the other neighborhood values and the room mix. There
is **no building-size term**: inside a neighborhood, size acts only through the room mix.

**The cohort split.** The apartment's children are split over kindergarten (kg, ages 3 to 6),
elementary (el, 7 to 12) and high school (hs, 13 to 18). A second hidden effect per building,
independent of $u_b$, stands for a building filling with one generation of families, since its
apartments are often sold within a few years:

$$
v_b \sim \mathcal N(\mu=0,\ \sigma^2=0.20^2)
$$

The **life-cycle score** $\theta_i$ is higher for older children:

$$
\theta_i = 0.20\,c_i - 0.20\,\mathrm{sat}(D_j) + 0.25\,\tilde M_j + 0.05\,\tilde H_j
- 0.25\,\mathbb 1\lbrace Z_j = \text{planned}\rbrace + v_b
$$

The score is zero for a 4-room apartment in the reference neighborhood (median age 31, household
size 3.0) with no daycares, no planned school and $v_b = 0$. One standard deviation of $v_b$
shifts the score by 0.20, as much as one more room does. Families move to larger homes as their
children grow, daycares draw families with young children, and an older neighborhood is further
along the family cycle. A planned school marks a new neighborhood
that is still filling with young families. The direct household-size term leans slightly older
(a household grows as its children are born), yet neighborhoods of large households hold younger
children overall, because they are younger (section 2.1) and have smaller apartments (section
2.2).

$$
p_{i,\mathrm{kg}} \propto 0.225\,e^{-1.0\,\theta_i},
\qquad
p_{i,\mathrm{el}} \propto 0.385,
\qquad
p_{i,\mathrm{hs}} \propto 0.390\,e^{1.1\,\theta_i}
$$

At $\theta_i = 0$ the probabilities are exactly 22.5%, 38.5% and 39.0%. Elementary, the middle
cohort, is the pivot: its weight does not depend on the score, so its probability moves less than
the other two, and a higher score moves children from kindergarten toward high school. A planned school, for example, multiplies the
odds of high school against kindergarten by $e^{-(1.0 + 1.1) \cdot 0.25} \approx 0.59$. Most
neighborhoods have daycares and about a fifth have a planned school, so the average score is
negative, and of all the children about 27%, 38% and 35% fall in the three cohorts. The reference
probabilities are set to give this split. Kindergarten is the smallest cohort mainly because it
spans 4 years of age, against 6 for each of the others.

A multinomial draw gives each child a cohort, with
$\mathbf p_i = (p_{i,\mathrm{kg}},\ p_{i,\mathrm{el}},\ p_{i,\mathrm{hs}})$:

$$
(N_{i,\mathrm{kg}},\ N_{i,\mathrm{el}},\ N_{i,\mathrm{hs}}) \sim \mathrm{Multinomial}(N_i,\ \mathbf p_i)
$$

The score has no SES term (SES acts through the other
values), no existing-school term (an existing school draws families but does not set their ages)
and no building-size term (new towers draw young families, while renewal towers bring older owners
back; inside a neighborhood, size acts only through the room mix).

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
