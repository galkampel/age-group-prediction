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
  how many children live there and how old they are. Larger apartments hold more children, and
  older ones.

Hidden values act at two places, and none of them is in the dataset. A building's type and two
values per neighborhood shape the building sizes (section 2.2). Two effects per building and one
per neighborhood shape the children (section 2.3): they make apartments in the same building or
neighborhood more alike than their features explain, as unmeasured local factors do in real data.

The whole draw is reproducible from one seed (42). The steps below follow the order of the draws.
Throughout, $j$ indexes neighborhoods, $b$ buildings and $i$ apartments.

### 2.1 Neighborhoods: the context

There are 150 neighborhoods, and their model is calibrated to urban neighborhoods in Israel. Five
values describe each neighborhood. The population comes first (who lives there), and the
facilities come last (what was built for them), because supply follows demand. The formulas of this
section are centered on a **reference neighborhood**: SES 0, median age 31 and household size 3.0.

**SES** is the neighborhood's socioeconomic level: how its residents' income, education and
employment compare with the rest of the country. It is a national index, so it places the
neighborhood against the whole population and not against its own city. 0 is the national average,
one unit is one standard deviation, and negative values are poorer neighborhoods.

$$
S_j = \mathrm{clip}\big(\mathcal N(\mu=0,\ \sigma^2=1^2),\ \min=-2.5,\ \max=2.5\big)
$$

SES is drawn first because nothing else in the model explains it. The other four values depend on
it, directly or through each other.

**Median age** is the median age, in years, of all the residents of neighborhood $j$, children
included. A neighborhood with many children therefore has a low median age.

$$
M_j = \mathrm{clip}\big(31 + 3.5\,S_j + \varepsilon^{M}_j,\ \min=16,\ \max=50\big),
\qquad
\varepsilon^{M}_j \sim \mathcal N(\mu=0,\ \sigma^2=4.5^2)
$$

The median age is 31 at the national-average SES and rises by 3.5 years per SES unit. In Israel
the lowest-SES neighborhoods are largely Haredi and Arab communities, where families have many
children, so the population is very young. Well-off neighborhoods have fewer children per family
and more older residents. The noise $\varepsilon^{M}_j$ lets neighborhoods of the same SES differ:
a new development full of young families and a long-established neighborhood can share an SES.

**Average household size** is the mean number of people who live in one home in the neighborhood,
adults and children together.

$$
H_j = \mathrm{clip}\big(3.0 - 0.06\,(M_j - 31) - 0.20\,S_j + 0.18\,S_j^2 + \varepsilon^{H}_j,\ \min=1.8,\ \max=6.5\big),
\qquad
\varepsilon^{H}_j \sim \mathcal N(\mu=0,\ \sigma^2=0.30^2)
$$

Two relations enter:

- **Median age.** Each year of median age above 31 removes 0.06 people from the average household.
  Children both lower the median age and enlarge the households, so a young neighborhood is a
  neighborhood of families. An old one holds couples whose children have left and people who live
  alone.
- **SES.** Among neighborhoods of the same median age, the linear and the squared SES terms make
  households largest at the lowest SES and smallest a little above the average, near SES 0.6.
  From there they grow again toward the top.

SES also acts through the median age: a higher SES means an older neighborhood, and an older
neighborhood has smaller households. Together, SES shifts the average household size by
$-0.41\,S_j + 0.18\,S_j^2$, which is lowest near SES 1.1. This is a reverse-J: a steep arm at
the bottom and a slight rise at the top. The steep arm is the large families of the Haredi and
Arab communities. The slight rise is well-off family neighborhoods, against mid-SES neighborhoods
with more singles and couples. The noise $\varepsilon^{H}_j$ lets neighborhoods of the same age
and SES differ.

**Daycares within 500 m** is the number of daycare centers within walking distance of the
neighborhood's homes.

$$
D_j \sim \mathrm{Poisson}\Big(\lambda = \exp\big[1.02 + 0.30\,(H_j - 3.0) + 0.35\,S_j - 0.03\,(M_j - 31)\big]\Big)
$$

The reference neighborhood expects $e^{1.02} \approx 2.8$ daycares. The Poisson draw gives a whole
number around the expected count, and zero is possible. Daycares open where there are small
children and where parents can pay. Each multiplier below holds with the other two values fixed:

- **Demand.** Larger households mean more small children: each additional person of household
  size multiplies the expected count by $e^{0.30} \approx 1.35$. Older neighborhoods have fewer
  toddlers: ten more years of median age multiply it by $e^{-0.3} \approx 0.74$.
- **Purchasing power.** Parents pay for daycare, and in well-off neighborhoods both parents more
  often work: each SES unit multiplies the expected count by $e^{0.35} \approx 1.42$.

The two forces work against each other at the bottom of the SES scale. Those neighborhoods have
the largest households and the least purchasing power, and on average the two roughly cancel
there.

**School status** says whether the neighborhood has a school: one exists, one is planned, or
there is none. Each status $z$ gets a weight, and its probability is its share of the three
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
where the children are: each additional person of household size multiplies the odds of an
existing school, against none, by $e^{0.8} \approx 2.23$. A planned school answers growth.
Young neighborhoods are often new ones that are still filling up, so a median age ten years
younger multiplies the odds of a planned school, against none, by the same
$e^{0.8} \approx 2.23$. SES does not enter: schools are public and are built by need, not by
what parents can pay.

The five values are therefore linked, as they are in real neighborhoods. Low-SES neighborhoods
are younger and have larger households, the daycares follow the population and its purchasing
power, and the schools follow the population. A model sees SES, median age and household size
move together, so their separate effects are harder to estimate than if they were independent.

### 2.2 Buildings: count, size and room mix

A building is described by two things: how many apartments it has, and how they split over
apartment sizes. Both follow the neighborhood. Three hidden values take part: two belong to the
neighborhood and one is the building's type. None of them is a column of the dataset, so a model
sees only their result, the building's apartment and room counts.

**Buildings per neighborhood** is the number of buildings sampled from neighborhood $j$: at least
one, and ten on average.

$$
B_j = 1 + \mathrm{Poisson}(\lambda=9)
$$

This count models nothing about the neighborhood. It only sets how much data each neighborhood
contributes, so it depends on no feature.

**Two hidden neighborhood values** come next. The tower propensity $\rho_j$ is how far the
neighborhood leans toward high-rise building, beyond what its features explain. The texture
$\tau_j$ shifts the typical size of the neighborhood's buildings on the log scale: every building
there is $e^{\tau_j}$ times larger, and one standard deviation is a factor of
$e^{0.20} \approx 1.22$.

$$
\rho_j \sim \mathcal N(\mu=0,\ \sigma^2=1.0^2),
\qquad
\tau_j \sim \mathcal N(\mu=0,\ \sigma^2=0.20^2)
$$

Towers are built by plan: a neighborhood has an urban-renewal or a new-construction plan, or it
does not. Towers therefore cluster in some neighborhoods and are absent from others that look the
same, and $\rho_j$ produces that. The buildings of one neighborhood were also mostly built in the
same period and under the same rules, so they are alike in size, and $\tau_j$ shifts all of them
together.

**Building type** says whether building $b$ is a tower ($T_b = 1$) or keeps the neighborhood's
usual mid-rise form, a texture building ($T_b = 0$). The median age enters in decades above the
reference neighborhood's 31, a scale that section 2.3 uses too:

$$
\tilde M_j = \frac{M_j - 31}{10}
$$

$$
P(T_b = 1) = \mathrm{logistic}\big(-2.35 + 0.7\,S_j + 0.3\,\tilde M_j
+ 0.7\,\mathbb 1\lbrace Z_j = \text{planned}\rbrace + \rho_j\big),
\qquad
\mathrm{logistic}(x) = \frac{1}{1 + e^{-x}}
$$

The indicator $\mathbb 1\lbrace\cdot\rbrace$ is 1 when its condition holds and 0 otherwise. In the
reference neighborhood, with no planned school and $\rho_j = 0$, a building is a tower with
probability $\mathrm{logistic}(-2.35) \approx 9\%$. Each term multiplies the odds of a tower:

- **SES.** A tower pays off only where land is expensive, and land is expensive where residents
  are well off: each SES unit multiplies the odds by $e^{0.7} \approx 2.01$.
- **Median age.** Urban renewal replaces old low buildings with towers, and it happens in
  established neighborhoods, whose residents are older: ten more years of median age multiply
  the odds by $e^{0.3} \approx 1.35$.
- **A planned school.** A planned school marks a new neighborhood that is still growing
  (section 2.1), and new neighborhoods are built high: it multiplies the odds by the same
  $e^{0.7} \approx 2.01$, against both other school statuses.
- **The hidden propensity.** One standard deviation of $\rho_j$ multiplies the odds by
  $e^{1.0} \approx 2.72$, more than one SES unit or a planned school does.

**Apartments per building** is the number of apartments in building $b$, a whole number. The type
sets the typical size $m_b$. The neighborhood's texture scales it, and so does the building's own
deviation $\varepsilon^{A}_b$, of which one standard deviation is a factor of
$e^{0.22} \approx 1.25$:

$$
A_b = \mathrm{clip}\Big(\mathrm{round}\big(m_b\,e^{\tau_j + \varepsilon^{A}_b}\big),\ \min=8,\ \max=160\Big),
\qquad
m_b = \begin{cases} 30 & \text{if } T_b = 0 \\ 90 & \text{if } T_b = 1 \end{cases},
\qquad
\varepsilon^{A}_b \sim \mathcal N(\mu=0,\ \sigma^2=0.22^2)
$$

A texture building has a median of 30 apartments, a mid-rise of about 6 to 10 floors, and a tower
has three times as many. The deviations multiply the size instead of adding to it, because
building sizes are right-skewed: most buildings are modest and a few are very large. The result is
many buildings near 30 apartments and a smaller group of towers around 90.

The size of a building therefore says something about its neighborhood. Large buildings are more
common where SES is high, where the residents are older and where a school is planned, and one
neighborhood's buildings resemble each other. The building's size is the models' exposure
(section 4), and section 7.3 returns to its importance as a feature.

**Room mix.** Each building splits its apartments over 3, 4, 5 and 6 rooms. Household size enters
on a scale of three quarters of a person above the reference neighborhood's 3.0, which section
2.3 uses too. One number, the tilt $q_b$, says how far the building leans toward large
apartments:

$$
\tilde H_j = \frac{H_j - 3.0}{0.75},
\qquad
q_b = 0.35\,S_j + 0.10\,\tilde H_j - 0.32\,\log\frac{A_b}{30}
$$

The tilt is zero for a building of 30 apartments in the reference neighborhood. Three relations
enter:

- **SES.** Apartment size is bought: well-off neighborhoods have larger apartments. Each SES unit
  adds 0.35 to the tilt, so over the SES scale this term runs from $-0.875$ to $0.875$, the
  widest range of the three.
- **Household size.** Larger households need more rooms, but the term is weak: 0.10 per three
  quarters of a person, so a household size of 4.5 adds only 0.20. In Israel the largest
  households live in the poorest neighborhoods (section 2.1), and there the negative SES term
  outweighs it. Neighborhoods of large, poor households therefore get smaller apartments, which
  is crowding, not a match of home to family.
- **Building size.** Large buildings are planned with many small units. $\log$ is the natural
  logarithm, so a building twice as large as the texture median lowers the tilt by
  $0.32 \log 2 \approx 0.22$, and a typical tower, three times as large, lowers it by
  $0.32 \log 3 \approx 0.35$: as much as one SES unit adds. Towers are more common at high SES,
  so there this term takes back a part of the SES term.

The tilt rotates the base shares of 30%, 41%, 24% and 5% for 3, 4, 5 and 6 rooms. Six-room
apartments are rare in apartment buildings.

$$
\tilde\omega_{br} = \pi_r \exp\Big(q_b\,\frac{r - 4}{2}\Big),
\qquad
(\pi_3, \pi_4, \pi_5, \pi_6) = (0.30,\ 0.41,\ 0.24,\ 0.05),
\qquad
\omega_{br} = \frac{\tilde\omega_{br}}{\sum_{r'=3}^{6} \tilde\omega_{br'}}
$$

Here $r$ is the number of rooms, and $\omega_{br}$ is the expected share of $r$-room apartments
in building $b$. Four rooms is the pivot: its multiplier is always 1. A positive tilt moves weight
from 3 rooms toward 5 and 6 rooms, and a negative tilt moves it back. With the division by 2,
$e^{q_b}$ is the multiplier of 6 rooms and $e^{-q_b/2}$ that of 3 rooms. At $q_b = 0.35$, a
building of 30 apartments at SES 1 and household size 3.0, the multipliers are 0.84, 1, 1.19 and
1.42, and the shares become 25%, 40%, 28% and 7%.

Buildings with the same expected shares still differ, as real buildings do. A Dirichlet draw
gives the building its own mix $\mathbf g_b$ around the expected shares
$\boldsymbol\omega_b = (\omega_{b3}, \omega_{b4}, \omega_{b5}, \omega_{b6})$, and a multinomial
draw turns that mix into room counts that sum exactly to $A_b$:

$$
\mathbf g_b \sim \mathrm{Dirichlet}\big(30\,\boldsymbol\omega_b\big),
\qquad
(A_{b3},\ A_{b4},\ A_{b5},\ A_{b6}) \sim \mathrm{Multinomial}(A_b,\ \mathbf g_b)
$$

$A_{br}$ is the number of $r$-room apartments in building $b$. A concentration of 30 keeps a
building near its expected shares while letting it differ; a larger value would make buildings
with the same expected shares more alike.

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
