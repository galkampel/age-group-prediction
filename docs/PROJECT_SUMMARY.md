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
