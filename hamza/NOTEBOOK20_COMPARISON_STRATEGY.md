# Notebook 20: Comparison Strategy

**Created**: 2026-01-12
**Purpose**: Structured approach to regional/encoding comparisons

---

## Core Hypothesis

**NB19/19b found NO spline advantage at global scale with SH(L=10).**

**NB20 tests**: Does scale (global → regional → local) or encoding dimensionality matter?

---

## Comparison Dimensions

### Dimension 1: Spatial Scale
- **Global**: Full planet, 15K samples (NB19 baseline)
- **Continental**: 5 continents, 5K samples each
- **Regional**: 100×100 km patches, 2-5K samples
- **Local**: 10×10 km patches, 500-1K samples

### Dimension 2: Terrain Type
- **Mountainous**: High relief, sharp peaks (Himalayas, Rockies, Andes)
- **Flat**: Low relief, smooth (Sahara, Great Plains, European Plain)
- **Mixed**: Moderate relief, varied (Europe, coastal regions)

### Dimension 3: SH Encoding Level
- **L=10**: 121 dims (current baseline from NB19)
- **L=20**: 441 dims (4× more features)
- **L=40**: 1681 dims (14× more features)

### Dimension 4: Spatial Resolution
- **Coarse**: 30 km spacing (~1000 samples/region)
- **Medium**: 2 km spacing (~2500 samples/region)
- **Fine**: 1 km spacing (~10,000 samples/region)

### Dimension 5: Task Type
- **Smooth**: Population density, temperature
- **Sharp**: Coastline distance, land cover boundaries
- **Mixed**: Elevation (multi-scale structure)

---

## Experimental Design

### Experiment 1: Continental Comparison Matrix

**Goal**: Test if terrain type matters more than we thought

**Design**: 5 continents × 3 activations × 1 encoding

| Region | Terrain | ReLU R² | Spline R² | Advantage | Hypothesis |
|--------|---------|---------|-----------|-----------|------------|
| North America | Mixed | ? | ? | ? | Rockies → spline |
| Europe | Mixed | ? | ? | ? | Alps → spline |
| Asia (Himalayas) | **Mountain** | ? | ? | ? | **Strongest spline** |
| Africa (Sahara) | **Flat** | ? | ? | ? | **ReLU wins** |
| S. America (Andes) | **Mountain** | ? | ? | ? | **Strong spline** |

**Success Criteria**:
- Mountainous regions show >1% spline advantage
- Flat regions show ReLU advantage or tie
- Clear terrain-dependent pattern

**Failure Mode**:
- All regions similar (like global result)
- No consistent terrain effect

---

### Experiment 2: SH Encoding Sensitivity Matrix

**Goal**: Test if encoding dimensionality changes spline vs ReLU comparison

**Design**: 3 SH levels × 3 activations × 2 terrains (mountain vs flat)

| SH Level | Terrain | ReLU R² | Spline R² | Advantage | Hypothesis |
|----------|---------|---------|-----------|-----------|------------|
| L=10 (121) | Mountain | ? | ? | ? | Baseline (NB19) |
| L=20 (441) | Mountain | ? | ? | ? | Moderate spline gain |
| L=40 (1681) | Mountain | ? | ? | ? | **Largest spline gain** |
| L=10 (121) | Flat | ? | ? | ? | ReLU wins |
| L=20 (441) | Flat | ? | ? | ? | Still ReLU |
| L=40 (1681) | Flat | ? | ? | ? | Still ReLU |

**Success Criteria**:
- Clear interaction: Higher L → larger spline advantage
- Effect stronger in mountainous regions
- At some L, spline advantage emerges (>2%)

**Failure Mode**:
- No SH level shows spline advantage
- Pattern inconsistent across terrains

**Key Insight This Tests**:
- If L=40 + Spline still loses → problem isn't encoding dimensionality
- If L=40 + Spline wins → SH(L=10) was over-smoothing

---

### Experiment 3: Multi-Resolution Within Region

**Goal**: Test resolution hypothesis at regional scale (vs global in NB19)

**Design**: 1 region × 3 resolutions × 3 activations

**Region**: North America (Rockies) - high relief, varied terrain

| Resolution | Grid Size | ReLU R² | Spline R² | Advantage | Hypothesis |
|------------|-----------|---------|-----------|-----------|------------|
| Coarse (30km) | 100×100 km → 10×10 | ? | ? | ? | ReLU (smooth) |
| Medium (2km) | 100×100 km → 50×50 | ? | ? | ? | Small spline |
| Fine (1km) | 100×100 km → 100×100 | ? | ? | ? | **Strong spline** |

**Comparison to NB19 Exp 3**:
- NB19: Global scale, ReLU won at fine (-0.47%)
- NB20: Regional scale, test if pattern reverses

**Success Criteria**:
- Fine resolution shows >1% spline advantage at regional scale
- Trend: coarse → medium → fine, increasing spline benefit

**Failure Mode**:
- Same as NB19: ReLU wins at fine resolution
- Confirms scale doesn't matter, result is fundamental

---

### Experiment 4: Urban vs Rural Typology

**Goal**: Test if human-modified landscapes (sharp boundaries) favor splines

**Design**: 3 density types × 3 activations × 1 encoding

| Density Type | Pop/km² | ReLU R² | Spline R² | Advantage | Hypothesis |
|--------------|---------|---------|-----------|-----------|------------|
| Dense Urban | >1000 | ? | ? | ? | **Strong spline** (sharp) |
| Suburban | 100-1000 | ? | ? | ? | Moderate spline |
| Rural | <100 | ? | ? | ? | ReLU wins (smooth) |

**Success Criteria**:
- Urban shows >2% spline advantage (sharp building boundaries)
- Rural shows ReLU advantage (smooth agricultural)
- Clear monotonic trend with density

**Failure Mode**:
- All density types similar
- No urban/rural distinction

---

### Experiment 5: Boundary Task Comparison

**Goal**: Test truly high-frequency tasks (step functions, sharp edges)

**Design**: 3 tasks × 3 activations × 1 encoding

| Task | Frequency | ReLU R² | Spline R² | Advantage | Hypothesis |
|------|-----------|---------|-----------|-----------|------------|
| Coastline Distance | **Step function** | ? | ? | ? | **Largest spline** |
| Elevation Gradient | Sharp edges | ? | ? | ? | Strong spline |
| Land Cover Binary | Sharp boundaries | ? | ? | ? | Strong spline |

**Success Criteria**:
- Coastline (true step function) shows >5% spline advantage
- At least 2/3 tasks show >2% advantage
- Validates "high-frequency = spline" hypothesis

**Failure Mode**:
- Even step functions show no advantage
- Would suggest splines fundamentally unsuited to geographic data

---

## Analysis Framework

### For Each Comparison

**1. Point Estimate**:
- ReLU R², Spline R², SIREN R²
- Absolute difference (Spline - ReLU)
- Percent difference: 100 × (Spline - ReLU) / ReLU

**2. Training Efficiency**:
- Training time (seconds)
- Params (should be similar)
- Time per epoch

**3. Patterns**:
- Does advantage increase/decrease systematically?
- Interactions between dimensions?

**4. Significance**:
- If effect >1%, worth validating with multiple seeds
- If effect <0.5%, likely noise

---

## Decision Tree

```
Start: NB20 Experiment 1 (Continental)
│
├─ If Mountain > Flat by >1%
│  ├─ PROCEED: Test Exp 2 (SH levels)
│  │  └─ If L=40 > L=10 significantly
│  │     ├─ PROCEED: Complete all experiments
│  │     └─ FINDING: "SH level matters, L=40 needed for mountains"
│  │  └─ If L=40 ≈ L=10
│  │     └─ FINDING: "Terrain matters but not encoding"
│  │
│  └─ PROCEED: Test Exp 3-5 for full picture
│
├─ If Mountain ≈ Flat (all <0.5%)
│  ├─ PROCEED: Test Exp 2 anyway (encoding hypothesis)
│  │  └─ If L=40 shows advantage
│  │     └─ PROCEED: Maybe scale + encoding interaction
│  │  └─ If L=40 also fails
│  │     └─ STOP: Strong evidence SH+ReLU is optimal
│  │        └─ Skip Exp 3-5, focus on mechanistic analysis (NB21)
│
└─ UNEXPECTED: Flat > Mountain
   └─ INVESTIGATE: Why? Data quality? Overfitting?
      └─ Check error patterns, visualize (NB21)
```

---

## Success Scenarios

### Scenario A: Terrain-Dependent (Best Case)
- Mountainous regions: Spline wins by >2%
- Flat regions: ReLU wins or ties
- **Publication**: "Learned Activations for High-Relief Geographic Prediction"
- **Actionable**: Use splines for mountains, ReLU for plains

### Scenario B: Encoding-Dependent
- L=10: ReLU wins (NB19 result)
- L=40: Spline wins by >2%
- **Publication**: "High-Dimensional Encoding Enables Learned Activations"
- **Actionable**: Use L=40+Spline for complex tasks

### Scenario C: Scale-Dependent
- Global: ReLU wins (NB19 result)
- Regional/Local: Spline wins by >2%
- **Publication**: "Scale Matters: Regional Analysis Reveals Activation Benefits"
- **Actionable**: Use splines for local predictions, ReLU for global

### Scenario D: Task-Dependent
- Smooth tasks: ReLU wins
- Boundary tasks: Spline wins by >5%
- **Publication**: "When Simplicity Bias Fails: Boundary Detection with Learned Activations"
- **Actionable**: Use splines for step functions, ReLU otherwise

### Scenario E: No Advantage Found (Negative Result)
- All experiments show ReLU ≥ Spline
- Consistent across scale, terrain, encoding, tasks
- **Publication**: "Spherical Harmonic Encoding Obviates Learned Activations for Geographic Data"
- **Actionable**: Always use SH+ReLU for geographic prediction

---

## Computational Budget

### Per Experiment Estimates

**Exp 1 (Continental)**:
- 5 regions × 3 activations × 100 epochs = 15 runs
- ~90 min/run = **~23 hours total**
- Can parallelize across regions

**Exp 2 (SH Levels)**:
- 3 SH levels × 2 terrains × 3 activations = 18 runs
- L=10: ~90 min, L=20: ~120 min, L=40: ~180 min
- **~40 hours total**

**Exp 3 (Resolution)**:
- 3 resolutions × 3 activations = 9 runs
- Fine resolution slower: ~120 min/run
- **~18 hours total**

**Exp 4 (Urban/Rural)**:
- 3 density types × 3 activations = 9 runs
- ~90 min/run = **~14 hours total**

**Exp 5 (Boundaries)**:
- 3 tasks × 3 activations = 9 runs
- ~90 min/run = **~14 hours total**

**Total**: ~109 hours (can split across sessions, run in parallel)

---

## Reporting Template

For each experiment, report:

### Summary Table
```
| Configuration | ReLU R² | Spline R² | Δ (abs) | Δ (%) | Time (s) | Verdict |
|---------------|---------|-----------|---------|-------|----------|---------|
| ...           | ...     | ...       | ...     | ...   | ...      | ...     |
```

### Key Finding
```
🎯 [EXPERIMENT NAME]

Hypothesis: [What we predicted]
Result: [What we found]
Verdict: ✅ Confirmed / ❌ Rejected / ⚠️ Partial

[2-3 sentence interpretation]
```

### Decision
```
Next Action:
- [ ] Proceed to next experiment
- [ ] Deep dive into this finding
- [ ] Stop - hypothesis rejected
```

---

## Publication Strategy (Post-NB20)

### If Positive Results (Any Scenario A-D)
- **Title**: "When and Where Learned Activations Excel: A Regional Analysis of Geographic Prediction"
- **Key Figure**: Heatmap of Spline advantage by (scale × terrain × encoding)
- **Contribution**: Actionable guidance on activation selection

### If Negative Results (Scenario E)
- **Title**: "Spherical Harmonic Encoding Obviates Learned Activations for Geographic Prediction"
- **Key Figure**: Systematic comparison across 5 experiments, all showing ReLU ≥ Spline
- **Contribution**: Strong negative result, practical guidance

### Either Way
- Comprehensive evaluation on real geographic data
- Tests paper's predictions in novel domain
- Clear practical implications

---

## Next Steps

1. ✅ Data sources documented
2. ✅ Comparison strategy defined
3. ⏳ Download priority data (SRTM tiles)
4. ⏳ Create NB20 Jupyter notebook
5. ⏳ Run Exp 1 (Continental) first
6. ⏳ Use decision tree to guide remaining experiments
7. ⏳ Update analysis documents with findings

**Ready to implement after user confirms approach.**
