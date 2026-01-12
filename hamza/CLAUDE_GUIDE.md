# Claude Guide: Analyzing New Experimental Results

**Purpose**: Standardized process for ingesting and analyzing completed notebooks
**Last Updated**: 2026-01-12

This guide provides a step-by-step workflow for analyzing new experimental results and updating project documentation. Follow this process whenever a new notebook is completed.

---

## Quick Start Checklist

When user says "notebook X is complete":

- [ ] Read the notebook and extract results
- [ ] Create analysis document (ANALYSIS_NOTEBOOKX.md)
- [ ] Update EXPERIMENTS_TRACKER.md
- [ ] Update README.md
- [ ] Update EXPERIMENTAL_TIMELINE.md (if comprehensive)
- [ ] Create visualizations if needed
- [ ] Commit changes with descriptive message

---

## Step 1: Read and Extract Results

### 1.1 Read the Completed Notebook

```
Read the notebook file: hamza/XX_notebook_name.ipynb
- Focus on cells with results (look for print statements, dataframes, plots)
- Extract R² scores, parameter counts, training times
- Note any experimental configurations tested
- Identify key findings and unexpected results
```

### 1.2 Key Information to Extract

For EVERY experiment, capture:

**Setup**:
- Date completed
- Environment (CPU/GPU, Colab/local)
- Data used (GPW, resolution, sample size)
- Input encoding (raw coords, SH L=10/40, etc.)
- Architecture (layers, hidden dim, activation type)
- Training config (epochs, batch_size, lr, optimizer)

**Results** (create tables):
- Model name
- R² score (test set)
- Parameter count
- Training time
- Any special metrics (gradient norms, convergence, etc.)

**Comparisons**:
- Performance vs baselines (SIREN, ReLU, SatCLIP)
- Absolute differences (e.g., +0.02 R²)
- Relative differences (e.g., +2.5%)

**Findings**:
- What worked well (✅)
- What failed (❌)
- Unexpected results (⚠️)
- Root causes identified

---

## Step 2: Create Analysis Document

### 2.1 Analysis Document Template

Create `ANALYSIS_NOTEBOOKX.md` with this structure:

```markdown
# Analysis: Notebook X - [Title]

**Date**: YYYY-MM-DD
**Status**: ✅ Complete
**Purpose**: [One sentence description]

---

## Executive Summary

[2-3 paragraphs summarizing the entire notebook]

### 🏆 Key Results

- Optimal configuration: ...
- Performance vs baselines: ...
- Verdict: ...

---

## Experiment 1: [Name]

**Goal**: [What question does this answer?]

### Results

[Table of results]

### Analysis

[2-3 paragraphs analyzing the results]
- What patterns emerged?
- Why did X perform better than Y?
- What does this tell us?

### Conclusion

[1 paragraph takeaway]

---

[Repeat for each experiment]

---

## Key Findings Summary

### 1. [Finding Name]

[Detailed explanation]

### 2. [Finding Name]

[Detailed explanation]

---

## Implications for Phase N

### What This Means

[How do these results affect the broader project?]

### Next Steps

[What should be done next based on these results?]

---

## Experimental Details

[Full setup details for reproducibility]

---

## Conclusions & Recommendations

### For Practitioners

[Actionable advice]

### For Researchers

[Theoretical insights]

### For This Project

[Internal next steps]

---

## Files Generated

[List of CSVs, PNGs, etc.]

---

## References

[Links to related notebooks and documents]

**Analysis complete**: YYYY-MM-DD
```

### 2.2 Analysis Writing Guidelines

**Be Specific**:
- ✅ "SH+Spline: 0.7354 (+1.88% vs SIREN, -0.63% vs ReLU)"
- ❌ "Spline performed reasonably well"

**Use Visual Hierarchy**:
- ✅ for successes
- ❌ for failures
- ⚠️ for unexpected/concerning results
- 📊 for data insights
- 💡 for key insights

**Include Tables**:
- Every major result should have a table
- Sort by performance (best first)
- Include vs baseline columns
- Add status column (Winner/Good/OK/Failed)

**Explain WHY**:
- Don't just report results
- Explain potential reasons
- Connect to theory/previous findings

---

## Step 3: Update EXPERIMENTS_TRACKER.md

### 3.1 Add Completed Experiment Section

Find the appropriate phase section and update:

```markdown
### Notebook X: [Title] ✅

**Date**: ~YYYY-MM
**Purpose**: [Brief description]
**Status**: ✅ Complete

#### Setup
- Data: [dataset, resolution]
- Samples: [train/test split]
- Architecture: [layers × width]
- Training: [epochs, lr, etc.]

#### Experiments Run

1. [Experiment name] - [brief result]
2. [Experiment name] - [brief result]
...

#### Key Results

| Model | R² | vs Baseline | Status |
|-------|-----|-------------|--------|
| ... | ... | ... | ... |

#### Major Findings

✅ [Success 1]
✅ [Success 2]
❌ [Failure 1]

#### Takeaway

[One paragraph summary]

#### Documentation

[ANALYSIS_NOTEBOOKX.md](ANALYSIS_NOTEBOOKX.md)
```

### 3.2 Update Summary Statistics

At the bottom of EXPERIMENTS_TRACKER.md:

```markdown
### Phase N Progress (In Progress)
- ✅ X/Y notebooks completed (list them)
- ✅ Z+ experiments run (update count)
- ✅ ~W hours GPU time (update)
- ✅ [Major milestones]
```

---

## Step 4: Update README.md

### 4.1 Update Notebook Table

Find the notebook table and change status:

```markdown
| **X** | ✅ Complete | [Title] | [Key result in 5-10 words] | [ANALYSIS_NOTEBOOKX.md](ANALYSIS_NOTEBOOKX.md) |
```

### 4.2 Update Key Metrics

If this is a major milestone, update the metrics section:

```markdown
### Phase N Progress (In Progress)
- ✅ X/Y notebooks completed (X-Y)
- ✅ Z+ experiments run
- ✅ ~W hours GPU time
```

### 4.3 Update Executive Summary (if major finding)

If the notebook has a major finding that changes the narrative:

```markdown
## Executive Summary: Complete Experimental Arc

[Update the phase summary with new findings]
```

---

## Step 5: Update EXPERIMENTAL_TIMELINE.md

### 5.1 Add Notebook Section

Add a new section in chronological order:

```markdown
## Notebook X: [Title] ✅

**Date**: ~YYYY-MM
**Purpose**: [Description]
**Status**: ✅ Complete

### Key Results

[Results table]

### Finding

✅ **[Main finding]**
- [Supporting point 1]
- [Supporting point 2]

### Baselines Established (if any)

[List any baselines this notebook establishes]

---
```

### 5.2 Update Timeline Overview (if needed)

```markdown
## Timeline Overview

```
[Update the ASCII tree if this notebook represents a phase transition]
```

---

## Step 6: Create Visualizations (Optional)

### 6.1 When to Create Visualizations

Create additional visualizations if:
- Notebook has many experiments (>5)
- Results show clear trends
- Comparisons are complex
- User specifically requests

### 6.2 Visualization Types

**Performance Comparison**:
```python
# Bar chart comparing models
models = ['Model A', 'Model B', ...]
r2_scores = [0.75, 0.72, ...]
plt.bar(models, r2_scores)
plt.axhline(baseline, linestyle='--', label='Baseline')
```

**Trend Analysis**:
```python
# Line plot for sweeps (knot count, learning rate, etc.)
plt.plot(x_values, y_values, 'o-')
plt.xlabel('Parameter')
plt.ylabel('R²')
```

**Heatmap** (for 2D sweeps):
```python
# Heatmap for depth × width, lr × batch_size, etc.
plt.imshow(results_matrix, cmap='viridis')
```

---

## Step 7: Integration Checklist

Before considering the analysis complete:

### Documentation Updates
- [ ] ANALYSIS_NOTEBOOKX.md created
- [ ] EXPERIMENTS_TRACKER.md updated
- [ ] README.md notebook table updated
- [ ] EXPERIMENTAL_TIMELINE.md updated (if comprehensive notebook)
- [ ] PROGRESS_UPDATE.md updated (if phase complete)

### Content Quality
- [ ] All R² scores extracted and tabulated
- [ ] Comparisons to baselines calculated (both absolute and %)
- [ ] Key findings clearly stated with ✅/❌/⚠️
- [ ] Implications for future work identified
- [ ] Tables formatted properly
- [ ] Markdown renders correctly

### Cross-References
- [ ] Links to related notebooks added
- [ ] References to previous findings included
- [ ] Next steps aligned with EXPERIMENTAL_DESIGN_V2.md

### User Communication
- [ ] Summarize findings in message to user
- [ ] Highlight unexpected results
- [ ] Recommend next steps
- [ ] Ask clarifying questions if results are unclear

---

## Example Workflow

### User Message: "Notebook 18 is complete, can you analyze it?"

**Step 1**: Read notebook
```
Read hamza/18_spline_deep_dive.ipynb
Extract: 6 experiments, 20 models trained, R² scores, optimal config
```

**Step 2**: Create analysis
```
Write ANALYSIS_NOTEBOOK18.md with:
- Executive summary (optimal k=15, relu init, fixed positions)
- 6 experiment sections (one per experiment)
- Key findings (ReLU still wins, but Spline beats SIREN)
- Implications (test on high-frequency tasks next)
```

**Step 3**: Update tracker
```
Edit EXPERIMENTS_TRACKER.md:
- Add "Notebook 18: Spline Deep Dive ✅" section
- Include results table
- Update Phase 2 progress (1/6 complete)
```

**Step 4**: Update README
```
Edit README.md:
- Change "🔄 Created" to "✅ Complete" for NB18
- Add link to ANALYSIS_NOTEBOOK18.md
- Update metrics (1 notebook complete, 20 experiments)
```

**Step 5**: Update timeline
```
Edit EXPERIMENTAL_TIMELINE.md:
- Add "## Notebook 18: Spline Deep Dive ✅" section
- Include key results and findings
```

**Step 6**: Summarize to user
```
"Notebook 18 analysis complete! Key findings:
- Optimal: k=15 knots, relu init, (-3,3) range, fixed positions
- ReLU still wins (0.7417), but Spline beats SIREN (0.7354 vs 0.7219)
- Zero init catastrophically fails (R²=-0.001)
- Ready for high-frequency tasks (NB20) next

All documentation updated:
- ANALYSIS_NOTEBOOK18.md (comprehensive analysis)
- EXPERIMENTS_TRACKER.md (NB18 section added)
- README.md (status updated)
- EXPERIMENTAL_TIMELINE.md (chronology updated)"
```

---

## Special Cases

### Case 1: Notebook Has Unexpected Failures

If experiments completely fail or show negative R²:

1. ⚠️ Flag this prominently in analysis
2. Investigate potential causes:
   - Data issues? (check sample sizes, distributions)
   - Architecture issues? (check initialization, gradients)
   - Training issues? (check loss curves, convergence)
3. Include "Diagnostic Analysis" section
4. Recommend follow-up experiments

### Case 2: Notebook Has No Clear Winner

If results are all very similar:

1. 📊 Focus on trends rather than absolute winners
2. Include error bars or variance if available
3. Discuss practical vs statistical significance
4. Recommend robustness testing (multiple seeds)

### Case 3: Notebook Contradicts Previous Results

If results differ from previous notebooks:

1. ⚠️ Flag the discrepancy clearly
2. Compare experimental setups:
   - Different random seeds?
   - Different data splits?
   - Different hyperparameters?
3. Include "Reconciliation" section
4. List possible explanations
5. Recommend follow-up to resolve

### Case 4: Notebook is Incomplete or Broken

If notebook has errors or is partially run:

1. ❌ Mark as "⚠️ Incomplete" in tracker
2. Document what was completed
3. Note what failed and why
4. Recommend fixes
5. Don't create full analysis document (wait for completion)

---

## Common Mistakes to Avoid

### ❌ Don't Do This

1. **Skip the analysis document**
   - Every notebook deserves analysis, even simple ones

2. **Only report numbers without interpretation**
   - Always explain WHY results turned out this way

3. **Forget to update all documents**
   - EXPERIMENTS_TRACKER.md is the source of truth - must be updated

4. **Use vague language**
   - "Performed well" → Specify R², comparisons, improvements

5. **Ignore failures**
   - Failed experiments are as important as successes

6. **Create inconsistent numbering**
   - Always check existing notebook numbers before adding new ones

7. **Forget cross-references**
   - Link related notebooks, reference previous findings

### ✅ Do This

1. **Be thorough but concise**
   - Executive summary: 2-3 paragraphs
   - Each experiment: 1-2 pages
   - Total: 5-10 pages depending on complexity

2. **Use consistent formatting**
   - Follow the template structure
   - Use tables for results
   - Use headers properly (##, ###, ####)

3. **Think about the reader**
   - Someone should be able to understand the notebook from the analysis alone
   - Include enough context for outsiders

4. **Highlight actionable insights**
   - What should practitioners do differently?
   - What should researchers investigate next?

5. **Update cross-references**
   - Link to previous related work
   - Reference the experimental design doc
   - Connect to broader project goals

---

## Templates

### Quick Email-Style Update (for user)

```
Notebook [X] analysis complete!

Key findings:
- ✅ [Success 1]
- ✅ [Success 2]
- ❌ [Failure 1]

Performance:
- [Winner]: [R²] ([comparison to baseline])
- [Runner-up]: [R²] ([comparison])

Implications:
- [What this means for the project]

Next steps:
- [Recommended next notebook or experiment]

All documentation updated:
- ANALYSIS_NOTEBOOK[X].md (full analysis)
- EXPERIMENTS_TRACKER.md (results logged)
- README.md (status updated)
```

### Commit Message Template

```
Add Notebook [X] analysis: [Brief Title]

- Completed [N] experiments testing [what]
- Key result: [main finding]
- [Winner] achieves [R²] ([comparison])
- [Critical insight if any]

Files:
- ANALYSIS_NOTEBOOK[X].md (full analysis)
- EXPERIMENTS_TRACKER.md (updated)
- README.md (updated)
- [any CSVs/visualizations]
```

---

## Troubleshooting

### Problem: Can't find the R² scores in notebook

**Solution**:
- Look for print statements with "R²" or "r2"
- Check for `r2_score()` function calls
- Look in dataframes (often in columns named 'r2' or 'test_r2')
- Check the summary cells at the end

### Problem: Results tables don't render properly

**Solution**:
- Ensure tables use `|` separators
- Include header separator row (|---|---|)
- Check for extra/missing `|` characters
- Use markdown table generators if needed

### Problem: Notebook has multiple random seeds/runs

**Solution**:
- Report mean ± std if available
- Create separate table for statistical analysis
- Note seed variance in limitations
- Include best run in main results table

### Problem: Unsure what baseline to compare against

**Solution**:
- Check previous notebooks for established baselines
- Use SIREN for SH features (standard in this project)
- Use SatCLIP L=10 for overall comparison
- If introducing new task/data, establish baseline first

---

## Advanced: Automated Analysis Script

For future automation, key steps would be:

```python
def analyze_notebook(notebook_path):
    """
    Automated analysis pipeline for completed notebooks.
    """
    # 1. Parse notebook
    cells = parse_ipynb(notebook_path)

    # 2. Extract results
    results = extract_results(cells)
    # - Find dataframes with 'r2', 'model', 'params'
    # - Extract print statements with metrics
    # - Parse markdown cells for setup info

    # 3. Generate analysis
    analysis = generate_analysis_md(results)
    # - Create executive summary
    # - Format result tables
    # - Compare to baselines
    # - Identify winners/losers

    # 4. Update tracker
    update_experiments_tracker(results)

    # 5. Update README
    update_readme(notebook_path, results)

    return analysis
```

This is a future enhancement - for now, follow manual process.

---

## Final Checklist

Before considering analysis complete:

### Content
- [ ] All experiments from notebook documented
- [ ] All R² scores extracted and tabulated
- [ ] Comparisons to baselines calculated
- [ ] Key findings identified with ✅/❌
- [ ] Implications for future work stated
- [ ] Recommendations provided

### Documentation
- [ ] ANALYSIS_NOTEBOOKX.md created
- [ ] EXPERIMENTS_TRACKER.md updated
- [ ] README.md updated
- [ ] EXPERIMENTAL_TIMELINE.md updated (if major)
- [ ] Cross-references added

### Quality
- [ ] Tables formatted correctly
- [ ] Markdown renders properly
- [ ] No broken links
- [ ] Consistent with project style
- [ ] Proofread for clarity

### Communication
- [ ] User notified of completion
- [ ] Key findings summarized
- [ ] Next steps recommended
- [ ] Questions asked if needed

---

## Conclusion

This guide provides a standardized process for analyzing experimental results. Following this workflow ensures:

✅ **Consistency** across all analyses
✅ **Completeness** - nothing gets missed
✅ **Quality** - thorough and professional documentation
✅ **Efficiency** - clear process, no wasted time
✅ **Continuity** - future Claude instances can continue seamlessly

When in doubt, refer to existing analyses (ANALYSIS_NOTEBOOK15.md, ANALYSIS_NOTEBOOK16.md, ANALYSIS_NOTEBOOK17.md, ANALYSIS_NOTEBOOK18.md) as examples.

---

**Last Updated**: 2026-01-12
**Maintained by**: Project contributors
**Questions?**: Check existing analyses or ask user for clarification
