# Soccer Data Analytics Hackathon - Getting Started

**Event Dates:** February 27–28, 2026  
**Location:** Northeastern University / Network Science Institute  
**Supported by:** PySport

## Overview

This repository provides starter code and instructions for the Soccer Data Analytics Hackathon. You'll work with **IMPECT Open Data** containing 306 German Bundesliga matches from the 2023/24 season to tackle one of two challenge prompts.

## Challenge Prompts (Choose One)

### Option A: Starting Eleven Lineup Construction
Recommend an optimal starting eleven and/or substitution plan to maximize team cohesion and ball progression. Build a player-to-player pass network, analyze network structure, and compare alternative lineups with clear visualizations.

### Option B: Transparent Player Valuation Metric
Define an interpretable attacking or defensive metric using event data. Create a metric definition, produce a leaderboard comparing players, present a case study, and discuss limitations.

## Quick Start

### 1. Installation (within Jupyter notebook)

```bash
!pip install "kloppy>=3.18.0" polars pyarrow
```

### 2. Explore the Notebook

Open `getting-started.ipynb` to see examples of:
- Loading matches and squad data
- Filtering for specific event types (passes, shots)
- Transforming coordinate systems
- Exporting to Polars/Pandas DataFrames

## Project Structure

```
.
├── getting-started.ipynb    # Tutorial notebook with data loading examples
├── README.md                # This file
├── environment.yml          # Conda environment (if using)
└── requirements.txt         # Python dependencies (if using pip)
```

## Setting up your virtual environment (if using)

### For conda:

```bash
conda env create -f environment.yml
conda activate soccer-hackathon # to activate your virtual environment
```

### For pip:

```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Deliverables (Due Friday, February 27, 2026)

1. **Slide deck** (PDF, max 8 slides) with clear visualizations
2. **GitHub repository** with:
   - Clean, reproducible code
   - `README.md` explaining your approach
   - `environment.yml` or `requirements.txt`
   - Open-source license (MIT or Apache 2.0)

## Timeline

| Milestone | Date |
|-----------|------|
| Release & Data Primer | Monday, November 3, 2025 |
| Registration Deadline | Wednesday, December 31, 2025 |
| Checkpoint (draft slides/repo) | Monday, February 2, 2026 |
| Final Work Session & Judging | Friday, February 27, 2026 |
| Industry Talks & Awards | Saturday, February 28, 2026 |

## Suggested Tools

- **Python:** kloppy, polars, pandas, numpy, mplsoccer, databallpy, networkx
- **R:** tidyverse, igraph
- Any language is acceptable as long as your work is reproducible

## Recommended Project Structure

your-team-name/
├── notebooks/          # Jupyter notebooks for exploration
├── src/               # Python scripts/modules
├── figures/           # Generated plots and visualizations
├── slides/            # Your presentation deck
├── data/              # Processed data (not raw IMPECT data)
├── README.md          # Describe your approach
├── requirements.txt   # Your dependencies
└── LICENSE            # Open-source license

## Resources

- [Kloppy Documentation](https://kloppy.pysport.org/)
- [IMPECT Open Data](https://github.com/ImpectAPI/open-data)
- [PySport](https://pysport.org/)

## License & Ethics

- IMPECT Open Data is for **non-commercial use only**
- Cite all sources appropriately
- If using AI tools, document where and how they were used
  - For example: This .README was generated with the help of Claude Sonnet 4.5
- Be transparent about limitations in your methodology

## Judging Criteria (100 points)

- Problem framing & soccer context (10 pts)
- Data engineering & correctness (15 pts)
- Methodology quality (15 pts)
- Validation & robustness (15 pts)
- Results & insight (15 pts)
- Communication & visualization (15 pts)
- Reproducibility & ethics (15 pts)

## Submission Checklist

Before submitting, ensure you have:
- [ ] Chosen one prompt (A or B)
- [ ] Created slide deck (PDF, max 8 slides)
- [ ] Pushed code to GitHub with clear README
- [ ] Included environment.yml or requirements.txt
- [ ] Added open-source license (MIT or Apache 2.0)
- [ ] Documented any AI tool usage
- [ ] Named files: `TeamName_Hackathon2026.pdf`
- [ ] Tested that code runs from a fresh environment

## Contact

Questions? Email **northeasternsportsanalytics@gmail.com**

---

**Good luck and happy hacking!** ⚽📊
