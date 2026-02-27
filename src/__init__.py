"""
Soccer analytics pipeline: figure out the best starting XI using Graph Attention Networks.

How it works (big picture):
  1. Pull match event data from IMPECT's open Bundesliga dataset
  2. Turn each ball possession into a graph (players = nodes, passes/actions = edges)
  3. Train a GAT to predict how "good" each possession was (did it lead to a shot? goal?)
  4. Use those predictions to build a player synergy matrix (who plays well together?)
  5. Solve for the optimal starting XI using integer programming (MILP)

Run it with:  python run_all_matches.py
"""
