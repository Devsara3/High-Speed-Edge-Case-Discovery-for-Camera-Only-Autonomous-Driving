# Meta-Learner Models Directory

This directory is intended to store the pre-trained neural network weights (`.pkl` files) for dynamic sensor fusion.

## Naming Convention
The simulation script dynamically loads the appropriate model based on the running scenario. Please follow this naming convention:

- `fusion_meta_learner_A.pkl` (For Scenario A: CPNA)
- `fusion_meta_learner_B.pkl` (For Scenario B: CCRb)
- `fusion_meta_learner_C.pkl` (For Scenario C: CCFtap)
- `fusion_meta_learner.pkl` (Fallback / Universal model)

Place your generated models directly into this folder.
