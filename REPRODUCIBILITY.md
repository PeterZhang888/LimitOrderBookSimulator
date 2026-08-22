# Reproducibility

All thesis experiments use the same simulator source and build. Experiment
directories change only runtime configuration and Slurm resources.

The frozen empirical runtime inputs are included under `data/empirical/` and
are addressed by repository-relative paths. Every submission writes an
`environment.txt` file containing the Seagull node type, compiler and OpenMPI
versions, CPU description, allocation and loaded modules. Use seven full
repetitions for performance tables unless the experiment README states
otherwise. Do not delete an unfavourable repetition; if the predefined
stability rule fails, rerun the complete matched block.

Generated results, Slurm logs, build directories and raw exchange messages do
not belong in Git. Keep them in the project result area or an archival deposit.
