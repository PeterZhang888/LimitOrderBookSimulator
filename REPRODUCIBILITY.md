# Reproducibility

All thesis experiments use the same simulator source and build. Experiment
directories change only runtime configuration and Slurm resources.

The frozen empirical runtime inputs are included under `data/empirical/` and
are addressed by repository-relative paths. Every submission writes an
`environment.txt` file containing the Seagull node type, compiler and OpenMPI
versions, CPU description, allocation and loaded modules. Use seven full
repetitions for performance tables unless the experiment README states
otherwise. Do not delete an unfavourable repetition. All completed repetitions
are retained. A maximum-to-minimum timing ratio above 1.15 is reported as a
variability warning and must be disclosed with the median, minimum and maximum;
it does not remove a run or trigger a replacement block.

Generated results, Slurm logs, build directories and raw exchange messages do
not belong in Git. Keep them in the project result area or an archival deposit.
