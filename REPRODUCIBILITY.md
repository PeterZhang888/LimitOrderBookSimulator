# Reproducibility

All thesis experiments use the same simulator source and build. Experiment
directories change only runtime configuration and Slurm resources.

Before submission, record the Seagull node type, compiler version, OpenMPI
version, `_OPENMP` value, rank placement, thread placement and exported input
paths in the result directory. Use seven full repetitions for performance
tables unless the experiment README states otherwise. Do not delete an
unfavourable repetition; if the predefined stability rule fails, rerun the
complete matched block.

Generated results, Slurm logs, build directories and licensed raw data do not
belong in Git. Keep them in the project result area or an archival deposit.
