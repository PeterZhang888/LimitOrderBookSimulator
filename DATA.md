# Data availability

The repository does not redistribute raw Nasdaq TotalView--ITCH files or the
large per-symbol derived directories. Users must obtain data under Nasdaq's
applicable terms and run the extraction workflow themselves.

The empirical protocol used five 2019 training sessions (30 January, 27 March,
30 July, 30 October and 30 December) and 30 January 2020 as a development-
validation session. The fixed balanced panel contains 1,480 symbols.

Useful entry points are:

1. `scripts/select_itch50_universe.py`
2. `scripts/extract_itch50_symbols.py`
3. `scripts/build_queue_reactive_empirical_augmentation.py`
4. `scripts/apply_queue_reactive_empirical_augmentation.py`
5. `submit_five_day_pooled_training.sh`
6. `submit_cluster_value_agent_calibration.sh`

Never commit raw `.NASDAQ_ITCH50`, `.gz`, per-symbol empirical directories, or
cluster filesystem paths containing usernames. Generated scientific artifacts
should be content-hashed and documented separately from source code.

