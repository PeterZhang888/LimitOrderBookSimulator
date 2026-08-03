# Superseded shared-dealer mechanism audit

This directory preserves a diagnostic campaign that must not be reported as
the final liquidity-shock result. Its purpose is to document why the shared
dealer mechanism was corrected and why the production launcher now fails
closed before starting the financial matrix.

## Data-completeness finding

The campaign contained all 40 expected full-session paths. Every raw path had
its market-wide metric series, target mask and per-book summary. All submitted
shock quantities executed. Thus the flat lower panels and zero entries in the
old tables were not caused by missing output files or a plotting error.

Across all 18,000 recorded post-shock, seed, capacity and time combinations,
the global and independent-capacity non-target spread and depth differences
were exactly zero. All 100 seed-by-cluster records and all 20 cluster summaries
were likewise zero.

## Mechanism failure

Under global capacity 25, the quote scale first became permanently zero between
75 and 248 seconds across seeds. Under capacity 100, permanent zero occurred
between 1,099 and 2,011 seconds. The intervention occurred at 11,700 seconds.
The requested shared quote depth was therefore zero at the shock in every
globally constrained path, and the global shared dealer absorbed no shock
volume. The independent-capacity dealer remained active and absorbed a
positive quantity in every shocked path.

The one-second targeted-book spread response provides an independent check.
Its seed means were 11.796 bps at global capacity 25, 12.345 bps at global
capacity 100 and 12.422 bps with the shared dealer absent, whereas the active
independent-capacity dealer produced only 0.062 bps. Thus both nominal global
treatments behaved like the no-dealer control, rather than like an active
shared-liquidity treatment.

The former quote rule multiplied both risk-increasing and risk-reducing sides
by the global scale. Once the scale reached zero, the dealer could not submit
an inventory-reducing order; zero became an absorbing state. The matrix
therefore did not expose the global treatment to an active shared dealer and
could not identify cross-book transmission.

## Corrective action

The production rule now suppresses only the risk-increasing quote side.
Inventory-reducing quotes remain active and are capped by the outstanding
inventory. The simulator records requested and actually resting dealer depth,
separating risk-increasing and risk-reducing sides. A full-horizon paired
preflight must certify positive resting dealer activity, positive
risk-increasing resting depth, shock execution and positive dealer absorption
before any financial path is run.

The rejected runs also showed gross exposure near 148,000 shares before the
shock. With 1,480 books, 25- and 100-unit capacity levels were already binding
in ordinary no-shock paths. The corrected protocol therefore uses 200 and 400
units per book and requires a 60-second pre-shock median quote scale of at
least 0.30. This prevents a numerical participation floor from being mistaken
for a substantively active capacity treatment.

The archived plots are retained only as diagnostics of the rejected mechanism.
They are not referenced by the thesis chapter.
