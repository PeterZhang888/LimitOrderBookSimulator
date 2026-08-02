# Reduced QQQ basket definition

The interaction experiment is intentionally a **three-component proxy**, not a
historical replication of the complete QQQ creation basket or NAV. It uses
AAPL, MSFT, and AMZN because they are liquid, highly valued QQQ holdings and
are present in both selected ITCH sessions.

To keep the 30 December 2019 training split strictly chronological, the model
uses the QQQ schedule of investments as of 30 September 2019 filed with the SEC
on 20 December 2019. Security values are divided by reported net assets of
USD 75,056,816,837:

| Symbol | Security value (USD) | QQQ portfolio share | Three-stock share |
|---|---:|---:|---:|
| AAPL | 8,220,421,303 | 0.1095226476 | 0.3451534666 |
| MSFT | 8,621,301,880 | 0.1148636759 | 0.3619853680 |
| AMZN | 6,974,990,535 | 0.0929294743 | 0.2928611654 |

The arbitrage agent normalizes the three positive portfolio shares internally.
The same frozen values are used for training and held-out validation. The
filing predates the training session, so this structural input does not leak
future information.

Source: <https://www.sec.gov/Archives/edgar/data/1067839/000119312519320290/d813945dn30b2.htm>
