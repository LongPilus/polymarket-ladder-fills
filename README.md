# polymarket-ladder-fills

Fetches, from Polygon's free public endpoints, the on-chain fill events (OrderFilled,
OrdersMatched, FeeRefunded) of the transactions of Polymarket's daily Bitcoin ladder
trades, and pushes them to a Hugging Face dataset. The events name the maker of every
order a taker filled, which Polymarket's trade API does not.

Runs as a GitHub Actions workflow (`.github/workflows/fetch.yml`): parallel jobs, each
fetching its share of 20,000-transaction chunks and committing them to the dataset.
Chunks already in the dataset are skipped, so rerunning the workflow resumes. Details in
`fetch_fills.py`.

Only code lives here; the transaction list and the output are in the (private) dataset.
