# equity-research

Factor research framework for US equities — value, quality, and momentum signals with proper backtesting infrastructure. Currently in Phase 1: data layer (universe, prices, fundamentals).

## Setup

```bash
git clone <repo-url>
cd equity-research
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
python -m equity_research.db
```
