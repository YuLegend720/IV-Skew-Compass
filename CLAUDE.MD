# IV-Skew-Compass - Claude Context

## Project Overview

IV-Skew-Compass is an automated options analysis tool that visualizes market sentiment through two key metrics: **Risk Premium** and **Risk Reversal**. The system fetches live options data from Yahoo Finance, calculates volatility metrics, and generates a compass visualization showing where each stock sits in the risk/sentiment landscape.

**Purpose**: Help traders identify options trading opportunities by mapping:
- Whether options are expensive or cheap (Risk Premium = IV - HV)
- Whether the market is bullish or bearish (Risk Reversal = Call IV - Put IV)

## Architecture

### Core Components

1. **`fetch_options_data.py`** - Main data fetcher
   - Fetches options chains from yfinance API
   - Calculates implied volatility (IV) and historical volatility (HV)
   - Computes Risk Premium and Risk Reversal metrics
   - Outputs `compass-data.json`

2. **`compass-data.json`** - Generated data file
   - Contains metrics for all tracked symbols
   - Updated automatically via GitHub Actions
   - Used by frontend visualization

3. **`compass-auto.html`** - Frontend visualization (not in repo scope)
   - Interactive compass chart
   - Plots symbols by Risk Premium (X-axis) and Risk Reversal (Y-axis)

4. **`.github/workflows/update-data.yml`** - Automation
   - Runs Monday-Friday at 21:30 UTC (4:30 PM EST)
   - Can be triggered manually via workflow_dispatch
   - Commits and pushes updated data automatically

### Tracked Symbols

The system monitors 22 symbols across different sectors:
- **Tech**: AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, AMD, INTC
- **Finance**: JPM, BAC, WFC
- **Entertainment**: DIS, NFLX
- **ETFs**: SPY, QQQ, IWM, TLT
- **Growth/Crypto**: COIN, PLTR, SOFI, UBER

## Key Metrics Explained

### 1. Risk Premium (IV - HV)
- **Formula**: Current Implied Volatility - 20-day Historical Volatility
- **Interpretation**:
  - **Positive**: Options are expensive (market paying premium for protection/speculation)
  - **Negative**: Options are cheap (no premium being paid)
  - **High values**: Good for selling options (premium collection)
  - **Low values**: Good for buying options (cheap protection/speculation)

### 2. Risk Reversal (Call IV - Put IV)
- **Formula**: IV of 25Δ Call - IV of 25Δ Put (approximated as 10% OTM)
- **Interpretation**:
  - **Positive**: Call skew (bullish sentiment, calls more expensive)
  - **Negative**: Put skew (bearish sentiment, puts more expensive)
  - **Neutral (near 0)**: Balanced sentiment

### 3. IV/HV Ratio
- **Formula**: Current IV / Historical Volatility
- **Interpretation**:
  - **> 1.5**: Options relatively expensive
  - **< 1.0**: Options relatively cheap
  - **Useful for**: Identifying mean reversion opportunities

## Compass Quadrants

The visualization creates four quadrants:

- **Q1 (↗)**: High Premium + Call Skew → Expected bullish move, expensive options
- **Q2 (↖)**: Low Premium + Call Skew → Bullish opportunity, cheap options
- **Q3 (↙)**: Low Premium + Put Skew → Cheap protection available
- **Q4 (↘)**: High Premium + Put Skew → Market panic, expensive protection

## Development Guidelines

### Working with the Python Script

**Key Functions**:
- `calculate_realized_volatility()`: Computes 20-day HV from price history
- `get_atm_implied_volatility()`: Fetches ATM IV from options chain (25-60 DTE)
- `calculate_risk_premium()`: Simple subtraction (IV - HV)
- `calculate_risk_reversal()`: Compares OTM call vs put IV
- `fetch_symbol_data()`: Main orchestrator for each symbol

**Important Notes**:
- ATM options are defined as strikes within ±5% of current price
- 25Δ options are approximated as 10% OTM
- Target expiration: 25-60 days out (falls back to nearest if none available)
- HV calculation uses last 20 trading days
- All volatility values are annualized percentages

### Adding New Symbols

To add symbols to tracking:
1. Edit `SYMBOLS` list in `fetch_options_data.py` (line 18)
2. Ensure symbol has liquid options (volume > 100 typically)
3. Test locally first: `python fetch_options_data.py`
4. Commit and push - automation will handle the rest

### Modifying Metrics

**To change HV period**:
- Modify `HV_PERIOD` constant (line 25)
- Standard periods: 10, 20, 30 days

**To adjust OTM delta approximation**:
- Modify multipliers in `calculate_risk_reversal()` (lines 200-201)
- Current: 1.10 for calls, 0.90 for puts (≈25Δ)

**To change ATM range**:
- Modify filters in `get_atm_implied_volatility()` (lines 111-118)
- Current: ±5% of spot price

### Error Handling

The script handles these scenarios gracefully:
- Missing options data → Symbol skipped with warning
- Insufficient price history → Symbol skipped
- Invalid IV values → Symbol skipped
- API rate limits → Caught and logged

**Debugging Tips**:
- Check symbol output for specific failures
- Verify options liquidity on Yahoo Finance
- Ensure symbol has options with 25-60 DTE
- Check that yfinance package is up to date

## GitHub Actions Workflow

### Automatic Updates

The workflow runs automatically at market close (4:30 PM EST) on trading days:
1. Checks out repository
2. Installs Python 3.11 + dependencies (yfinance, pandas, numpy)
3. Runs `fetch_options_data.py`
4. Commits updated `compass-data.json`
5. Pushes to main branch with retry logic

### Manual Triggering

Can be triggered manually from GitHub Actions tab:
1. Go to repository → Actions
2. Select "Update Compass Data" workflow
3. Click "Run workflow"

### Troubleshooting Workflow Failures

**Common Issues**:
- API rate limits → Wait and retry
- Network timeouts → Automatic retry (3 attempts)
- Merge conflicts → Workflow pulls latest before pushing
- No data fetched → Check yfinance API status

## Testing Locally

### Setup
```bash
# Install dependencies
pip install yfinance pandas numpy

# Run script
python fetch_options_data.py
```

### Expected Output
```
COMPASS DATA FETCHER V2 - PURE METRICS
Procesando AAPL...
  ✓ AAPL:
    IV=26.78%, HV=11.44%
    Risk Premium=+15.34% (IV-HV)
    Risk Reversal=-3.48% (C-P)
...
✓ Archivo generado: compass-data.json
```

### Validation
- Check `compass-data.json` is created
- Verify `symbols_count` matches expected
- Ensure all metrics are reasonable (IV: 10-100%, HV: 5-80%)
- Check timestamp is current

## Common Tasks

### Update Symbol List
```bash
# Edit symbols
vim fetch_options_data.py  # Modify SYMBOLS list

# Test locally
python fetch_options_data.py

# Commit
git add fetch_options_data.py
git commit -m "Add/remove symbols from tracking"
git push
```

### Adjust Calculation Parameters
```bash
# Edit constants
vim fetch_options_data.py  # Modify HV_PERIOD or OTM multipliers

# Test
python fetch_options_data.py

# Verify output makes sense
cat compass-data.json | jq '.data[] | select(.symbol=="AAPL")'
```

### Investigate Symbol Issues
```bash
# Run script and check specific symbol output
python fetch_options_data.py 2>&1 | grep "SYMBOL_NAME"

# Check if options exist
python -c "import yfinance as yf; print(yf.Ticker('SYMBOL').options)"
```

## Data Schema

### compass-data.json Structure
```json
{
  "last_updated": "ISO8601 timestamp",
  "symbols_count": 22,
  "metrics": { /* metric descriptions */ },
  "data": [
    {
      "symbol": "AAPL",
      "risk_premium": 15.34,     // IV - HV
      "risk_reversal": -3.48,    // Call IV - Put IV
      "current_iv": 26.78,       // ATM implied vol
      "hv_20": 11.44,            // 20-day historical vol
      "iv_hv_ratio": 2.34        // IV / HV
    }
  ]
}
```

## Project Constraints

### API Limitations
- **yfinance**: Unofficial Yahoo Finance API wrapper
  - May be rate limited during heavy use
  - Data quality depends on Yahoo Finance
  - Options data may be delayed 15-20 minutes

### Data Accuracy
- IV calculations depend on Yahoo Finance's options data
- HV uses closing prices only (not intraday)
- 25Δ options approximated (not exact delta calculation)
- No adjustment for dividends or carry costs

### Market Hours
- Options data most reliable during market hours (9:30 AM - 4:00 PM EST)
- Post-market updates may have stale data
- Weekend/holiday runs will use Friday's close

## Future Enhancements

Potential improvements to consider:
- Add historical tracking (store time series data)
- Implement exact delta calculations using Black-Scholes
- Add more sophisticated filtering (volume, open interest)
- Support for custom symbol watchlists
- Alert system for extreme metrics
- Comparison to historical percentiles
- Support for futures options (ES, NQ)

## Security Considerations

- No API keys required (yfinance is free)
- GitHub Actions has write permissions (needed for auto-commit)
- No sensitive data stored
- All code runs in sandboxed GitHub Actions environment

## Dependencies

### Python Packages
- **yfinance** (^0.2.0): Yahoo Finance data fetcher
- **pandas** (^2.0.0): Data manipulation
- **numpy** (^1.24.0): Numerical calculations

### Runtime
- Python 3.11+ recommended
- Works on Linux, macOS, Windows
- Requires internet connection for API access

## Maintenance Notes

### Regular Checks
- Verify workflow runs successfully (check Actions tab)
- Monitor for yfinance API changes
- Review symbols for liquidity issues
- Check data quality periodically

### When Things Break
1. Check yfinance package updates: `pip install --upgrade yfinance`
2. Verify Yahoo Finance hasn't changed their API
3. Check GitHub Actions logs for specific errors
4. Test locally to isolate issue: workflow vs script

---

**Last Updated**: 2026-01-13
**Maintainer**: Project owner via Claude Code
**License**: Not specified (check repository)
