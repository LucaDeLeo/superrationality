# Cooperation Rate Formatting Bug Fix

## Summary
Fixed a data formatting bug that caused cooperation rates to be displayed incorrectly in strategy prompts, leading to confusion among AI agents during experiments.

## The Bug
- **Location**: `src/core/prompts.py` line 157
- **Issue**: Used `.1f%` format instead of `.1%`
- **Effect**: Cooperation rate of 1.0 (100%) was displayed as "1.0%" instead of "100.0%"

## Impact
This bug caused significant confusion in the experiment:
- Agents received conflicting information:
  - Reported: "Average cooperation rate: 1.0%"
  - Observed: All games showing "COOPERATE/COOPERATE"
- Led to divergent interpretations in Round 2-4
- Some agents considered defection due to the "low" cooperation rate
- Despite confusion, agents maintained cooperation based on observed behavior

## The Fix
Changed the format string from:
```python
round_text += f"\n  - Average cooperation rate: {summary.cooperation_rate:.1f}%"
```

To:
```python
round_text += f"\n  - Average cooperation rate: {summary.cooperation_rate:.1%}"
```

## Technical Details
- Python's `.1%` format multiplies by 100 and adds the % sign
- Python's `.1f%` format just adds the % sign without multiplication
- cooperation_rate is stored as a float between 0 and 1
- Other parts of the codebase correctly used `.1%` format

## Files Modified
1. `src/core/prompts.py` - Fixed the main bug
2. `test_integration_anonymization.py` - Updated test output formatting

## Verification
Created test script `test_cooperation_rate_fix.py` that verifies:
- cooperation_rate=1.0 → "100.0%"
- cooperation_rate=0.5 → "50.0%"
- cooperation_rate=0.01 → "1.0%"

## Lessons Learned
1. **Consistent formatting**: Always use `.1%` for percentage display of 0-1 floats
2. **Data presentation matters**: Even small formatting errors can cause significant confusion
3. **Robustness**: The agents showed remarkable resilience by trusting observed behavior over misleading statistics