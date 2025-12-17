# Regret Calculation Problem Summary

## Problem Statement
**Current Status:** Only 11.4% successful regret calculations (9 out of 79), expected 100%

## Root Cause Analysis

### 1. **Constraint Rejection Issues**
The model's predicted placements are being rejected by constraints in `decode_predicted_placement()`, resulting in empty `placement_pred` dictionaries. This prevents regret calculation.

### 2. **Three Possible Rejection Reasons**

#### A. Uniqueness Constraint Violation
- **What:** Two or more tasks are predicted to use the same `(node_id, platform_id)` replica
- **Where:** Lines 566-582 in `14-12-17-52.py`
- **Impact:** Placement is rejected, returns empty dict
- **Expected:** Should not happen if model learns correctly, but may occur during early training

#### B. Missing node_id in Predicted Placement
- **What:** `node_id` is `None` for one or more tasks in the predicted placement
- **Where:** Lines 560-562 in `14-12-17-52.py` - lookup from `_plat_id_to_node_id`
- **Impact:** Cannot build combo tuple for hash table lookup
- **Possible Causes:**
  1. Graph cache doesn't have `_plat_id_to_node_id` mapping (needs regeneration)
  2. Platform ID doesn't exist in the mapping
  3. Mapping lookup fails for some reason

#### C. Placement Not in Hash Table
- **What:** Predicted placement combo doesn't exist in `placement_rtt_hash_table`
- **Where:** Lines 584-607 in `14-12-17-52.py`
- **Impact:** Placement rejected even though it's valid
- **Possible Causes:**
  1. Combo format mismatch (task ID ordering, tuple format)
  2. Dataset ID mismatch
  3. Placement genuinely wasn't simulated (shouldn't happen if validation script is correct)

### 3. **Current Debugging Status**

#### ✅ Fixed Issues:
1. **Graph cache regeneration:** Added `_plat_id_to_node_id` mapping to graphs
2. **Uniqueness constraint:** Implemented same logic as `executecosimulation.py`
3. **Hash table constraint:** Added check to only allow simulated placements
4. **Validation script:** Confirmed all 108 combinations are in `placements.jsonl`

#### ❌ Still Failing:
1. **Low success rate:** Only 11.4% successful regret calculations
2. **Generic error messages:** Current logs don't show specific rejection reason
3. **Missing diagnostics:** Can't tell if it's uniqueness, missing node_id, or hash lookup failure

## Next Steps for Debugging

### 1. **Enhanced Logging** (Just Added)
- Added detailed rejection reason logging in `decode_predicted_placement()`
- Shows specific violation type (uniqueness, missing node_id, hash lookup failure)
- Includes combo format details for comparison

### 2. **Verify Graph Cache**
Run this check:
```python
# Check if graphs have _plat_id_to_node_id
graphs, _ = load_graphs_from_cache()
missing = sum(1 for g in graphs if not hasattr(g, '_plat_id_to_node_id'))
print(f"Graphs missing mapping: {missing}/{len(graphs)}")
```

### 3. **Check Combo Format Matching**
The hash table stores combos as:
```python
((node_id_0, platform_id_0), (node_id_1, platform_id_1), ...)
```
Sorted by task ID (0, 1, 2, 3, 4).

Verify predictions build the same format.

### 4. **Test with Known Good Placement**
Pick a placement from `placements.jsonl` and verify:
- Can decode it correctly
- Can build combo correctly
- Can find it in hash table

## Expected Behavior After Fix

1. **All valid predictions should pass constraints:**
   - Uniqueness: No two tasks share same replica
   - Hash lookup: Placement exists in hash table
   - node_id: All tasks have valid node_id

2. **100% successful regret calculations** for predictions that:
   - Pass uniqueness constraint
   - Exist in hash table
   - Have valid node_id for all tasks

3. **Clear error messages** showing why predictions fail (if any)

## Files Modified

1. **`src/notebooks/new_new/14-12-17-52.py`:**
   - Added uniqueness constraint check (lines 566-582)
   - Added hash table constraint check (lines 584-607)
   - Added detailed rejection logging (lines 609-640)
   - Added node_id lookup from `_plat_id_to_node_id` (lines 560-562)

2. **`src/notebooks/new_new/prepare_graphs_cache.py`:**
   - Added `plat_id_to_node_id` mapping creation (line 400)
   - Added mapping to graph data (line 708)

3. **`scripts_analysis/validate_placements_against_state.py`:**
   - Added completeness check
   - Confirmed all 108 combinations are present

## Key Questions to Answer

1. **Why are 88.6% of predictions being rejected?**
   - Is it uniqueness violations? (model predicting same replica for multiple tasks)
   - Is it missing node_id? (cache issue)
   - Is it hash lookup failure? (format mismatch)

2. **Is the model learning correctly?**
   - If uniqueness violations are common, model may need more training
   - If hash lookup failures are common, there's a format mismatch

3. **Is the cache up to date?**
   - Need to verify `_plat_id_to_node_id` exists in all graphs
   - May need to regenerate cache

## Immediate Action Items

1. ✅ **Enhanced logging added** - will show specific rejection reasons
2. ⏳ **Run training with new logs** - see what's actually failing
3. ⏳ **Verify cache has `_plat_id_to_node_id`** - check all graphs
4. ⏳ **Compare combo formats** - ensure hash table and predictions match


