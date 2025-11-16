# Trip Sequence Dataset Processing

## Overview
This project processes the original **MOBIS trip-level dataset** into a clean, ordered sequence format suitable for modeling daily user behavior. Each row in the processed dataset represents a single trip event, aligned to a user and a specific day.

---

## Processing Steps

### **1. Select core fields**
We keep only the fields required for sequence modeling:

- `user_id`  
- `trip_mode`  
- `end_purpose`  
- `track_finished_at_swiss` (used as the event timestamp)

`start_purpose` is intentionally dropped because purpose transitions are already captured by the sequence of `end_purpose`.

---

### **2. Assign event dates (Swiss local time)**
Each trip is assigned to a local calendar day using the Switzerland timezone:

```sql
DATE(track_finished_at_swiss, "Europe/Zurich") AS event_date
```

This ensures that daily sequences follow the correct Swiss local day boundaries.

---

### **3. Order trips within each day**
Trips for each `user_id` and `event_date` are sorted by finish time.  
A sequential index is assigned:

```sql
ROW_NUMBER() OVER (
  PARTITION BY user_id, event_date
  ORDER BY track_finished_at_swiss
) AS seq_idx
```

This creates a clean chronological order.

---

## Final Dataset Format
Each row of the processed dataset contains:

| Column       | Description                                  |
|--------------|----------------------------------------------|
| `user_id`    | User identifier                               |
| `event_date` | Swiss local date of the trip                  |
| `seq_idx`    | Ordered index of the trip within the day      |
| `trip_mode`  | Transportation mode                           |
| `end_purpose`| Purpose at the end of the trip                |
| `event_time` | Timestamp (trip finish time in Swiss local)   |

**Example:**

```
user_id   event_date   seq_idx   trip_mode   end_purpose   event_time
u123      2019-11-20   1         Car         coworking     2019-11-20 08:32
u123      2019-11-20   2         Walk        home          2019-11-20 09:10
u123      2019-11-20   3         Bus         shopping      2019-11-20 11:45
```

---

## How to Use the Processed Dataset

### **1. Group by user and day**
```python
df_group = df.groupby(["user_id", "event_date"])
```

### **2. Sort within each group**
```python
df_sorted = df_group.apply(lambda x: x.sort_values("seq_idx"))
```

### **3. Extract daily sequences**
```python
daily_seq = list(zip(
    df_sorted["trip_mode"],
    df_sorted["end_purpose"]
))
```

This gives you sequences like:

```
[(Car, coworking), (Walk, home), (Bus, shopping), ...]
```

---

## Downstream Applications

- Daily **trip-purpose prediction**
- **Next-mode** or **next-purpose** modeling
- Mobility pattern mining
- Lifestyle sequence classification
- Markov models / RNN / Transformers
- Detecting missing or abnormal trips
