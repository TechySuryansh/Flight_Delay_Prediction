# Flight Delay Prediction: Data Dictionary

| Feature Name | Data Type | Available at Prediction Time | Description |
| :--- | :--- | :--- | :--- |
| **`departure_hour`** | Integer (0-23) | Yes (Scheduled) | Hour of scheduled departure extracted from CRSDepTime. |
| **`departure_minute`** | Integer (0-59) | Yes (Scheduled) | Minute of scheduled departure. |
| **`dep_hour_sin`** | Float [-1, 1] | Yes (Scheduled) | Cyclic sine encoding of scheduled departure hour: sin(2*pi*hour/24). |
| **`dep_hour_cos`** | Float [-1, 1] | Yes (Scheduled) | Cyclic cosine encoding of scheduled departure hour: cos(2*pi*hour/24). |
| **`dep_minute_of_day_sin`** | Float [-1, 1] | Yes (Scheduled) | Cyclic sine encoding of minute-of-day: sin(2*pi*min/1440). |
| **`dep_minute_of_day_cos`** | Float [-1, 1] | Yes (Scheduled) | Cyclic cosine encoding of minute-of-day: cos(2*pi*min/1440). |
| **`day_of_week`** | Integer (1-7) | Yes (Calendar) | Day of week (1=Monday, 7=Sunday). |
| **`is_weekend`** | Binary (0 or 1) | Yes (Calendar) | Flag indicating if flight is scheduled on Saturday or Sunday. |
| **`day_of_week_sin`** | Float [-1, 1] | Yes (Calendar) | Cyclic sine encoding of day of week: sin(2*pi*dow/7). |
| **`day_of_week_cos`** | Float [-1, 1] | Yes (Calendar) | Cyclic cosine encoding of day of week: cos(2*pi*dow/7). |
| **`month`** | Integer (1-12) | Yes (Calendar) | Scheduled month of flight. |
| **`month_sin`** | Float [-1, 1] | Yes (Calendar) | Cyclic sine encoding of month: sin(2*pi*month/12). |
| **`month_cos`** | Float [-1, 1] | Yes (Calendar) | Cyclic cosine encoding of month: cos(2*pi*month/12). |
| **`distance`** | Float / Miles | Yes (Planned) | Flight distance between origin and destination airports. |
| **`scheduled_elapsed_time`** | Float / Minutes | Yes (Planned) | Scheduled planned flight duration from gate to gate. |
| **`airline_delay_prior`** | Float [0, 1] | Yes (Historical prior from training set only) | Smoothed historical delay rate of the carrier calculated exclusively on earlier training flights. |
| **`origin_delay_prior`** | Float [0, 1] | Yes (Historical prior from training set only) | Smoothed historical delay rate of the origin airport calculated exclusively on earlier training flights. |
| **`destination_delay_prior`** | Float [0, 1] | Yes (Historical prior from training set only) | Smoothed historical delay rate of the destination airport calculated exclusively on earlier training flights. |
| **`route_delay_prior`** | Float [0, 1] | Yes (Historical prior from training set only) | Smoothed historical delay rate of the route calculated exclusively on earlier training flights. |
| **`airline`** | Categorical String | Yes (Scheduled) | Operating airline code (e.g. AA, DL, UA, WN). One-hot encoded. |
| **`origin`** | Categorical String | Yes (Scheduled) | Origin airport IATA code (e.g. ORD, ATL, DFW). One-hot encoded. |
| **`destination`** | Categorical String | Yes (Scheduled) | Destination airport IATA code (e.g. LAX, JFK, SFO). One-hot encoded. |
| **`route`** | Categorical String | Yes (Scheduled) | Route origin_dest string (e.g. ORD_LAX). High cardinality. |
| **`is_delayed_15`** | Binary (0 or 1) | NO (Target variable) | Supervised learning target: 1 if flight arrival delay >= 15 min, 0 otherwise. |