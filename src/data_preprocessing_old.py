import pandas as pd
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

print("Base directory:", BASE_DIR)
print("Raw data directory:", RAW_DATA_DIR)
print("Processed data directory:", PROCESSED_DATA_DIR)


print("\nFiles inside telangana_gtfs:")
print(list((RAW_DATA_DIR / "telangana_gtfs").iterdir()))

# --- Load Telangana GTFS files ---

gtfs_dir = RAW_DATA_DIR / "telangana_gtfs"

routes = pd.read_csv(gtfs_dir / "routes.txt")
trips = pd.read_csv(gtfs_dir / "trips.txt")
stop_times = pd.read_csv(gtfs_dir / "stop_times.txt")
calendar = pd.read_csv(gtfs_dir / "calendar.txt")
stops = pd.read_csv(gtfs_dir / "stops.txt")

print("\nLoaded Telangana GTFS files successfully")
print("Routes:", routes.shape)
print("Trips:", trips.shape)
print("Stop times:", stop_times.shape)
print("Calendar:", calendar.shape)
print("Stops:", stops.shape)

# --- Feature Engineering: Extract hour of day ---

# GTFS times can exceed 24 (e.g., 25:30:00), so mod 24
stop_times["arrival_hour"] = (
    stop_times["arrival_time"]
    .str.split(":")
    .str[0]
    .astype(int)
    % 24
)

print("\nArrival hour extracted")
print(stop_times[["arrival_time", "arrival_hour"]].head())

# --- Join stop_times with trips to get route_id ---

stop_trips = stop_times.merge(
    trips[["trip_id", "route_id"]],
    on="trip_id",
    how="left"
)

print("\nstop_trips shape:", stop_trips.shape)
print(stop_trips[["trip_id", "route_id", "arrival_hour"]].head())

# --- Trips per route per hour ---

trips_per_route_hour = (
    stop_trips
    .groupby(["route_id", "arrival_hour"])
    .size()
    .reset_index(name="num_trips")
)

print("\nTrips per route per hour:")
print(trips_per_route_hour.head())

# --- Number of stops per route ---

# Join stop_times with trips to get route_id for each stop
route_stops = stop_times.merge(
    trips[["trip_id", "route_id"]],
    on="trip_id",
    how="left"
)

# Count unique stops per route
stops_per_route = (
    route_stops
    .groupby("route_id")["stop_id"]
    .nunique()
    .reset_index(name="num_stops")
)

print("\nStops per route:")
print(stops_per_route.head())

# --- Weekday / Weekend flag ---

# If any weekday is active, treat as weekday service
calendar["is_weekday"] = calendar[
    ["monday", "tuesday", "wednesday", "thursday", "friday"]
].any(axis=1).astype(int)

calendar["is_weekend"] = calendar[
    ["saturday", "sunday"]
].any(axis=1).astype(int)

print("\nCalendar flags:")
print(calendar[["service_id", "is_weekday", "is_weekend"]])

# --- Build final GTFS feature table ---

gtfs_features = trips_per_route_hour.merge(
    stops_per_route,
    on="route_id",
    how="left"
)

# Since calendar has 1 row, broadcast weekday flag
gtfs_features["is_weekday"] = calendar["is_weekday"].iloc[0]
gtfs_features["is_weekend"] = calendar["is_weekend"].iloc[0]

print("\nFinal GTFS feature table:")
print(gtfs_features.head())
print("\nGTFS features shape:", gtfs_features.shape)



# =========================================================
# Load Traffic (GTFS Traffic Prediction) Dataset
# =========================================================

traffic_path = RAW_DATA_DIR / "traffic_gtfs" / "traffic_data.csv"
traffic = pd.read_csv(traffic_path)

print("\nLoaded Traffic dataset")
print("Traffic shape:", traffic.shape)
print("Traffic columns:", traffic.columns.tolist())


# =========================================================
# Extract arrival hour from arrival_time
# =========================================================

traffic["arrival_hour"] = pd.to_datetime(
    traffic["arrival_time"],
    format="%H:%M:%S",
    errors="coerce"
).dt.hour

# Drop rows where hour could not be extracted
traffic = traffic.dropna(subset=["arrival_hour"])
traffic["arrival_hour"] = traffic["arrival_hour"].astype(int)

print("\nTraffic arrival hour extracted")
print(traffic[["arrival_time", "arrival_hour"]].head())


# =========================================================
# Map congestion text → numeric values
# =========================================================

congestion_map = {
    "Very smooth": 0,
    "Smooth": 1,
    "Mild congestion": 2,
    "Heavy congestion": 3
}

traffic["congestion_level"] = traffic["Degree_of_congestion"].map(congestion_map)

print("\nMapped congestion levels")
print(
    traffic[["Degree_of_congestion", "congestion_level"]]
    .drop_duplicates()
)


# =========================================================
# Extract route_id from traffic trip_id
# =========================================================

traffic["route_id"] = traffic["trip_id"].str.extract(r'NORMAL_(\d+)')
traffic["route_id"] = traffic["route_id"].astype(str)

# Drop rows where route_id could not be extracted
traffic = traffic.dropna(subset=["route_id"])

print("\nExtracted route_id from traffic trip_id")
print(traffic[["trip_id", "route_id"]].head())


# =========================================================
# Clean numeric columns before aggregation
# =========================================================

traffic["speed"] = pd.to_numeric(traffic["speed"], errors="coerce")
traffic["SRI"] = pd.to_numeric(traffic["SRI"], errors="coerce")

# Drop rows with invalid numeric values
traffic = traffic.dropna(
    subset=["speed", "SRI", "congestion_level", "arrival_hour"]
)

print("\nTraffic cleaned")
print(
    traffic[["route_id", "arrival_hour", "speed", "SRI", "congestion_level"]]
    .head()
)


# =========================================================
# Aggregate traffic per route per hour
# =========================================================

traffic_features = (
    traffic.groupby(["route_id", "arrival_hour"])
    .agg(
        avg_speed=("speed", "mean"),
        avg_sri=("SRI", "mean"),
        avg_congestion=("congestion_level", "mean")
    )
    .reset_index()
)

print("\nTraffic features per route per hour")
print(traffic_features.head())
print("Traffic features shape:", traffic_features.shape)

# =========================================================
# Final merge: GTFS + Traffic features
# =========================================================

final_features = gtfs_features.merge(
    traffic_features,
    on=["route_id", "arrival_hour"],
    how="left"
)

print("\nFinal merged feature table:")
print(final_features.head())
print("Final features shape:", final_features.shape)
