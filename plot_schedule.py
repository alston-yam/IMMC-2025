# this is what the schedule looks like
# 4	2	2	5	4	5
# 0	1	0	1	5	5
# 0	1	4	0	4	1
# 5	4	3	3	4	5
# 0	0	5	3	3	5

import folium

team = 1
names = [
    "Brazil",  # 0
    "Spain",  # 1
    "France",  # 2
    "Argentina",  # 3
    "Uruguay",  # 4
    "Colombia",  # 5
    "United Kingdom",
    "Paraguay",
    "Germany",
    "Ecuador",
    "Portugal",
    "Italy",
    "Morocco",
    "Egypt",
    "South Korea",
    "Japan",
    "Mexico",
    "Costa Rica",
    "New Zealand",
    "Australia",
]

elo_ratings = [
    1994,
    2150,
    2031,
    2140,
    1922,
    1953,
    2012,
    1799,
    1988,
    1911,
    1988,
    1914,
    1807,
    1668,
    1745,
    1875,
    1817,
    1653,
    1596,
    1736,
]

locations = [
    (-14.2350, -51.9253),
    (40.4637, -3.7492),
    (46.6034, 1.8883),
    (-38.4161, -63.6167),
    (-32.5228, -55.7659),
    (4.5709, -74.2973),
    (55.3781, -3.4360),
    (-23.4420, -58.4438),
    (51.1657, 10.4515),
    (-1.8312, -78.1834),
    (39.3999, -8.2245),
    (41.8719, 12.5674),
    (31.7915, -7.0926),
    (26.8206, 30.8025),
    (35.9078, 127.7669),
    (36.2048, 138.2529),
    (23.6345, -102.5528),
    (9.7489, -83.7534),
    (-40.9006, 174.8860),
    (-25.2744, 133.7751),
]

with open("solution.txt", "r") as f:
    lines = f.readlines()
    schedule = []
    for line in lines:
        if line.strip():
            schedule.append(list(map(int, line.split())))

for row in schedule:
    print("\t".join(map(str, row)))

number_of_teams = len(schedule[0])
# plot the teams on the map
m = folium.Map(location=[0, 0], zoom_start=2)
for i in range(number_of_teams):
    folium.Marker(
        location=locations[i],
        popup=f"{names[i]} [{elo_ratings[i]}]",
        icon=folium.Icon(color="blue", icon="info-sign"),
    ).add_to(m)
m.save("map.html")

# plot the schedule on the map with lines connecting countries

colors = [
    "blue",
    "green",
    "red",
    "purple",
    "orange",
    "darkred",
    "lightred",
    "beige",
    "darkblue",
]

folium.PolyLine(
    (locations[team], locations[schedule[0][team]]),
    color=colors[team],
    weight=2.5,
    opacity=1,
).add_to(m)

folium.PolyLine(
    (locations[team], locations[schedule[-1][team]]),
    color=colors[team],
    weight=2.5,
    opacity=1,
).add_to(m)

for r in range(len(schedule) - 1):
    folium.PolyLine(
        (locations[schedule[r][team]], locations[schedule[r + 1][team]]),
        color=colors[team],
        weight=2.5,
        opacity=1,
    ).add_to(m)
m.save("map_with_lines.html")
