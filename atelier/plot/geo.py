from typing import Tuple

import folium
from folium.plugins import HeatMapWithTime
from IPython import display
from pandas import DataFrame


# show dataset on a map
def heatmap_plot(
    dataframe: DataFrame,
    *,
    group_column: str = "departure_time",
    # Rennes GPS coordinates
    location: Tuple[float, float] = (48.1147, -1.6794),
) -> None:
    _dataframe = dataframe.copy(deep=True)
    timestamps = []
    coordinates = []
    for timestamp, coordinate in _dataframe.groupby(group_column):
        timestamps.append(str(timestamp))
        coordinates.append(
            coordinate[
                [
                    "stop_lat",
                    "stop_lon",
                ]
            ].values.tolist()
        )

    base_map = folium.Map(
        location=location,
        zoom_start=11,
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        # tiles="https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png",
        attr="CartoDB",
    )

    heat_map = HeatMapWithTime(
        data=coordinates,
        index=timestamps,
        auto_play=True,
        min_speed=1,
        radius=4,
        max_opacity=0.5,
    )

    heat_map.add_to(base_map)
    display.display(base_map)
