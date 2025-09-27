import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import requests
import os
from io import BytesIO

warnings.filterwarnings("ignore")

# Page config
st.set_page_config(
    page_title="Weather & Boulder Dashboard",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Style settings
plt.style.use("default")
sns.set_palette("husl")


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data_from_github():
    """Load data from GitHub repository or local files"""
    try:
        # Try to load local files first (for local development and Streamlit Cloud)
        if os.path.exists("swiss_weather_forecast.parquet"):
            weather_df = pd.read_parquet("swiss_weather_forecast.parquet")
            boulder_df = pd.read_parquet("bouldering_areas.parquet")
            sport_df = pd.read_parquet("sportclimbing_areas.parquet")

            # Try to load climbing recommendations if it exists
            if os.path.exists("climbing_recommendations.csv"):
                climbing_recommendations = pd.read_csv("climbing_recommendations.csv")
            else:
                climbing_recommendations = pd.DataFrame()

            return weather_df, boulder_df, sport_df, climbing_recommendations

        # Fallback: Load from GitHub (for deployment on external servers)
        base_url = "https://raw.githubusercontent.com/steffeol/outdoor_weather_data/main/"

        # Load weather data
        weather_url = base_url + "swiss_weather_forecast.parquet"
        weather_response = requests.get(weather_url)
        weather_df = pd.read_parquet(BytesIO(weather_response.content))

        # Load boulder data
        boulder_url = base_url + "bouldering_areas.parquet"
        boulder_response = requests.get(boulder_url)
        boulder_df = pd.read_parquet(BytesIO(boulder_response.content))

        # Load sport climbing data
        sport_url = base_url + "sportclimbing_areas.parquet"
        sport_response = requests.get(sport_url)
        sport_df = pd.read_parquet(BytesIO(sport_response.content))

        # Load climbing recommendations
        try:
            recommendations_url = base_url + "climbing_recommendations.csv"
            recommendations_response = requests.get(recommendations_url)
            climbing_recommendations = pd.read_csv(
                BytesIO(recommendations_response.content)
            )
        except:
            climbing_recommendations = pd.DataFrame()

        return weather_df, boulder_df, sport_df, climbing_recommendations

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None


def normalize_location_name(name):
    """Normalize location names for better readability"""
    name_clean = (
        name.replace("Switzerland", "")
        .replace("Italy", "")
        .replace("Austria", "")
        .replace("France", "")
        .strip()
    )
    return name_clean if name_clean else name


def parse_routes_in_db(value):
    """Parse 'ROUTES IN DB' values with spaces as thousand separators"""
    if pd.isna(value):
        return 0
    return int(str(value).replace(" ", ""))


def parse_stars(value):
    """Parse STARS values to float"""
    if pd.isna(value):
        return 0.0
    return float(str(value))


def match_climbing_location(climbing_name, location_list):
    """Try to match climbing areas to weather locations"""
    climbing_name_clean = str(climbing_name).strip().lower()

    # Exact matches first
    for location in location_list:
        if location.lower() == climbing_name_clean:
            return location

    # Substring matches
    for location in location_list:
        if (
            location.lower() in climbing_name_clean
            or climbing_name_clean in location.lower()
        ):
            return location

    return None


@st.cache_data
def process_data(weather_df, boulder_df, sport_df, climbing_recommendations):
    """Process and combine all data"""

    # Normalize location names in weather data
    weather_df["location_normalized"] = weather_df["location_name"].apply(
        normalize_location_name
    )
    available_locations = sorted(weather_df["location_normalized"].unique())

    # Process climbing data
    all_climbing_stats = []

    # Boulder data
    if not boulder_df.empty:
        boulder_df["routes_parsed"] = boulder_df["ROUTES IN DB"].apply(
            parse_routes_in_db
        )
        boulder_df["stars_parsed"] = boulder_df["STARS"].apply(parse_stars)
        boulder_df["location_normalized"] = boulder_df["NAME"].apply(
            lambda x: match_climbing_location(x, available_locations)
        )

        boulder_matched = boulder_df[boulder_df["location_normalized"].notna()].copy()
        if not boulder_matched.empty:
            boulder_stats = (
                boulder_matched.groupby("location_normalized")
                .agg({"stars_parsed": "mean", "routes_parsed": "sum"})
                .round(2)
            )
            boulder_stats.columns = ["avg_stars", "route_count"]
            boulder_stats["type"] = "Boulder"
            boulder_stats = boulder_stats.reset_index()
            all_climbing_stats.append(boulder_stats)

    # Sport climbing data
    if not sport_df.empty:
        sport_df["routes_parsed"] = sport_df["ROUTES IN DB"].apply(parse_routes_in_db)
        sport_df["stars_parsed"] = sport_df["STARS"].apply(parse_stars)
        sport_df["location_normalized"] = sport_df["NAME"].apply(
            lambda x: match_climbing_location(x, available_locations)
        )

        sport_matched = sport_df[sport_df["location_normalized"].notna()].copy()
        if not sport_matched.empty:
            sport_stats = (
                sport_matched.groupby("location_normalized")
                .agg({"stars_parsed": "mean", "routes_parsed": "sum"})
                .round(2)
            )
            sport_stats.columns = ["avg_stars", "route_count"]
            sport_stats["type"] = "Sport"
            sport_stats = sport_stats.reset_index()
            all_climbing_stats.append(sport_stats)

    # Combine climbing statistics
    if all_climbing_stats:
        combined_climbing_stats = pd.concat(all_climbing_stats, ignore_index=True)
        final_climbing_stats = (
            combined_climbing_stats.groupby("location_normalized")
            .agg({"avg_stars": "mean", "route_count": "sum"})
            .round(2)
            .reset_index()
        )
    else:
        # Fallback: dummy data
        final_climbing_stats = pd.DataFrame(
            {
                "location_normalized": available_locations,
                "avg_stars": np.random.uniform(3.0, 4.5, len(available_locations)),
                "route_count": np.random.randint(20, 200, len(available_locations)),
            }
        )

    # Process climbing recommendations
    if not climbing_recommendations.empty:
        climbing_recommendations["location_normalized"] = climbing_recommendations[
            "location_clean"
        ].apply(normalize_location_name)
        climbing_with_weather = climbing_recommendations[
            climbing_recommendations["location_normalized"].isin(available_locations)
        ].copy()
    else:
        climbing_with_weather = pd.DataFrame()

    return weather_df, final_climbing_stats, climbing_with_weather, available_locations


def calculate_climbing_recommendation(
    location, weather_data, climbing_data, climbing_recommendations
):
    """Calculate overall climbing recommendation score"""
    score = 0
    factors = []

    # Weather factors
    loc_weather = weather_data[weather_data["location_normalized"] == location]

    if not loc_weather.empty:
        # Precipitation (bad from 5mm, catastrophic from 10mm)
        precip = loc_weather[
            loc_weather["parameter"].str.contains("Precipitation", na=False)
        ]["value"].mean()
        wind = loc_weather[loc_weather["parameter"].str.contains("Wind", na=False)][
            "value"
        ].mean()

        if not np.isnan(precip):
            if precip == 0:
                precip_score = 100
            elif precip <= 2:
                precip_score = 90
            elif precip <= 5:
                precip_score = 70
                if not np.isnan(wind) and wind > 2:
                    precip_score += 15  # Wind bonus for drying
            elif precip <= 10:
                precip_score = 30
            else:
                precip_score = 0

            score += precip_score * 0.35
            factors.append(f"Niederschlag: {precip:.1f}mm ({precip_score:.0f}P)")

        # Temperature (optimal up to 15°C)
        temp_data = loc_weather[
            loc_weather["parameter"].str.contains("Temperature", na=False)
        ]
        if not temp_data.empty:
            temp_k = temp_data["value"].mean()
            temp_c = temp_k - 273.15 if temp_data["units"].iloc[0] == "K" else temp_k
            if not np.isnan(temp_c):
                if temp_c <= 0:
                    temp_score = 20
                elif temp_c <= 15:
                    temp_score = min(100, 40 + temp_c * 4)
                elif temp_c <= 20:
                    temp_score = max(60, 100 - (temp_c - 15) * 8)
                elif temp_c <= 25:
                    temp_score = max(20, 60 - (temp_c - 20) * 8)
                else:
                    temp_score = 0

                score += temp_score * 0.25
                factors.append(f"Temperatur: {temp_c:.1f}°C ({temp_score:.0f}P)")

        # Humidity (lower is better)
        humidity = loc_weather[
            loc_weather["parameter"].str.contains("Humidity", na=False)
        ]["value"].mean()
        if not np.isnan(humidity):
            if humidity <= 40:
                humidity_score = 100
            elif humidity <= 60:
                humidity_score = 100 - (humidity - 40) * 2
            elif humidity <= 80:
                humidity_score = max(20, 60 - (humidity - 60) * 2)
            else:
                humidity_score = 0

            score += humidity_score * 0.2
            factors.append(f"Luftfeuchtigkeit: {humidity:.1f}% ({humidity_score:.0f}P)")

        # Wind (good until it becomes gusty)
        if not np.isnan(wind):
            if wind <= 1:
                wind_score = 70
            elif wind <= 4:
                wind_score = 100
            elif wind <= 7:
                wind_score = 80
            elif wind <= 10:
                wind_score = 40
            else:
                wind_score = 0

            score += wind_score * 0.15
            factors.append(f"Wind: {wind:.1f}m/s ({wind_score:.0f}P)")

    # Area quality
    loc_climbing = climbing_data[climbing_data["location_normalized"] == location]
    if not loc_climbing.empty:
        stars = loc_climbing["avg_stars"].iloc[0]
        routes = loc_climbing["route_count"].iloc[0]

        stars_score = (stars - 1) * 25
        score += stars_score * 0.08
        factors.append(f"Bewertung: {stars:.1f}⭐ ({stars_score:.0f}P)")

        routes_score = min(100, routes * 0.1)
        score += routes_score * 0.02
        factors.append(f"Routen: {int(routes)} ({routes_score:.0f}P)")

    return score, factors


def create_weather_plots(weather_data, selected_locations):
    """Create weather plots for selected locations"""
    if not selected_locations:
        st.warning("Bitte wählen Sie mindestens einen Ort aus.")
        return

    weather_selected = weather_data[
        weather_data["location_normalized"].isin(selected_locations)
    ]

    # Create 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"Wetterprognose - {', '.join(selected_locations)}",
        fontsize=16,
        fontweight="bold",
    )

    # Colors for locations
    colors = plt.cm.Set3(np.linspace(0, 1, len(selected_locations)))
    location_colors = dict(zip(selected_locations, colors))

    # 1. Precipitation
    ax1 = axes[0, 0]
    precip_data = weather_selected[
        weather_selected["parameter"].str.contains("Precipitation", na=False)
    ]

    for location in selected_locations:
        loc_data = precip_data[precip_data["location_normalized"] == location]
        if not loc_data.empty:
            loc_grouped = loc_data.groupby("time")["value"].sum().reset_index()
            ax1.plot(
                loc_grouped["time"],
                loc_grouped["value"],
                marker="o",
                label=location,
                color=location_colors[location],
                linewidth=2,
                markersize=4,
            )

    ax1.set_title("Niederschlag", fontweight="bold")
    ax1.set_ylabel("mm")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis="x", rotation=45)

    # 2. Temperature
    ax2 = axes[0, 1]
    temp_data = weather_selected[
        weather_selected["parameter"].str.contains("Temperature", na=False)
    ]

    for location in selected_locations:
        loc_data = temp_data[temp_data["location_normalized"] == location]
        if not loc_data.empty:
            loc_data_celsius = loc_data.copy()
            if loc_data_celsius["units"].iloc[0] == "K":
                loc_data_celsius["value"] = loc_data_celsius["value"] - 273.15

            ax2.plot(
                loc_data_celsius["time"],
                loc_data_celsius["value"],
                marker="s",
                label=location,
                color=location_colors[location],
                linewidth=2,
                markersize=4,
            )

    ax2.set_title("Bodentemperatur", fontweight="bold")
    ax2.set_ylabel("°C")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis="x", rotation=45)

    # 3. Humidity
    ax3 = axes[1, 0]
    humidity_data = weather_selected[
        weather_selected["parameter"].str.contains("Humidity", na=False)
    ]

    for location in selected_locations:
        loc_data = humidity_data[humidity_data["location_normalized"] == location]
        if not loc_data.empty:
            ax3.plot(
                loc_data["time"],
                loc_data["value"],
                marker="^",
                label=location,
                color=location_colors[location],
                linewidth=2,
                markersize=4,
            )

    ax3.set_title("Luftfeuchtigkeit", fontweight="bold")
    ax3.set_ylabel("%")
    ax3.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis="x", rotation=45)

    # 4. Wind Speed
    ax4 = axes[1, 1]
    wind_data = weather_selected[
        weather_selected["parameter"].str.contains("Wind", na=False)
    ]

    for location in selected_locations:
        loc_data = wind_data[wind_data["location_normalized"] == location]
        if not loc_data.empty:
            ax4.plot(
                loc_data["time"],
                loc_data["value"],
                marker="d",
                label=location,
                color=location_colors[location],
                linewidth=2,
                markersize=4,
            )

    ax4.set_title("Windgeschwindigkeit", fontweight="bold")
    ax4.set_ylabel("m/s")
    ax4.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    return fig


def create_climbing_plots(
    climbing_data, climbing_recommendations_data, weather_data, selected_locations
):
    """Create climbing-related plots"""
    climbing_selected = climbing_data[
        climbing_data["location_normalized"].isin(selected_locations)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Routes and Ratings Bar Chart
    ax1 = axes[0]
    if not climbing_selected.empty:
        climbing_sorted = climbing_selected.sort_values("route_count", ascending=True)

        x_pos = np.arange(len(climbing_sorted))
        bar_width = 0.35

        normalized_stars = climbing_sorted["avg_stars"] * 20

        bars1 = ax1.barh(
            x_pos - bar_width / 2,
            climbing_sorted["route_count"],
            bar_width,
            label="Anzahl Routen",
            color="steelblue",
            alpha=0.8,
        )

        bars2 = ax1.barh(
            x_pos + bar_width / 2,
            normalized_stars,
            bar_width,
            label="Bewertung (⭐×20)",
            color="gold",
            alpha=0.8,
        )

        ax1.set_title("Routen & Bewertung", fontweight="bold")
        ax1.set_xlabel("Anzahl / Normalisierte Bewertung")
        ax1.set_yticks(x_pos)
        ax1.set_yticklabels(climbing_sorted["location_normalized"])
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis="x")

        # Add value labels
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            width1 = bar1.get_width()
            ax1.text(
                width1 + max(climbing_sorted["route_count"]) * 0.01,
                bar1.get_y() + bar1.get_height() / 2,
                f"{int(width1)}",
                ha="left",
                va="center",
                fontweight="bold",
                fontsize=9,
            )

            width2 = bar2.get_width()
            original_stars = climbing_sorted["avg_stars"].iloc[i]
            ax1.text(
                width2 + max(normalized_stars) * 0.01,
                bar2.get_y() + bar2.get_height() / 2,
                f"{original_stars:.1f}⭐",
                ha="left",
                va="center",
                fontweight="bold",
                fontsize=9,
            )

    # 2. Climbing Recommendations
    ax2 = axes[1]
    recommendations = []

    for location in selected_locations:
        score, factors = calculate_climbing_recommendation(
            location, weather_data, climbing_data, climbing_recommendations_data
        )
        recommendations.append({"location": location, "score": score})

    recommendations.sort(key=lambda x: x["score"], reverse=True)

    locations_ordered = [r["location"] for r in recommendations]
    scores = [r["score"] for r in recommendations]

    colors_score = [
        "#2E8B57" if s >= 80 else "#FFD700" if s >= 60 else "#FF6347" for s in scores
    ]

    bars_score = ax2.barh(
        range(len(locations_ordered)), scores, color=colors_score, alpha=0.8
    )

    ax2.set_title("Kletterempfehlung\n(Wetter + Gebiet)", fontweight="bold")
    ax2.set_xlabel("Gesamtbewertung (0-100)")
    ax2.set_yticks(range(len(locations_ordered)))
    ax2.set_yticklabels(locations_ordered)
    ax2.grid(True, alpha=0.3, axis="x")

    # Add score labels and emojis
    for i, (bar, score) in enumerate(zip(bars_score, scores)):
        width = bar.get_width()
        if score >= 80:
            emoji = "🟢 Excellent"
        elif score >= 60:
            emoji = "🟡 Good"
        else:
            emoji = "🔴 Fair"

        ax2.text(
            width + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.0f} {emoji}",
            ha="left",
            va="center",
            fontweight="bold",
        )

    plt.tight_layout()
    return fig, recommendations


# Main Streamlit App
def main():
    st.title("🏔️ Weather & Boulder Dashboard")
    st.markdown("""
    Interaktive Visualisierung der Wetter- und Klettergebietsdaten für alle von MeteoSwiss abgedeckten Orte.

    Dieses Dashboard zeigt:
    - **Wetterdaten**: Niederschlag, Bodentemperatur, Luftfeuchtigkeit, Windgeschwindigkeit
    - **Boulder/Kletter-Daten**: Bewertung & Anzahl Routen kombiniert
    - **Kletterempfehlung**: Gesamtbewertung basierend auf Wetter- und Gebietsdaten
    """)

    # Load data
    with st.spinner("Lade Daten vom GitHub Repository..."):
        weather_df, boulder_df, sport_df, climbing_recommendations = (
            load_data_from_github()
        )

    if weather_df is None:
        st.error(
            "Daten konnten nicht geladen werden. Bitte versuchen Sie es später erneut."
        )
        return

    # Process data
    with st.spinner("Verarbeite Daten..."):
        (
            weather_data,
            climbing_data,
            climbing_recommendations_data,
            available_locations,
        ) = process_data(weather_df, boulder_df, sport_df, climbing_recommendations)

    st.success(f"Daten erfolgreich geladen! {len(available_locations)} Orte verfügbar.")

    # Sidebar for location selection
    st.sidebar.header("📍 Orte auswählen")

    # Preset selections
    preset_options = {
        "Top 6 Empfohlen": [
            "Chironico",
            "Brione",
            "Magic Wood",
            "Gottardo / Gotthardpass",
            "Sustenpass",
            "Silvapark",
        ],
        "Alle Schweiz": [
            loc
            for loc in available_locations
            if any(
                x in loc
                for x in [
                    "Chironico",
                    "Brione",
                    "Magic Wood",
                    "Gottardo",
                    "Susten",
                    "Silva",
                    "Vernayaz",
                    "Kandersteg",
                    "Fionnay",
                ]
            )
        ],
        "Top 10 nach Routen": climbing_data.nlargest(10, "route_count")[
            "location_normalized"
        ].tolist(),
        "Top 10 nach Bewertung": climbing_data.nlargest(10, "avg_stars")[
            "location_normalized"
        ].tolist(),
    }

    st.sidebar.subheader("Schnellauswahl:")
    selected_preset = st.sidebar.selectbox(
        "Vordefinierte Auswahl:", ["Benutzerdefiniert"] + list(preset_options.keys())
    )

    if selected_preset != "Benutzerdefiniert":
        default_locations = [
            loc for loc in preset_options[selected_preset] if loc in available_locations
        ]
    else:
        default_locations = (
            available_locations[:6]
            if len(available_locations) >= 6
            else available_locations
        )

    # Manual selection
    st.sidebar.subheader("Manuelle Auswahl:")
    selected_locations = st.sidebar.multiselect(
        "Orte auswählen (max. 8 für bessere Darstellung):",
        available_locations,
        default=default_locations,
        max_selections=8,
    )

    if not selected_locations:
        st.warning("Bitte wählen Sie mindestens einen Ort aus der Seitenleiste aus.")
        return

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🌤️ Wetter", "🧗 Klettern", "📊 Datenübersicht", "ℹ️ Info"]
    )

    with tab1:
        st.header("Wetterprognose")

        if len(selected_locations) > 8:
            st.warning(
                "⚠️ Mehr als 8 Orte können die Darstellung unübersichtlich machen."
            )

        with st.spinner("Erstelle Wetterdiagramme..."):
            weather_fig = create_weather_plots(weather_data, selected_locations)
            st.pyplot(weather_fig)

    with tab2:
        st.header("Kletter-Analysen & Empfehlungen")

        with st.spinner("Erstelle Kletter-Analysen..."):
            climbing_fig, recommendations = create_climbing_plots(
                climbing_data,
                climbing_recommendations_data,
                weather_data,
                selected_locations,
            )
            st.pyplot(climbing_fig)

        # Detailed recommendations
        st.subheader("🎯 Detaillierte Empfehlungen")

        for i, rec in enumerate(recommendations, 1):
            score = rec["score"]
            location = rec["location"]

            if score >= 80:
                status = "🟢 Excellent"
            elif score >= 60:
                status = "🟡 Good"
            else:
                status = "🔴 Fair"

            with st.expander(f"{i}. {status} {location} - {score:.0f} Punkte"):
                _, factors = calculate_climbing_recommendation(
                    location, weather_data, climbing_data, climbing_recommendations_data
                )

                for factor in factors:
                    st.write(f"• {factor}")

                # Additional CSV recommendation if available
                if not climbing_recommendations_data.empty:
                    csv_rec = climbing_recommendations_data[
                        climbing_recommendations_data["location_normalized"] == location
                    ]
                    if not csv_rec.empty:
                        csv_score = csv_rec["climbing_score"].iloc[0]
                        csv_recommendation = csv_rec["recommendation"].iloc[0]
                        st.write(
                            f"📊 **CSV-Empfehlung**: {csv_score:.1f} - {csv_recommendation}"
                        )

    with tab3:
        st.header("📊 Datenübersicht")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Wetterdaten")
            weather_summary = (
                weather_data.groupby(["location_normalized", "parameter"])
                .agg(
                    {
                        "value": ["count", "mean", "min", "max"],
                        "data_source": "first",
                        "units": "first",
                    }
                )
                .round(2)
            )
            weather_summary.columns = ["count", "mean", "min", "max", "source", "units"]
            weather_summary = weather_summary.reset_index()

            st.write(
                f"Wetterdaten für {weather_summary['location_normalized'].nunique()} Orte:"
            )
            st.dataframe(weather_summary.head(20))

            st.subheader("Zeitbereich")
            st.write(f"**Von**: {weather_data['time'].min()}")
            st.write(f"**Bis**: {weather_data['time'].max()}")
            st.write(f"**Zeitpunkte**: {weather_data['time'].nunique()}")

        with col2:
            st.subheader("Kletter-Statistiken")
            st.write(f"Statistiken für {len(climbing_data)} Orte:")
            st.dataframe(climbing_data.sort_values("route_count", ascending=False))

            if not climbing_recommendations_data.empty:
                st.subheader("CSV Kletterempfehlungen")
                display_cols = [
                    "location_normalized",
                    "climbing_score",
                    "recommendation",
                    "precip_rate",
                    "t_2m",
                ]
                available_cols = [
                    col
                    for col in display_cols
                    if col in climbing_recommendations_data.columns
                ]
                st.dataframe(climbing_recommendations_data[available_cols].head(10))

    with tab4:
        st.header("ℹ️ Informationen")

        st.markdown("""
        ### Über diese Anwendung

        Diese Streamlit-Anwendung nutzt Daten aus dem GitHub-Repository
        [outdoor_weather_data](https://github.com/steffeol/outdoor_weather_data),
        welche täglich über GitHub Actions aktualisiert werden.

        ### Datenquellen
        - **Wetterdaten**: MeteoSwiss OGD API
        - **Kletterdaten**: 8a.nu (Boulder- und Sportklettergebiete)
        - **Empfehlungen**: Kombinierte Algorithmus aus Wetter- und Gebietsdaten

        ### Funktionen
        - **Interaktive Ortauswahl**: Bis zu 8 Orte gleichzeitig
        - **Live-Wetterprognose**: Niederschlag, Temperatur, Luftfeuchtigkeit, Wind
        - **Klettergebiet-Bewertungen**: Sterne-Ratings und Routenanzahl
        - **Intelligente Empfehlungen**: Kombiniert Wetter- und Gebietsdaten

        ### Deployment
        Diese App kann kostenlos auf folgenden Plattformen gehostet werden:
        - **Streamlit Cloud** (empfohlen)
        - **Heroku**
        - **Railway**
        - **Render**
        """)

        st.subheader("Verfügbare Orte")
        st.write(f"Insgesamt **{len(available_locations)}** Orte verfügbar:")

        # Display locations in columns
        cols = st.columns(3)
        for i, location in enumerate(available_locations):
            with cols[i % 3]:
                st.write(f"• {location}")


if __name__ == "__main__":
    main()
