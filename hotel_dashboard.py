# hotel_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from io import BytesIO

st.set_page_config(page_title="Hotel Booking Dashboard", layout="wide", initial_sidebar_state="expanded")

MONTHS = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]

@st.cache_data
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # parse dates
    if "reservation_status_date" in df.columns:
        df["reservation_status_date"] = pd.to_datetime(df["reservation_status_date"], errors="coerce")

    # fix numeric columns and missing values
    for c in ["children", "babies", "adults", "stays_in_weekend_nights", "stays_in_week_nights"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # guests and nights
    df["guests"] = df.get("adults", 0) + df.get("children", 0) + df.get("babies", 0)
    df["total_nights"] = df.get("stays_in_weekend_nights", 0) + df.get("stays_in_week_nights", 0)

    # months ordering
    if "arrival_date_month" in df.columns:
        df["arrival_date_month"] = pd.Categorical(
            df["arrival_date_month"], categories=MONTHS, ordered=True
        )

    # safety: ensure essential columns present
    expected = ["hotel","is_canceled","adr","lead_time","is_repeated_guest","arrival_date_year","reservation_status_date"]
    for e in expected:
        if e not in df.columns:
            df[e] = np.nan

    # cast some types
    df["is_canceled"] = pd.to_numeric(df["is_canceled"], errors="coerce").fillna(0).astype(int)
    df["is_repeated_guest"] = pd.to_numeric(df["is_repeated_guest"], errors="coerce").fillna(0).astype(int)
    df["arrival_date_year"] = pd.to_numeric(df["arrival_date_year"], errors="coerce")

    return df

def load_data(default_path="hotel_booking.csv"):
    p = Path(default_path)
    if p.exists():
        df = pd.read_csv(p)
        return preprocess(df)
    else:
        uploaded = st.sidebar.file_uploader("Upload hotel_booking.csv", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            return preprocess(df)
        else:
            st.sidebar.info("Place `hotel_booking.csv` in the app folder or use the uploader.")
            st.stop()

def to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="filtered")
        writer.save()
    return output.getvalue()

# ---------------------
# Load
# ---------------------
df = load_data()

# ---------------------
# Sidebar filters
# ---------------------
st.sidebar.header("Filters")

hotel_types = sorted(df["hotel"].dropna().unique())
hotel_sel = st.sidebar.multiselect("Hotel Type", options=hotel_types, default=hotel_types)

years = sorted(pd.Series(df["arrival_date_year"].dropna().unique()).astype(int))
years = [int(y) for y in years]
year_sel = st.sidebar.multiselect("Arrival Year", options=years, default=years)

customer_types = sorted(df["customer_type"].dropna().unique()) if "customer_type" in df.columns else []
customer_sel = st.sidebar.multiselect("Customer Type", options=customer_types, default=customer_types)

# date range on reservation_status_date
if df["reservation_status_date"].notna().any():
    min_date = df["reservation_status_date"].min().date()
    max_date = df["reservation_status_date"].max().date()
    date_range = st.sidebar.date_input("Reservation status date range", [min_date, max_date], min_value=min_date, max_value=max_date)
else:
    date_range = None

agg_map = {"Daily":"D", "Weekly":"W", "Monthly":"M"}
agg_sel = st.sidebar.selectbox("Trend granularity", options=list(agg_map.keys()), index=2)

# extra quick filters
st.sidebar.markdown("---")
show_top_countries = st.sidebar.slider("Top countries to show", min_value=5, max_value=20, value=10, step=1)

# ---------------------
# Filter dataframe
# ---------------------
mask = df["hotel"].isin(hotel_sel)
if year_sel:
    mask &= df["arrival_date_year"].isin(year_sel)
if customer_sel:
    mask &= df["customer_type"].isin(customer_sel)
if date_range:
    start_dt = pd.to_datetime(date_range[0])
    end_dt = pd.to_datetime(date_range[1])
    mask &= (df["reservation_status_date"] >= start_dt) & (df["reservation_status_date"] <= end_dt)

df_filtered = df.loc[mask].copy()

# ---------------------
# KPIs
# ---------------------
total_bookings = len(df_filtered)
canceled_count = int(df_filtered["is_canceled"].sum()) if total_bookings else 0
cancellation_rate = round((canceled_count / total_bookings) * 100, 2) if total_bookings else 0.0
avg_adr = round(df_filtered["adr"].replace([np.inf, -np.inf], np.nan).mean() or 0, 2)
median_adr = round(df_filtered["adr"].median() or 0, 2)
avg_lead_time = round(df_filtered["lead_time"].mean() or 0, 1)
avg_nights = round(df_filtered["total_nights"].mean() or 0, 2)
repeated_pct = round(df_filtered["is_repeated_guest"].mean() * 100 if total_bookings else 0, 2)

# header and KPIs layout
st.title("🏨 Hotel Booking Dashboard")
st.markdown("Interactive dashboard with KPIs and filters")

k1, k2, k3, k4, k5, k6 = st.columns([1.2,1,1,1,1,1])
k1.metric("Total Bookings", f"{total_bookings:,}")
k2.metric("Cancellations", f"{canceled_count:,}", f"{cancellation_rate}%")
k3.metric("Avg Daily Rate (ADR)", f"${avg_adr}", f"median ${median_adr}")
k4.metric("Avg Lead Time", f"{avg_lead_time} days")
k5.metric("Avg Nights per Booking", f"{avg_nights}")
k6.metric("Repeated Guests", f"{repeated_pct}%")

st.markdown("---")

# ---------------------
# Trend (Arrivals vs Cancellations)
# ---------------------
if not df_filtered.empty and df_filtered["reservation_status_date"].notna().any():
    freq = agg_map[agg_sel]
    temp = df_filtered.set_index("reservation_status_date").groupby([pd.Grouper(freq=freq), "is_canceled"]).size().unstack(fill_value=0)
    # ensure columns 0 and 1
    if 0 not in temp.columns:
        temp[0] = 0
    if 1 not in temp.columns:
        temp[1] = 0
    temp = temp.sort_index().rename(columns={0: "Arrived", 1: "Canceled"})
    temp = temp.reset_index()
    fig_trend = px.line(
        temp,
        x="reservation_status_date",
        y=["Arrived", "Canceled"],
        labels={"value": "Bookings", "reservation_status_date": "Date"},
        title=f"📈 Booking Trends ({agg_sel}) — Arrivals vs Cancellations"
    )
    st.plotly_chart(fig_trend, use_container_width=True)
else:
    st.info("No reservation_status_date data available for trend chart.")

# ---------------------
# Two-column charts
# ---------------------
col1, col2 = st.columns(2)

with col1:
    # Bookings per arrival month
    if "arrival_date_month" in df_filtered.columns:
        monthly = df_filtered.groupby("arrival_date_month").size().reindex(MONTHS).fillna(0)
        fig_month = px.bar(
            x=monthly.index.astype(str),
            y=monthly.values,
            labels={"x":"Month", "y":"Bookings"},
            title="📅 Bookings per Arrival Month"
        )
        st.plotly_chart(fig_month, use_container_width=True)
    else:
        st.info("No arrival_date_month column for monthly chart.")

with col2:
    # Top countries
    if "country" in df_filtered.columns:
        top_countries = df_filtered["country"].value_counts().head(show_top_countries)
        fig_country = px.bar(
            x=top_countries.index, y=top_countries.values,
            labels={"x":"Country","y":"Bookings"},
            title=f"🌍 Top {show_top_countries} Guest Countries"
        )
        st.plotly_chart(fig_country, use_container_width=True)
    else:
        st.info("No country column in dataset.")

# ---------------------
# ADR distribution and Room demand
# ---------------------
col3, col4 = st.columns(2)

with col3:
    if "adr" in df_filtered.columns and "hotel" in df_filtered.columns:
        fig_adr = px.box(df_filtered, x="hotel", y="adr", labels={"adr":"ADR","hotel":"Hotel"}, title="💵 ADR Distribution by Hotel")
        st.plotly_chart(fig_adr, use_container_width=True)
    else:
        st.info("adr or hotel column missing.")

with col4:
    if "reserved_room_type" in df_filtered.columns:
        room_counts = df_filtered["reserved_room_type"].value_counts().head(8)
        fig_room = px.pie(names=room_counts.index, values=room_counts.values, title="🛏️ Top Reserved Room Types")
        st.plotly_chart(fig_room, use_container_width=True)
    else:
        st.info("No reserved_room_type column.")

st.markdown("---")

# ---------------------
# Data table + downloads
# ---------------------
st.subheader("Filtered Data")

show_table = st.checkbox("Show filtered table (first 500 rows)", value=False)
if show_table:
    st.dataframe(df_filtered.head(500))

# download buttons
csv_bytes = df_filtered.to_csv(index=False).encode("utf-8")
excel_bytes = to_excel_bytes(df_filtered)

coldl, coldr = st.columns(2)
coldl.download_button("⬇️ Download CSV", data=csv_bytes, file_name="filtered_bookings.csv", mime="text/csv")
coldr.download_button("⬇️ Download Excel", data=excel_bytes, file_name="filtered_bookings.xlsx",
                      mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown("---")
st.caption("Tip: use the sidebar filters to drill into particular hotels, years or dates.")
