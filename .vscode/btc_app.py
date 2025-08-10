import yfinance as yf
import streamlit as st

# Fetch Bitcoin data
btc_ticker = yf.Ticker("BTC-USD")
btc_data = btc_ticker.history(interval="4h")

# Prepare data for plotting
btc_data = btc_data.reset_index()
btc_data = btc_data.rename(columns={'Datetime': 'Date'})
btc_data_for_plot = btc_data[['Date', 'Close']]

# Create a Streamlit application
st.title("Bitcoin (BTC-USD) 4-Hour Close Price")
st.write("Historical data for BTC-USD with a 4-hour interval:")
st.dataframe(btc_data_for_plot)

st.line_chart(btc_data_for_plot, x='Date', y='Close')

print("To run the Streamlit application:")
print("1. Save the above code to a Python file (e.g., btc_app.py).")
print("2. Open a terminal or command prompt.")
print("3. Navigate to the directory where you saved the file.")
print("4. Run the command: streamlit run btc_app.py")
print("5. This command will open the Streamlit application in your web browser.")
