"""
Dashboard for the stock analyzer - built with Dash/Plotly
"""
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
from io import StringIO
from datetime import datetime

from src.data_fetcher.fetcher import DataFetcher
from src.analysis.analyzer import StockAnalyzer
from src.prediction.predictor import get_model, get_model_options
from src.utils.helpers import load_config, ensure_directories, save_predictions
from src.utils.symbols import get_all_symbols

# chart styling defaults
CHART_TEMPLATE = "plotly_dark"
TRANSPARENT_BG = "rgba(0,0,0,0)"


class StockDashboard:
    
    def __init__(self):
        ensure_directories()
        self.config = load_config()
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.DARKLY],
            title="Stock Analyzer"
        )
        self._build_layout()
        self._register_callbacks()

    def _build_layout(self):
        all_symbols = get_all_symbols()
        default_symbol = self.config['symbols'][0] if self.config['symbols'] else "AAPL"
        
        # searchable dropdown with all symbols
        symbol_opts = [{"label": html.Span([s], style={'color': 'black'}), "value": s} for s in all_symbols]
        period_opts = [
            {"label": html.Span([label], style={'color': 'black'}), "value": val}
            for label, val in [("1 Month", "1mo"), ("3 Months", "3mo"), 
                               ("6 Months", "6mo"), ("1 Year", "1y"), ("2 Years", "2y")]
        ]

        self.app.layout = dbc.Container([
            # header
            dbc.Row([
                dbc.Col([
                    html.H1("Stock Exchange Analyzer", className="text-center my-4"),
                    html.P("Analysis and price predictions", className="text-center text-muted")
                ])
            ]),

            # controls
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Stock Selection"),
                        dbc.CardBody([
                            dbc.Label("Symbol (type to search)"),
                            dcc.Dropdown(
                                id="symbol-dropdown",
                                options=symbol_opts,
                                value=default_symbol,
                                clearable=False,
                                searchable=True,
                                placeholder="Type symbol...",
                                className="mb-3"
                            ),
                            dbc.Label("Period"),
                            dcc.Dropdown(
                                id="period-dropdown",
                                options=period_opts,
                                value="6mo",
                                clearable=False,
                                className="mb-3"
                            ),
                            dbc.Button("Fetch Data", id="fetch-button", color="primary", className="w-100"),
                        ])
                    ])
                ], md=3),

                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Predictions"),
                        dbc.CardBody([
                            dbc.Label("Model"),
                            dcc.Dropdown(
                                id="model-dropdown",
                                options=[
                                    {"label": html.Span([name], style={'color': 'black'}), "value": mid}
                                    for mid, name in get_model_options()
                                ],
                                value="rf",
                                clearable=False,
                                className="mb-3"
                            ),
                            dbc.Label("Forecast Days"),
                            dcc.Slider(
                                id="prediction-days-slider",
                                min=7, max=90, step=7, value=30,
                                marks={7: "7", 30: "30", 60: "60", 90: "90"},
                                className="mb-3"
                            ),
                            dbc.Button("Run Prediction", id="predict-button", color="success", className="w-100"),
                        ])
                    ])
                ], md=3),

                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Stats"),
                        dbc.CardBody(id="stats-card", children=[
                            html.P("No data loaded", className="text-muted")
                        ])
                    ])
                ], md=6),
            ], className="mb-4"),

            dcc.Loading(id="loading", type="circle", children=[html.Div(id="loading-output")]),

            # main charts
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Price"),
                        dbc.CardBody([dcc.Graph(id="price-chart", style={"height": "500px"})])
                    ])
                ], md=8),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Volume"),
                        dbc.CardBody([dcc.Graph(id="volume-chart", style={"height": "230px"})])
                    ]),
                    dbc.Card([
                        dbc.CardHeader("RSI"),
                        dbc.CardBody([dcc.Graph(id="rsi-chart", style={"height": "230px"})])
                    ], className="mt-2")
                ], md=4),
            ], className="mb-4"),

            # prediction chart
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Price Forecast"),
                        dbc.CardBody([dcc.Graph(id="prediction-chart", style={"height": "400px"})])
                    ])
                ])
            ], className="mb-4"),

            dbc.Row([dbc.Col([html.Div(id="status-bar", className="text-center text-muted p-2")])]),

            # client-side data storage
            dcc.Store(id="stock-data-store"),
            dcc.Store(id="analysis-data-store"),

        ], fluid=True, className="p-4")

    def _register_callbacks(self):
        
        @self.app.callback(
            [Output("stock-data-store", "data"),
             Output("analysis-data-store", "data"),
             Output("stats-card", "children"),
             Output("status-bar", "children"),
             Output("loading-output", "children")],
            Input("fetch-button", "n_clicks"),
            [State("symbol-dropdown", "value"), State("period-dropdown", "value")],
            prevent_initial_call=True
        )
        def on_fetch(n_clicks, symbol, period):
            if not n_clicks:
                return dash.no_update

            try:
                fetcher = DataFetcher([symbol])
                fetcher.set_period(period)
                data = fetcher.fetch_historical_data()

                if data is None or data.empty:
                    return None, None, html.P("No data", className="text-danger"), f"Failed: {symbol}", ""

                # handle multi-index columns from yfinance
                if hasattr(data.columns, 'levels'):
                    stock_df = data.xs(symbol, level=1, axis=1)
                else:
                    stock_df = data[['Open', 'High', 'Low', 'Close', 'Volume']]

                analyzer = StockAnalyzer(stock_df.copy())
                analyzer.calculate_moving_averages()
                analyzer.calculate_rsi()
                analyzer.calculate_macd()
                analyzer.calculate_volatility()
                stats = analyzer.get_stats()

                stock_json = stock_df.reset_index().to_json(date_format='iso')
                analysis_json = analyzer.data.reset_index().to_json(date_format='iso')

                stats_row = dbc.Row([
                    dbc.Col([
                        html.H4(f"${stats.get('price', 0):.2f}", className="text-success"),
                        html.Small("Price")
                    ], className="text-center"),
                    dbc.Col([
                        html.H4(f"${stats.get('mean', 0):.2f}"),
                        html.Small("Avg")
                    ], className="text-center"),
                    dbc.Col([
                        html.H4(f"{stats.get('volatility', 0):.1%}"),
                        html.Small("Vol")
                    ], className="text-center"),
                    dbc.Col([
                        html.H4(f"{stats.get('return_pct', 0)/100:.1%}"),
                        html.Small("Return")
                    ], className="text-center"),
                ])

                return stock_json, analysis_json, [stats_row], f"Loaded {symbol}", ""

            except Exception as e:
                return None, None, html.P(str(e), className="text-danger"), f"Error: {e}", ""

        @self.app.callback(
            [Output("price-chart", "figure"),
             Output("volume-chart", "figure"),
             Output("rsi-chart", "figure")],
            Input("analysis-data-store", "data"),
            State("symbol-dropdown", "value"),
            prevent_initial_call=True
        )
        def update_charts(analysis_json, symbol):
            empty = self._empty_figure()
            if not analysis_json:
                return empty, empty, empty

            df = pd.read_json(StringIO(analysis_json))
            df = self._fix_date_index(df)

            # candlestick + moving averages
            price_fig = go.Figure()
            price_fig.add_trace(go.Candlestick(
                x=df.index, open=df['Open'], high=df['High'], 
                low=df['Low'], close=df['Close'], name="OHLC"
            ))
            
            for col, color in [('SMA_20', 'orange'), ('SMA_50', 'blue'), ('EMA_12', 'green')]:
                if col in df.columns:
                    price_fig.add_trace(go.Scatter(
                        x=df.index, y=df[col], name=col, 
                        line=dict(color=color, width=1, dash='dash' if 'EMA' in col else 'solid')
                    ))
            
            price_fig.update_layout(
                title=symbol, template=CHART_TEMPLATE,
                paper_bgcolor=TRANSPARENT_BG, plot_bgcolor=TRANSPARENT_BG,
                xaxis_rangeslider_visible=False,
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )

            # volume bars
            bar_colors = ['#ef5350' if c < o else '#26a69a' for c, o in zip(df['Close'], df['Open'])]
            vol_fig = go.Figure(go.Bar(x=df.index, y=df['Volume'], marker_color=bar_colors))
            vol_fig.update_layout(
                template=CHART_TEMPLATE, paper_bgcolor=TRANSPARENT_BG, 
                plot_bgcolor=TRANSPARENT_BG, showlegend=False, margin=dict(l=0, r=0, t=0, b=0)
            )

            # RSI with overbought/oversold lines
            rsi_fig = go.Figure()
            if 'RSI' in df.columns:
                rsi_fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#ab47bc', width=2)))
                rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
                rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
            rsi_fig.update_layout(
                template=CHART_TEMPLATE, paper_bgcolor=TRANSPARENT_BG, plot_bgcolor=TRANSPARENT_BG,
                yaxis=dict(range=[0, 100]), showlegend=False, margin=dict(l=0, r=0, t=0, b=0)
            )

            return price_fig, vol_fig, rsi_fig

        @self.app.callback(
            [Output("prediction-chart", "figure"),
             Output("status-bar", "children", allow_duplicate=True)],
            Input("predict-button", "n_clicks"),
            [State("symbol-dropdown", "value"),
             State("model-dropdown", "value"),
             State("prediction-days-slider", "value"),
             State("stock-data-store", "data")],
            prevent_initial_call=True
        )
        def run_prediction(n_clicks, symbol, model_type, days, stock_json):
            if not n_clicks or not stock_json:
                return dash.no_update, "Load data first"

            try:
                df = pd.read_json(StringIO(stock_json))
                df = self._fix_date_index(df)

                predictor = get_model(model_type)
                predictor.train(df[['Close']])
                recent_prices = df['Close'].tail(predictor.lookback).values
                
                preds, lower, upper = predictor.predict_with_ci(recent_prices, days)
                model_name = get_model_options()
                model_name = dict(model_name).get(model_type, model_type)
                save_predictions(symbol, preds.tolist(), days)

                forecast_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=days)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df.tail(60).index, y=df.tail(60)['Close'],
                    name="History", line=dict(color='cyan', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=forecast_dates, y=preds,
                    name="Forecast", line=dict(color='yellow', width=2, dash='dash')
                ))
                
                # 95% confidence interval
                fig.add_trace(go.Scatter(
                    x=list(forecast_dates) + list(forecast_dates[::-1]),
                    y=list(upper) + list(lower[::-1]),
                    fill='toself', fillcolor='rgba(255,255,0,0.15)',
                    line=dict(color='rgba(0,0,0,0)'), name="95% CI", showlegend=True
                ))

                fig.update_layout(
                    title=f"{symbol} {days}d forecast ({model_name})",
                    template=CHART_TEMPLATE, paper_bgcolor=TRANSPARENT_BG, plot_bgcolor=TRANSPARENT_BG,
                    xaxis_title="Date", yaxis_title="Price",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02)
                )

                return fig, f"Forecast complete ({days}d)"

            except Exception as e:
                return self._empty_figure(), f"Prediction failed: {e}"

    def _empty_figure(self):
        fig = go.Figure()
        fig.update_layout(template=CHART_TEMPLATE, paper_bgcolor=TRANSPARENT_BG)
        return fig

    def _fix_date_index(self, df):
        """Convert date column to proper datetime index"""
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        elif 'index' in df.columns:
            df['index'] = pd.to_datetime(df['index'])
            df.set_index('index', inplace=True)
        return df

    def run(self, debug=True, port=8050):
        print(f"Starting dashboard at http://localhost:{port}")
        self.app.run(debug=debug, port=port)


def run_dashboard():
    StockDashboard().run()


if __name__ == "__main__":
    run_dashboard()
