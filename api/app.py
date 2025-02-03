from flask import Flask, render_template
import numpy as np
import yfinance as yf
import plotly.express as px
import datetime
import io
import base64

from qiskit_optimization import QuadraticProgram
from qiskit_algorithms import NumPyMinimumEigensolver, SamplingVQE, QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import Sampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer

app = Flask(_name_)


def download_stock_data():
    tickers = ["MSFT", "AAPL", "GOOG", "META", "IBM", "GS", "TSLA", "V", "UNH"]
    start = "2021-01-01"
    end = "2021-12-31"

    # Download data and select closing prices
    data = yf.download(tickers, start=start, end=end)
    close_prices = data.xs('Close', level=0, axis=1)
    return close_prices, tickers


def create_interactive_stock_plot(price_data):
    """Create interactive stock price trajectory visualization with enhanced interactivity."""
    # Convert wide-form to long-form DataFrame for Plotly
    df = price_data.reset_index().melt(id_vars='Date', var_name='Stock', value_name='Price')

    # Create interactive line plot with Plotly Express
    fig = px.line(df, x='Date', y='Price', color='Stock',
                  title='Stock Price Trajectories',
                  labels={'Price': 'Adjusted Close Price (USD)'},
                  template='plotly_white')

    # Update layout to include range selectors and slider
    fig.update_layout(
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=1, label='1M', step='month', stepmode='backward'),
                    dict(count=6, label='6M', step='month', stepmode='backward'),
                    dict(count=1, label='YTD', step='year', stepmode='todate'),
                    dict(count=1, label='1Y', step='year', stepmode='backward'),
                    dict(step='all')
                ]
            ),
            rangeslider=dict(visible=True),
            type='date'
        )
    )
    # Return the Plotly graph as an HTML snippet with a fixed div id
    return fig.to_html(full_html=False, include_plotlyjs='cdn', div_id='interactive-plot')


def create_quadratic_program(mu, sigma, tickers):
    risk_factor = 0.5  # Risk factor
    num_assets = len(tickers)
    budget = num_assets // 2  # Budget constraint: select half of the assets

    qp = QuadraticProgram()

    # Add binary decision variables for each asset
    for ticker in tickers:
        qp.binary_var(name=ticker)

    # Build the linear part of the objective (-mu^T x)
    linear_objective = {ticker: -mu[i] for i, ticker in enumerate(tickers)}

    # Build the quadratic part of the objective (risk_factor * x^T Sigma x)
    quadratic_objective = {}
    for i in range(num_assets):
        for j in range(num_assets):
            quadratic_objective[(tickers[i], tickers[j])] = risk_factor * sigma[i, j]

    # Set the objective: minimize risk - return
    qp.minimize(linear=linear_objective, quadratic=quadratic_objective)

    # Add the budget constraint: sum(x_i) == budget
    constraint_linear = {ticker: 1 for ticker in tickers}
    qp.linear_constraint(linear=constraint_linear, sense='==', rhs=budget, name='budget')

    return qp


def print_result(result, qp):
    # Print the optimal solution found
    print("Optimal: selection " + str(result.x) + ", value " + str(result.fval))
    print("\n----------------- Full result ---------------------")
    print("selection\tvalue\t\tprobability")
    print("---------------------------------------------------")

    if hasattr(result, 'samples'):
        for sample in result.samples:
            x = sample.x
            probability = sample.probability
            value = qp.objective.evaluate(x)
            print("%10s\t%.4f\t\t%.4f" % (x, value, probability))
    else:
        print("No detailed sample information available.")


@app.route('/')
def index():
    # Download stock data and create an interactive Plotly graph
    close_prices, tickers = download_stock_data()
    plot_html = create_interactive_stock_plot(close_prices)

    # Compute log returns
    log_returns = np.log(close_prices / close_prices.shift(1))

    # Compute mean returns (mu) and covariance matrix (sigma)
    mu = log_returns.mean().values
    sigma = log_returns.cov().values

    # Create quadratic program for portfolio optimization
    qp = create_quadratic_program(mu, sigma, tickers)

    # Classical approach using NumPyMinimumEigensolver
    exact_mes = NumPyMinimumEigensolver()
    exact_eigensolver = MinimumEigenOptimizer(exact_mes)
    result = exact_eigensolver.solve(qp)
    print_result(result, qp)

    # Using SamplingVQE
    cobyla = COBYLA()
    cobyla.set_options(maxiter=500)
    ry = TwoLocal(len(tickers), "ry", "cz", reps=3, entanglement="full")
    svqe_mes = SamplingVQE(sampler=Sampler(), ansatz=ry, optimizer=cobyla)
    svqe = MinimumEigenOptimizer(svqe_mes)
    result = svqe.solve(qp)
    print_result(result, qp)

    # Using QAOA
    cobyla = COBYLA()
    cobyla.set_options(maxiter=250)
    qaoa_mes = QAOA(sampler=Sampler(), optimizer=cobyla, reps=3)
    qaoa = MinimumEigenOptimizer(qaoa_mes)
    result = qaoa.solve(qp)
    print_result(result, qp)

    # For display, determine selected assets from the last optimization result.
    selected_assets = [ticker for i, ticker in enumerate(tickers) if result.x[i] > 0.5]

    return render_template('index.html', plot_html=plot_html, selected_assets=selected_assets)


if _name_ == '_main_':
    app.run(debug=True)