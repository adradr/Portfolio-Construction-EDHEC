import pandas as pd
from scipy.optimize import minimize
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_ffme_returns():
    """
    Load the Fama-French Dataset for the returns of Top and Bottom Deciles by marketcap
    """
    me_m = pd.read_csv("data/Portfolios_Formed_on_ME_monthly_EW.csv",
                          header=0,
                          index_col=0,
                          parse_dates=True,
                          na_values=-99.99)
    rets = me_m[['Lo 10', 'Hi 10']]
    rets.columns = ['SmallCap', 'LargeCap']
    rets = rets / 100
    rets.index = pd.to_datetime(rets.index, format="%Y%m").to_period('M')
    return rets

def get_hfi_returns():
    """
    Load and format the EDHEC Hedge Fund Index Returns
    """
    hfi = pd.read_csv("data/edhec-hedgefundindices.csv",
                          header=0,
                          index_col=0,
                          parse_dates=True)
    hfi = hfi / 100
    hfi.index = hfi.index.to_period('M')
    return hfi

def get_ind_returns():
    """
    Load and format the Index Returns
    """
    ind = pd.read_csv("data/ind30_m_vw_rets.csv", header=0, index_col=0, parse_dates=True)/100
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period("M")
    ind.columns = ind.columns.str.strip()
    return ind

def portfolio_return(weights, returns):
    """
    Weights -> Returns
    """
    return weights.T @ returns

def portfolio_vol(weights, covmat):
    """
    Weight -> Vol
    """
    return (weights.T @ covmat @ weights)**0.5

def plot_ef2(n_points, er, cov, style=".-"):
    """
    Plots the 2-asset efficient frontier
    """
    if er.shape[0] != 2 or er.shape[0] != 2:
        raise ValueError("plot ef can only plot 2-asset frontiers")
    weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets,
        "Volatility": vols
    })
    return ef.plot.line(x="Volatility", y="Returns", style=style)

def minimize_vol(target_return, er, cov):
    """
    Target return -> Weight vector
    """
    n_assets = er.shape[0]
    init_guess = np.repeat(1/n_assets, n_assets)
    bounds = ((0.0, 1.0),)*n_assets
    return_is_target = {
        'type': 'eq',
        'args': (er,),
        'fun': lambda weights, er: target_return - portfolio_return(weights, er)
    }
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    results = minimize(portfolio_vol,
                        init_guess,
                        args=(cov,), 
                        method="SLSQP",
                        constraints=(return_is_target, weights_sum_to_1),
                        bounds=bounds)
    return results.x

def optimal_weights(n_points, er, cov):
    """
    Generate a list of weights to run the optimizer on
    """
    target_returns = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_returns]
    return weights

def gmv(cov):
    """
    Returns the weights of the Global Minimum Volatility portfolio
    given the covariance matrix
    """
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov)

def plot_ef(n_points, er, cov, show_cml=False, riskfree=0.05, show_ew=False, show_gmv=False, style=".-"):
    """
    Plots the multi-asset efficient frontier
    """
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets,
        "Volatility": vols
    })
    ax = ef.plot.line(x="Volatility", y="Returns", style=style, title="Multi-asset Efficient Frontier")
    ax.figure.set_figwidth(12)
    ax.figure.set_figheight(6)
    ax.set_xlim(0, 0.15)
    
    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_return(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)
        # Display EW Portfoliop
        ax.plot([vol_ew],[r_ew], color="red", marker="o")    
    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = portfolio_return(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        # Display EW Portfoliop
        ax.plot([vol_gmv],[r_gmv], color="blue", marker="o")
    if show_cml:
        rf = riskfree
        w_msr = msr(rf, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)

        cml_x = [0, vol_msr]
        cml_y = [rf, r_msr]

        ax.plot(cml_x,cml_y, color="green", marker="o", linestyle="dashed")
    return ax
    
def msr(riskfree_rate, er, cov):
    """
    Maximum Sharpe Ratio
    Riskfree rate + ER + COV -> Weight vector
    """
    n_assets = er.shape[0]
    init_guess = np.repeat(1/n_assets, n_assets)
    bounds = ((0.0, 1.0),)*n_assets
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    def neg_sharpe_ratio(weights, riskfree_rate, er, cov):
        """
        Returns the negative of the sharpe ratios given weights.
        """
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate)/vol
        
    results = minimize(neg_sharpe_ratio,
                        init_guess,
                        args=(riskfree_rate,er,cov,), 
                        method="SLSQP",
                        constraints=(weights_sum_to_1),
                        bounds=bounds)
    return results.x

def plot_ef_random(data, n_portfolio, n_periods):
    """
    Plots the Efficient Frontier with random guessing
    given a DataFrame of returns, the number of portfolios to generate
    and the periodicity for annualization
    Returns a scatter plot
    """
    daily_return= data.dropna()
    q1_return= daily_return.mean()*n_periods
    q1_cov= daily_return.cov()*n_periods
    pf_returns, pf_volatility, pf_sharpe_ratio, pf_coins_weights=([] for i in range(4))
    num_portfolios= n_portfolio
    for portfolio in range(num_portfolios):
        weights= np.random.random(data.shape[1])
        weights /= np.sum(weights)
        returns = np.dot(weights, q1_return)
        volatility = np.sqrt(np.dot(weights.T, np.dot(q1_cov, weights)))
        sharpe = returns / volatility
        pf_coins_weights.append(weights)
        pf_returns.append(returns)
        pf_volatility.append(volatility)
        pf_sharpe_ratio.append(sharpe)
    plt.figure(figsize=(12,7))
    plt.scatter(x=pf_volatility, y=pf_returns, c= pf_sharpe_ratio, cmap='viridis')
    plt.colorbar(label='Sharpe Ratio')
    sns.set(style='darkgrid')
    plt.title('Efficient Frontier')
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.show()


def drawdown(return_series: pd.Series):
    """
    `drawdown` takes a time series of asset returns
    Computes and returns a DataFrame that contains:
    wealth index
    previous peaks
    percent drawdowns
    """
    wealth_index = 1000*(1 + return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    return pd.DataFrame({
        "Wealth" : wealth_index,
        "Peaks" : previous_peaks,
        "Drawdown" : drawdowns
    })

def annualized_return(r, periods_per_year):
    """
    Computes annualized return on a time series of asset returns
    """
    compounded_growth = (r + 1).prod()
    n_periods = r.shape[0]
    return compounded_growth ** (periods_per_year / n_periods) - 1 

def annualized_volatility(r, periods_per_year):
    """
    Computes annualized volatility on a time series of asset returns
    """
    return r.std()*np.sqrt(periods_per_year)

def sharpe_ratio(r, periods_per_year, riskfree=0.03 ):
    """
    Computes sharpe ratio on a series of returns
    Sharpe ratio is the ratio of the excess return divided by the annualized volatility
    """
    rf_per_period = (1+riskfree)**(1/periods_per_year)-1
    excess_return = r - rf_per_period
    ann_excess_return = annualized_return(excess_return, periods_per_year)
    ann_volatility = annualized_volatility(r, periods_per_year)
    return ann_excess_return / ann_volatility

def semideviation(r):
    is_negative = r < 0
    return r[is_negative].std(ddof=0)

def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3

def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4

def is_normal(r, level=0.01):
    """
    Applies the Jarque-Bera test to determine if a Seris is a normal or not 
    Test is applied at the 1% level by default
    Returns True if the hypothesis of normality is accepted , False otherwise
    """
    statistic, p_value = scipy.stats.jarque_bera(r)
    x = p_value>level
    return x

def var_historic(r, level=5):
    """
    VaR Historic
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be Series or DataFrame")        
    
def var_gaussian(r, level=5, modified=False):
    z = scipy.stats.norm.ppf(level/100)
    if modified:
        # modify the Z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = ( z + 
             (z**2 - 1)*s/6 + 
             (z**3 -3*z)*(k-3)/24 - 
             (2*z**3 - 5*z)*(s**2)/36
            )
        
    return -(r.mean() + z*r.std(ddof=0))

def cvar_historic(r, level=5):
    """
    Computes the Conditional VaR of Series or DataFrame
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be Series or DataFrame")