**Strategic Intelligence: Multi-Stock Predictive Dashboard**

A real-time financial analysis platform that leverages Machine Learning to forecast the next-day closing prices for major NSE (National Stock Exchange) tickers. The system features a sophisticated Market Calendar Aware engine that automatically adjusts for weekends and Indian market holidays.

**Key Features**

AI-Driven Forecasting: Uses trained regression models to predict price movements based on Technical Indicators (Moving Averages, Daily Returns, Prev Close).

NSE Holiday Intelligence: Integrated with pandas_market_calendars to detect holidays (like Ram Navami, Diwali) and shift "Target Dates" automatically.

Dual-Asset Comparison: High-performance UI allowing users to compare two stocks side-by-side with equal-height "Bento" cards.

Historical Scorecard: Automatically tracks yesterday's predictions against today's actual close to monitor AI accuracy.

Smart UI Banners: Displays a contextual warning banner when the market is closed, informing the user that data is focused on the next trading session.

**Technical Stack**

Backend: Python 3.x, Flask

AI/ML: Scikit-learn, Joblib (Model Persistence)

Data Science: Pandas, Numpy, YFinance (Live Market Data)

Market Logic: Pandas-Market-Calendars (NSE Exchange)

Frontend: HTML5, CSS3, Bootstrap 5.3, Bootstrap Icons
