from setuptools import setup, find_packages

setup(
    name="stock-exchange-analyzer",
    version="0.1.0",
    description="A stock exchange analyzer that pulls data on a recurring basis and predicts stock evolution",
    author="Senseijuba",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "yfinance>=0.2.0",
        "APScheduler>=3.10.0",
        "python-dotenv>=1.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "joblib>=1.3.0",
        "tensorflow>=2.13.0",
        "requests>=2.31.0",
    ],
)
