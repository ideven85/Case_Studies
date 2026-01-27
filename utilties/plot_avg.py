import matplotlib.pyplot as plt
import seaborn as sns


def plot_avg_price_by_category(df, category, col):
    avg_price = df.groupby(category)[col].mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=avg_price.values, y=avg_price.index)
    plt.title(f"Average Price by {category}")
    plt.xlabel(f"Average {col}")
    plt.ylabel(category)
    plt.grid(True)
    plt.show()
