import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
from statsmodels.tsa.stattools import adfuller


def plot_day(df, day_string):
  """Create line plot for a specific day

  Args:
      df (pd.DataFrame): DataFrame containing generation data with a datetime index.
      day_string (str): data string in a format recognized by pandas (e.g., 'YYYY-MM-DD').
  """
  date_to_focus = pd.to_datetime(day_string).date()

  timezone = df.index.tz
  start_date = date_to_focus - pd.Timedelta(days=2)
  end_date = date_to_focus + pd.Timedelta(days=2)

  mask = (df.index.date >= start_date) & (df.index.date <= end_date)
  data_to_plot = df[mask]
  fig, ax = plt.subplots(figsize=(14, 8))
  for column in list(data_to_plot.columns):
      plt.plot(data_to_plot[[column]], label=column)

  plt.xlabel('Date')
  plt.ylabel('Generation')
  plt.title(f'Generation for {date_to_focus}')
  plt.grid(True)
  plt.legend(bbox_to_anchor=(1, 1))
  plt.show()

def plot_null_values_specific_day(df, day_string):
  """Line plot for a specific day showing all columns with null values.

  Args:
      df (pd.DataFrame): DataFrame containing generation data with a datetime index.
      day_string (str): data string in a format recognized by pandas (e.g., 'YYYY-MM-DD').
  """
  date_to_focus = pd.to_datetime(day_string).date()
  timezone = df.index.tz
  start_date = date_to_focus - pd.Timedelta(days=2)
  end_date = date_to_focus + pd.Timedelta(days=2)

  mask = (df.index.date >= start_date) & (df.index.date <= end_date)

  data_to_plot = df[mask]
  columns_with_nan_values = data_to_plot.columns[data_to_plot.isnull().any()].tolist()
  data_to_plot = data_to_plot[columns_with_nan_values]

  fig, ax = plt.subplots(figsize=(14, 8))
  for column in list(data_to_plot.columns):
      plt.plot(data_to_plot[[column]], label=column)

  plt.xlabel('Date')
  plt.ylabel('Generation')
  plt.title(f'Generation for {date_to_focus}')
  plt.grid(True)
  plt.legend(bbox_to_anchor=(1, 1))
  plt.show()
  
def see_seasonality(df, column):
  """Visualize the seasonal patterns of a specific column in a DataFrame.

  Args:
      df (pd.DataFrame): DataFrame containing generation data with a datetime index.
      column (str): DataFrame column name.
  """
  fig, ax = plt.subplots(figsize=(12,7), nrows=2, ncols=2)
  ax = ax.flatten()
  ax[0].plot(df.groupby(df.index.hour)[column].mean())
  ax[0].set_title('Hourly')
  ax[0].set_xlabel('Hour')

  ax[1].plot(df.groupby(df.index.dayofweek)[column].mean())
  ax[1].set_title('Weekly')
  ax[1].set_xlabel('Day of Week')

  ax[2].plot(df.groupby(df.index.month)[column].mean())
  ax[2].set_title('Monthly')
  ax[2].set_xlabel('Month')

  # hide blank plots
  for ax in ax[3:]:
        ax.set_visible(False)

  fig.suptitle(f'Check Seasonality for feature "{column}"', fontsize=16)
  plt.tight_layout()
  plt.show()

def see_intermittency(df):
  """Check sparsity for each DataFrame column.

  Args:
      df (pd.DataFrame)
  """
  zero_counts = (df==0).sum(axis=0) # count number of zeros
  zero_percentages = (zero_counts/len(df)*100).round(2) # count number of zeros
  zero_stats = pd.DataFrame({'number_of_zeros': zero_counts, 'Percentage of Zeros': zero_percentages})
  print(zero_stats)

  plt.figure(figsize=(12, 6))
  sns.barplot(data=zero_stats, x=zero_stats.index, y='number_of_zeros')
  plt.title('Number of Zeros per Column')
  plt.xlabel('Columns')
  plt.ylabel('Number of Zeros')
  plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
  plt.show()

def detect_outliers(df, column_name):
    """Detect and visualize outliers for a specific column in a DataFrame.

    Args:
        df (pd.DataFrame): _description_
        column_name (str): DataFrame column_name

    Returns:
        pd.Index: A list of unique dates that contain outliers in the specified column.
    """
    series = df[column_name]
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df_outliers = df[(series < lower_bound) | (series > upper_bound)]
    outlier_dates = set(df_outliers.index.date)
    outlier_dates = sorted(outlier_dates,reverse=False)

    print(f'Number of different days with outliers: {len(outlier_dates)}')
    print(df_outliers.groupby(df_outliers.index.date).size())

    grouped_data = df.groupby(df.index.date)

    for date in outlier_dates:
        print(f"Plotting data for date: {date}")

        start_date = date -  pd.DateOffset(days=2)
        end_date = date + pd.DateOffset(days=2)

        # Create a mask for the dates that are in the dates_to_exclude
        mask = (df.index.date >= start_date.date()) & (df.index.date <= end_date.date())
        data_to_plot = df[mask]

        # Create a line plot for the original data on this specific date
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=data_to_plot, x=data_to_plot.index, y=column_name, label=f"Data for {date}")

        # Highlight the outliers in red
        outliers_for_date = df_outliers[df_outliers.index.date == date]
        sns.scatterplot(data=outliers_for_date, x=outliers_for_date.index, y=column_name, color='red', s=100, label="Outliers")

        plt.title(f"Original Data with Outliers for {column_name} on {date}")
        plt.xlabel("Time")
        plt.ylabel(column_name)
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.show()
    return df_outliers.index.date

# create distribution plot
def data_visualizations_plots(df, type):
  """Function to display different Plot types.

  Args:
      df (pd.DataFrame): pandas DataFrame that is going to be analyzed.
      type (str): type of visual to display
  """
  fig, axs = plt.subplots(int(np.ceil(len(df.columns) / 2)),2, figsize=(15,20))
  axs = axs.flatten()
  for i, col in enumerate(list(df.columns)):
      if type == 'kdeplot':
        sns.kdeplot(data=df, x=col, ax=axs.flatten()[i], fill=True)
      elif type == 'boxplot':
        sns.boxplot(data=df, x=col, ax=axs.flatten()[i])
      elif type == 'violin':
        sns.violinplot(data=df, x=col, ax=axs.flatten()[i])
      elif type == 'histoplot':
        sns.histplot(data=df, x=col, ax=axs.flatten()[i])

      axs[i].set_title(f'{col} Distribution')  # Set the title for each subplot
      plt.title(f'{col} Distribution')

  # hide blank plots
  for ax in axs[len(df.columns):]:
      ax.set_visible(False)

  plt.tight_layout()
  plt.show()


def apply_linear_interpolation_nan_values(df, columns, date, hour):
  """Function to apply linear interpolation for NaN values in specified columns.

  Args:
      df (pd.DataFrame):
      columns (list): DataFrame columns names.
      date (str): data string in a format recognized by pandas (e.g., 'YYYY-MM-DD').
      hour (int): target hour for filtering rows, represented as an integer (0–23).

  Returns:
      pd.DataFrame: A DataFrame with linear interpolation applied to specified columns within the target date and hour.
  """
  mask = (df.index.date == np.datetime64(date)) & (df.index.hour == hour)
  df.loc[mask, columns] = df.loc[mask, columns].replace(0, np.nan)
  df.interpolate('linear', inplace=True)
  return df[columns]

def plot_correlation(df, type):
  """Display correlation matrix for a specific DataFrame.

  Args:
      df (pd.DataFrame)
      type (str): type of correlation to apply
  """
  plt.figure(figsize=(12,8))
  if type == 'pearson':
    corr_matrix = df.corr(method='pearson')

  elif type == 'spearman':
    corr_matrix = df.corr(method='spearman')

  sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', mask=np.triu(df.corr()))
  plt.show()
  print(corr_matrix['price actual'].sort_values(ascending=False))
  
  
def season_plot_daily_consumption(df_to_plot, column='price actual'):
  """Plots the daily seasonal pattern of a specified column in the DataFrame by averaging values for each hour of the week.

  This function creates a seasonal line plot showing how values in the specified column vary by hour across different days 
  of the week. This is useful for visualizing daily trends and identifying patterns in time-series data, such as consumption 
  or pricing patterns throughout the week.

  Args:
      df_to_plot (pd.DataFrame): DataFrame containing a datetime index and at least one numeric column for analysis.
      column (str, optional): Name of the column to plot for daily seasonality, with a default value of 'price actual'.

  """
  df = df_to_plot.copy()
  df['hour'] = df.index.hour
  df['day'] = df.index.day_name()
  df_plot = df.groupby(['hour', 'day']).mean()[[column]].reset_index()

  plt.figure(figsize=(10,8))
  sns.lineplot(data = df_plot, x='hour', y=column, hue='day', legend=True)
  plt.locator_params(axis='x', nbins=24)
  plt.title("Seasonal Plot - Daily Price", fontsize=20)
  plt.ylabel('€/MWh')
  plt.xlabel('Hour')
  plt.legend()
  plt.show()

def season_plot_yearly_consumption(df_to_plot, column='price actual'):
  """Plots the yearly seasonal pattern of a specified column in the DataFrame by averaging values for each month of the year.

  This function creates a seasonal line plot showing how values in the specified column vary by month across different years. 
  This is useful for visualizing yearly trends and identifying patterns in time-series data, such as consumption 
  or pricing patterns throughout the year.

  Args:
      df_to_plot (pd.DataFrame): DataFrame containing a datetime index and at least one numeric column for analysis.
      column (str, optional): Name of the column to plot for yearly seasonality, with a default value of 'price actual'.
  """
  
  df = df_to_plot.copy()
  df['year'] = df.index.year
  df['month'] = df.index.month
  df_plot = df.groupby(['month', 'year']).mean()[[column]].reset_index()

  plt.figure(figsize=(10,8))
  sns.lineplot(data = df_plot, x='month', y=column, hue='year', legend=True)
  plt.locator_params(axis='x', nbins=24)
  plt.title("Seasonal Plot - Yearly Price", fontsize=20)
  plt.ylabel('€/MWh')
  plt.xlabel('Month')
  plt.legend()
  plt.show()

def plot_box_plots_distributions(df_to_plot, distribution, column_to_plot='price actual'):
  """Plots box plots to show the distribution of values by daily, monthly, or annual seasonality.
  
    Args:
        df_to_plot (pd.DataFrame): DataFrame containing a datetime index and at least one numeric column for analysis.
        distribution (str): Specifies the seasonal distribution type, must be one of 'hour', 'month', or 'day'.
            - 'hour': Plots box plot by hour of the day (0-23).
            - 'month': Plots box plot by month of the year (1-12).
            - 'day': Plots box plot by day of the week (Monday-Sunday).
        column_to_plot (str, optional): Column name in df_to_plot to plot; defaults to 'price actual'.

  """
  df = df_to_plot.copy()
  plt.figure(figsize=(10, 6))

  if distribution == 'hour':
    df['hour'] = df.index.hour
    sns.boxplot(x='hour', y=column_to_plot, data=df)
    plt.xlabel('Hour of Day')
    plt.title('Box Plot of Prices by Hour')
  elif distribution == 'month':
      df['month'] = df.index.month
      sns.boxplot(x='month', y=column_to_plot, data=df)
      plt.xlabel('Month of Year')
      plt.title('Box Plot of Prices by Month')
  elif distribution == 'day':
      df['day'] = df.index.day_name()
      day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
      df['day'] = pd.Categorical(df['day'], categories=day_order, ordered=True)
      sns.boxplot(x='day', y=column_to_plot, data=df)
      plt.xlabel('Day of Week')
      plt.title('Box Plot of Prices by Day')

  plt.show()
  
"""
'The autocorrelation analysis helps in detecting hidden patterns and seasonality and in checking for randomness.'
"""
def check_stationarity(series):
    # Copied from https://machinelearningmastery.com/time-series-data-stationary-python/
    """
    p-value > 0.05: Fail to reject the null hypothesis (H0), the data has a unit root and is non-stationary.
    p-value <= 0.05: Reject the null hypothesis (H0), the data does not have a unit root and is stationary.
    """

    result = adfuller(series)

    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

    if (result[1] <= 0.05) & (result[4]['5%'] > result[0]):
        print("\u001b[32mStationary\u001b[0m")
    else:
        print("\x1b[31mNon-stationary\x1b[0m")

