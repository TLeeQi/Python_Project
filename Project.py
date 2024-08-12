import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class project:
    def __init__(self):
        pass
        
    def load_data(self, url):
        try:
            df = pd.read_parquet(url)
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def process_date_column(self, df, date_column='date'):
       if date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column])
       return df

    def filter_and_group_data(self, df, threshold = 50, column = 'rate', group_by = 'state'):
       # Filter the DataFrame
       filtered_df = df[df[column] > threshold]
       # Group by the specified column and count
       grouped_df = filtered_df.groupby(group_by)[column].count().reset_index(name='count')
       return grouped_df

    def aggregate_state_data(self, df, group_by='state'):
       grp_state = df.groupby(group_by)
       state_list = []

       for state, state_info in grp_state:
           state_dict = {
               'State': state,
               'Cases': state_info['date'].count(),
               'Rate': state_info['rate'].sum()
           }
           state_list.append(state_dict)

       state_df = pd.DataFrame(state_list)
       return state_df

    def save_dataframe_to_csv(self, df, filename):
        try:
            df.to_csv(filename, index=False)
            print(f"Data saved to {filename}")
        except Exception as e:
            print(f"Error saving data to {filename}: {e}")

    def aggregate_yearly_data(self, df, date_column='date', sum_column='rate'):
        df['Year'] = df[date_column].dt.year
        yearly_df = df.groupby('Year')[sum_column].sum().reset_index(name=f'Total {sum_column.capitalize()}')
        return yearly_df

    def finalize_plot(self, title='', xlabel='', ylabel=''):
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

    def plot_bar_chart(self, df, x, y, color='skyblue', xlabel='', ylabel='', title=''):
        plt.figure(figsize=(12, 6))
        plt.bar(df[x], df[y], color=color)
        plt.xticks(rotation=45, ha='right')
        self.finalize_plot(title, xlabel if xlabel else x, ylabel if ylabel else y)
    
    def plot_horizontal_bar_chart(self, df, x, y, color='skyblue', xlabel='', ylabel='', title=''):
        plt.figure(figsize=(12, 6))
        plt.barh(df[y], df[x], color=color)
        plt.xticks(rotation=45, ha='right')
        self.finalize_plot(title, xlabel if xlabel else x, ylabel if ylabel else y)

    def plot_pie_chart(self, df, values, labels, colors=None, title='', save_as_pdf=False, filename='pie_chart.pdf'):
        plt.figure(figsize=(8, 8))
        plt.pie(df[values], labels=df[labels], colors=colors, autopct='%1.1f%%', startangle=140)
        self.finalize_plot(title)
        
        if save_as_pdf:
            try:
                plt.savefig(filename, format='pdf')
                print(f"Pie chart saved as {filename}")
            except Exception as e:
                print(f"Error saving pie chart as PDF: {e}")
    
    def plot_scatter_plot(self, df, x, y, color='blue', xlabel='', ylabel='', title='', size=100, alpha=0.7):
        plt.figure(figsize=(12, 6))
        plt.scatter(df[x], df[y], color=color, s=size)
        self.finalize_plot(title, xlabel if xlabel else x, ylabel if ylabel else y)

    def plot_box_plot(self, df, x=None, y=None, color='skyblue', xlabel='', ylabel='', title='', rotate_labels=True):
        plt.figure(figsize=(12, 6))
        sns.set(style="whitegrid")  # Improve the background style
        sns.boxplot(x=df[x], y=df[y], palette=[color])
        
        if rotate_labels:
            plt.xticks(rotation=45, ha='right')  # Rotate x labels to prevent overlap
        
        self.finalize_plot(title, xlabel if xlabel else x, ylabel if ylabel else y)

    def plot_histogram(self, df, column, bins=10, color='purple', xlabel='', ylabel='', title=''):
        plt.figure(figsize=(12, 6))
        plt.hist(df[column], bins=bins, color=color)
        self.finalize_plot(title, xlabel if xlabel else column, ylabel if ylabel else 'Frequency')

    def calculate_statistics(self, df, column='rate'):
        rate_array = df[column].to_numpy()
        mean_rate = np.mean(rate_array)
        median_rate = np.median(rate_array)
        std_dev_rate = np.std(rate_array)
        return mean_rate, median_rate, std_dev_rate

    def filter_high_rates(self, df, threshold=50):
        rate_array = df['rate'].to_numpy()
        high_rate_array = rate_array[rate_array > threshold]
        return high_rate_array

    def adjust_and_combine_rates(self, df, additional_rates):
        rate_array = df['rate'].to_numpy()
        adjusted_rate_array = rate_array + additional_rates
        return adjusted_rate_array