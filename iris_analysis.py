# Step 1: Import necessary libraries
import pandas as pd  # pandas for data manipulation and analysis
import matplotlib.pyplot as plt  # matplotlib for creating plots
import seaborn as sns  # seaborn for more aesthetically pleasing plots
from sklearn.datasets import load_iris  # sklearn to load the Iris dataset

# Step 2: Load the Iris dataset
try:
    iris_raw = load_iris()  # Load the Iris dataset from sklearn
    # Convert it into a pandas DataFrame for easier manipulation
    df = pd.DataFrame(iris_raw.data, columns=iris_raw.feature_names)
    # Add the 'species' column to the DataFrame, which stores the target labels
    df['species'] = pd.Categorical.from_codes(iris_raw.target, iris_raw.target_names)
    print("✅ Dataset loaded successfully.")
except Exception as e:
    # In case there is an error while loading the dataset
    print("❌ Error loading dataset:", e)

# Step 3: Display the first 5 rows to inspect the dataset
print("\nFirst 5 rows:")
print(df.head())  # This will give you a quick look at the first five rows

# Step 4: Check the info of the dataset (data types, number of non-null values)
print("\nInfo:")
print(df.info())  # This tells us the structure and type of each column in the DataFrame

# Step 5: Check for any missing values in the dataset
print("\nMissing Values:")
print(df.isnull().sum())  # This will show the count of missing values in each column

# Step 6: Basic statistical analysis for numerical columns (mean, std, min, max, etc.)
print("\nDescribe:")
print(df.describe())  # Summary of statistics for all numerical columns

# Step 7: Group data by 'species' and calculate the average 'sepal length' for each species
print("\nMean Sepal Length by Species:")
print(df.groupby('species')['sepal length (cm)'].mean())  # Average sepal length per species

# Step 8: Data Visualization

sns.set(style="whitegrid")  # Set seaborn style for better aesthetics

# Line Chart: Shows the trend of 'sepal length' over the index
plt.figure()
plt.plot(df.index, df['sepal length (cm)'], label='Sepal Length')  # Plot the sepal length vs. index
plt.title('Line Chart of Sepal Length over Index')  # Title for the plot
plt.xlabel('Index')  # X-axis label
plt.ylabel('Sepal Length (cm)')  # Y-axis label
plt.legend()  # Display legend
plt.show()  # Display the plot

# Bar Chart: Average 'petal length' for each species
plt.figure()
sns.barplot(x='species', y='petal length (cm)', data=df)  # Bar plot comparing petal length for each species
plt.title('Average Petal Length per Species')  # Title for the plot
plt.show()  # Display the plot

# Histogram: Distribution of 'sepal width' across all samples
plt.figure()
plt.hist(df['sepal width (cm)'], bins=15, color='green', edgecolor='black')  # Plot histogram of sepal width
plt.title('Histogram of Sepal Width')  # Title for the plot
plt.xlabel('Sepal Width (cm)')  # X-axis label
plt.ylabel('Frequency')  # Y-axis label
plt.show()  # Display the plot

# Scatter Plot: Relationship between 'sepal length' and 'petal length' by species
plt.figure()
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df)  # Scatter plot colored by species
plt.title('Sepal Length vs Petal Length')  # Title for the plot
plt.xlabel('Sepal Length (cm)')  # X-axis label
plt.ylabel('Petal Length (cm)')  # Y-axis label
plt.legend()  # Display legend
plt.show()  # Display the plot
