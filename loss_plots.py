
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Read the Data
# Replace 'your_file.txt' with the path to your text file
data = []
with open('SAC_BASE_ANT.txt', 'r') as file:
    for line in file:
        parts = line.split(',')
        episode = int(parts[0].split(': ')[1])
        total_numsteps = int(parts[1].split(': ')[1])
        episode_steps = int(parts[2].split(': ')[1])
        reward = float(parts[3].split(': ')[1])
        data.append([episode, total_numsteps, episode_steps, reward])

# Step 2: Convert to DataFrame
df = pd.DataFrame(data, columns=['Episode', 'Total NumSteps', 'Episode Steps', 'Reward'])

sns.set_theme(style="darkgrid")

# Step 3: Plotting
# Plot total number of steps
#sns.lineplot(data=df, x='Episode', y='Total NumSteps')
#plt.title('Total Number of Steps per Episode')
#plt.show()
#
## Plot episode steps
#sns.lineplot(data=df, x='Episode', y='Episode Steps')
#plt.title('Steps per Episode')
#plt.show()


# Calculate the rolling mean and standard deviation
window_size = 30  # Define the window size for the rolling calculation
df['Rolling_Mean'] = df['Reward'].rolling(window=window_size).mean()
df['Rolling_Std'] = df['Reward'].rolling(window=window_size).std()

sns.set_theme(style="darkgrid")

# Plot the mean and variance (standard deviation) as a shaded area
#plt.figure(figsize=(14, 7))
sns.lineplot(data=df, x='Episode', y='Rolling_Mean', label='Mean Reward')
plt.fill_between(df['Episode'],
                 df['Rolling_Mean'] - df['Rolling_Std'],
                 df['Rolling_Mean'] + df['Rolling_Std'],
                 alpha=0.15)
plt.title('Reward per Episode with Mean and Variance')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.show()

'''
# Plot rewards
sns.lineplot(data=df, x='Episode', y='Reward')
plt.title('Reward per Episode')
plt.show()

'''

