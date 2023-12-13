from env import get_db_url
import pandas as pd
from pydataset import data
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from scipy import stats
from scipy.stats import pearsonr

def plot_categorical_and_continuous_vars(df, categorical_var, continuous_var):
    """
    Visualizes the relationship between a categorical variable and a continuous variable using three different plots.

    """
    # Set up the subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Boxplot
    sns.boxplot(x=categorical_var, y=continuous_var, hue='county', data=df, ax=axes[0])
    axes[0].set_title('Boxplot')

    # Plot 2: Violinplot
    sns.violinplot(x=categorical_var, y=continuous_var, hue='county', data=df, ax=axes[1])
    axes[1].set_title('Violinplot')

    # Plot 3: Swarmplot
    sns.swarmplot(x=categorical_var, y=continuous_var, hue='county',data=df, ax=axes[2])
    axes[2].set_title('Swarmplot')

    # Adjust layout
    plt.tight_layout()
    plt.show()

def plot_variable_pairs(df):
    '''
    Plots all pairwise relationships along with the regression line for each pair.

    '''
    sns.set(style="ticks")
    
    # Create a pairplot with regression lines
    sns.pairplot(df, kind='reg', height=3, aspect=1.2, corner=True)
    
    # Show the plot
    plt.show()

    
def explore_categorical(train, target, subgroup, alpha=0.05):
    '''
    Explore the relationship between a binary target variable and a categorical variable.

    Parameters:
    train: The training data split set.
    target (str): The name of the binary target variable.
    cat_var (str): The name of the categorical variable to explore.
    alpha (float): Significance level for hypothesis testing.

    '''
    # Print the name of the categorical variable
    print()
    print(subgroup, '&', target)
    print('')

    # Calculate the chi-squared test statistic, p-value, degrees of freedom, and expected values
    ct = pd.crosstab(train[subgroup], train[target], margins=True)
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    print(f"Chi2: {chi2}")
    print(f"P-value: {p}")
    print(f"Degrees of Freedom: {dof}")
    print('')

    # Check for statistical significance
    if p < alpha:
        print('The null hypothesis can be rejected due to statistical significance.')
        print('Ergo there is a relationship between the target variable and corresponding feature(s)')
    else:
        print('The null hypothesis cannot be rejected at the chosen significance level.')
        print('Ergo there is not a relationship between the target variable and corresponding feature(s)')
        
def explore_continuous(train, target, subgroups, alpha=0.05):
    '''
    Explore the relationship between a binary target variable and continuous variables.

    Parameters:
    - train: The training data split set.
    - target (str): The name of the binary target variable.
    - subgroups (list): The list of continuous variables to explore.
    - alpha (float): Significance level for hypothesis testing.
    '''
    # Loop through each continuous variable
    for subgroup in subgroups:
        # Print the name of the continuous variable
        print(f"\n{subgroup} & {target}\n")

        # Perform Pearson correlation test
        correlation_coefficient, p_value = pearsonr(train[target], train[subgroup])

        # Print results
        print(f"Pearson correlation coefficient for {subgroup}: {correlation_coefficient:.4f}, p-value: {p_value:.4f}\n")

        # Check for statistical significance
        if p_value < alpha:
            print('The null hypothesis can be rejected due to statistical significance.')
            print('Ergo there is a relationship between the target variable and corresponding feature(s)\n')
            print()
        else:
            print('The null hypothesis cannot be rejected at the chosen significance level.')
            print('Ergo there is not a relationship between the target variable and corresponding feature(s)\n')
            print()


        
def plot_all_continuous_vars(train, target, subgroups):
    '''
    Melt the dataset to "long-form" representation
    boxenplot of measurement x value with color representing the target variable. 
    '''
    sns.set(style="whitegrid", palette="muted")
    
    # Create subplots
    fig, axes = plt.subplots(nrows=len(subgroups), ncols=1, figsize=(8, 6 * len(subgroups)))

    for i, subgroup in enumerate(subgroups):
        # Melt the dataset for the current subgroup
        melt = train[[subgroup, target]].melt(id_vars=target, var_name=subgroup)

        # Create a boxenplot for the current subgroup
        sns.boxenplot(x=subgroup, y="appraisal", hue=target, data=melt, ax=axes[i])

        # Apply log scale to the y-axis if needed
        if subgroup in ['sqft', 'taxes']:
            axes[i].set_yscale('log')

        axes[i].set_xscale('log')  # Apply logarithmic scale to x
        
        axes[i].set_xlabel(subgroup)
        axes[i].set_ylabel('Value')
        axes[i].set_title(f'{subgroup} by {target}')

    plt.tight_layout()
    plt.show()
    print()
    print()
