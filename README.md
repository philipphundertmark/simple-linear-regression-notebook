# linear-regression-demo

This is a simple demo project using linear regression model to predict student test scores based on the amount of hours they study.

> You can just run ```pip install -r requirements.txt``` in terminal to install the necessary dependencies.

Type ```python linearRegression.py``` in your terminal to get optimized parameters **m** and **b** to predict the test scores given the hours of study as input.

Executing ```linearRegression.py``` will plot the cost after each gradient step to show it's converging to local minimum:
![Cost_per_iter](/cost_per_iteration.png)

The line which best fits the data is shown below:
![Line_best_fit](/data_visualized.png)


