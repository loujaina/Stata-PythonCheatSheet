# Stata ->> Python Cheat Sheet

N.B.: This Cheat Sheet is work-in-progress. I keep on adding commands as I learn them. 

#### Below are the list of Python Libraries that I always upload before I start any work
* `import numpy as np`
* `import pandas as pd`
* `import math as math`
* `import statsmodels.formula.api as smf`
* `from statsmodels.sandbox.regression.gmm import IV2SLS`
* `from patsy import dmatrices, dmatrix`
* `%matplotlib inline`
* `import matplotlib.pyplot as plt`
* `import seaborn as sn           # Library to create fancy graphs`

#### The table below shows Stata vs. Python commands to perform different operations
In Python commands sometimes I write the code to import the relevant package and sometimes not. But if you just import all the packages above in the beginning of the code, you should not need to import anything else as you go

| Operation           | Stata Command |  Python Command | 
| -------------       | ------------- | -------------   |
|_**Import Data**_    |               |                 |
| Read csv            | `import delimited "dir\filename.csv", clear` |  `df = pd.read_csv('dir/filename.dta')`  |
| Read Stata file     | `use "dir\filename.dta", clear`              |  `df = pd.read_csv('dir/filename.dta')`  |
| _**Generate variables**_ |          |                 |		
| y = x^2             | `gen y = x^2 ` |  `df["y"] = np.power(df["x"], 2)` |
| y = x + z           | `gen y = x + z` | `df["y"] = df["x"] + df["z"]`  |
| y = exp(x)	      | `gen y = exp(x)` | `import math as math`<br>`df[y] = np.exp(df.x)` |
| dummy = 1 if x > #  | `gen dummy = (x> #)`  |  `df["dummy"] = 0` <br> `df.loc[df["x"] > #, "dummy"] = 1`  |
| RV x ~ N(m,sd)      | `gen x=rnormal(m,sd)`  |  `import numpy as np`<br> `x = random.normal(loc=m, scale=sd, size=# of obs)`
|_**Drop**_           |               |                  |
| Drop obs if x > #   |	`drop if x > #` | `df.drop(df[df['x'] > #].index, inplace = True)` |
| Drop variable x     | `drop x`        | `df.drop(['x'], axis=1)`   |
| Drop variables x & y  | `drop x y`        | `df.drop(['x', 'y'], axis=1)`   |
|_**Summary Stats**_  |              |                |
|Summarize all variables | `sum`    |  `df.describe()`  |  
| Summarize only x and y | `sum x y` | `np.round(df.describe(), 2).T[['count','mean', 'std', 'min', 'max']]`
|Summary stats by group of variable x |     |  `df.groupby(['x']).describe()` |
|Correlation matrix |  `correlate x y z` | `df_corr = df[['x', 'y', 'z']].copy()`<br>`df.corr()` |
|_**Sorting**_  |               |                 |		
| Sort the data by variable x | `sort x` | `df.sort_values('x', ascending=True)` |
|_**Joining Dataset**_  |               |                 |		
| One-to-one merge  | `use "path\df1.dta", clear` <br> `merge 1:1 varname using "path\df2.dta"`     |    `df3 = pd.merge(df1, df2, on='varname')` <br> OR <br> `df3 = pd.merge(df1, df2, left_on="varname_df1", right_on="varname_df2")`  |
| Many-to-one merge  | `use "path\df1.dta", clear` <br> `merge m:1 varname using "path\df2.dta"`     |    `df3 = pd.merge(df1, df2, on='varname')` <br> OR <br> `df3 = pd.merge(df1, df2, left_on="varname_df1", right_on="varname_df2")`  |
| Append datasets vertically | `use "path\df1.dta", clear` <br> `append using "path\df2.dta"`   |    `pd.concat([df1, df2]), ignore_index=True)` <br> OR <br> `df1.append(df2)`  |
| Append datasets horizontally |    |  `pd.concat([df1, df2], axis='col')` | 
|_**Collapse data**_  |               |                 |		
| Frequency table     |	`collapse (count) x` | `collapsed_data = df.groupby('x')['x'].count()` |
| _**OLS**_           |               |                 |
| Simple reg of y on x | `reg y x`	| `import statsmodels.formula.api as smf`<br> `print(smf.ols(formula = "y ~ x", data=df).fit().summary())` |
| Generate predicted values of y | ` predict (xb) y_hat`    | `y_hat = smf.ols(formula = "y ~ x", data=data).fit().predict()` |
| Reg y on categorical variable | `reg y i.a`	| `import statsmodels.formula.api as smf`<br> `print(ols = smf.ols(formula = "y ~ C(a)", data=df).fit().summary())` |
|Fixed effects regression<br>(only unit FE)  |	`xtset cvar`<br>`xtreg y x, fe`<br>*OR*<br>`areg y x, absorb(cvar)` | `import statsmodels.formula.api as smf`<br> `print(fe = smf.ols(formula = "y ~ x + C(cvar)", data=df).fit().summary())` |  
| _**IV regression (2SLS)**_           |               |                 |
2sls of y on x with instrument z<br>(No controls) |  `ivreg y (x = z)` | `from statsmodels.sandbox.regression.gmm import IV2SLS`<br>`endog = df.y`<br>`exog=df.x`<br>`z = df.z`<br>`print(IV2SLS(y, x, instrument = z).fit().summary())` |
2sls of y on x1 with instrument z<br>(With control x2) |  `ivreg y x2 (x1 = z)` | `from statsmodels.sandbox.regression.gmm import IV2SLS`<br>`from patsy import dmatrices, dmatrix`<br>`y, x = dmatrices('y~ x1 + x2', df)`<br>`z = dmatrix('z + x2', df)`<br>`print(IV2SLS(y, x, instrument = z).fit().summary())` |
2sls of y on x with instrument z<br>(Add Fixed Effects) | `xtset cvar`<br>`xtivreg2 y (x = z), fe` | `from statsmodels.sandbox.regression.gmm import IV2SLS`<br>`from patsy import dmatrices, dmatrix`<br>`y, x = dmatrices('y~ x + C(cvar)', df)`<br>`z = dmatrix('z + C(cvar)', df)`<br>`print(IV2SLS(y, x, instrument = z).fit().summary())` |
| _**Probit**_           |               |                 |
| Probit model |    `probit y x1`    | `from statsmodels.discrete.discrete_model import Probit`<br>`X = sm.add_constant(df.x)`<br> `Y = df.y`<br> `print(Probit(Y, X).fit().summary())`    |
| Predict y_hat from probit            |   ` predict (xb) yhat_probit`          |  `yhat_probit = Probit(Y, X).fit().predict()`   |
| _**Plots**_           |               |                 |
| Simple histogram            |  `histogram x`   | `%matplotlib inline`<br>`import matplotlib.pyplot as plt`<br>`plt.hist(df.x)` |
| Simple scatterplot         | `scatter y x, xlab(x) ylab(y)` | `import matplotlib.pyplot as plt`<br>`plt.scatter(df.x, df.y)`<br>`plt.xlabel("x-label")`<br> `plt.ylabel("y-label")`<br> `plt.show()` |
| Overlay scatterplot and regression line |   |  After predicting y_hat:<br> `plt.scatter(df.x, df.y)`<br> `plt.plot(df.x, y_hat)`<br> `plt.xlabel("X")`<br> `plt.ylabel("Y")`| 

