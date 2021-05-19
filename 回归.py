import pandas as pd
from matplotlib import pyplot as plt
import sklearn.linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import statsmodels.api as sm
from  sklearn  import linear_model
import joblib
import numpy as np
from statsmodels.formula.api import ols
from  sklearn.preprocessing  import StandardScaler



#data
data=pd.read_excel(r'C:\Users\ant.zheng\Desktop\regre.xlsx')
print(data.head())


#观察单因素拟合情况
plt.rcParams['font.sans-serif']=['SimHei']   #显示中文
plt.rcParams['axes.unicode_minus']=False     #显示正负号
sns.pairplot(data,x_vars=['Spend','CTR','CVR','CPA','PCVR','PCPA','Impression','Clicks','Prime','Install','SCPA']
             ,y_vars='Sub',kind='reg',height=5,aspect=0.7)
plt.show()

#输出相关系数图
plt.figure()
temp=data.corr()
mask = np.zeros_like(temp)
mask[np.triu_indices_from(mask)] = True
cmap='YlGnBu'
with sns.axes_style("white"):
    ax = sns.heatmap(temp, mask=mask,cmap=plt.get_cmap(cmap), vmax=1, square=True)
plt.show()
#检测共线性情况
#方差膨胀因子
def vif(df,col_i):
    '''
    col_i:被检测的列名
    '''
    cols=list(df.columns)
    cols.remove(col_i)
    cols_noti=cols
    formula=col_i+'~'+'+'.join(cols_noti)
    r2=ols(formula,df).fit().rsquared
    return round(1./(1.- r2),2)
data_x=data[['Spend','CTR','CVR','CPA','PCVR','PCPA','Impression','Clicks','Prime','Install','SCPA']]
for i in data_x.columns:
    print('各项VIF：','\t',i,'\t',vif(df=data_x,col_i=i))

#多重共线性存在，考虑使用岭回归建模处理

#数据标准化
dt=data[['Spend','CTR','CVR','CPA','PCVR','PCPA','Impression','Clicks','Prime','Install','SCPA','Sub']]
ss=StandardScaler()
std_data=ss.fit_transform(dt)
x=std_data[:,:-1]
y=std_data[:,-1]
X=pd.DataFrame(x)
Y=pd.DataFrame(y)
X.columns=['Spend','CTR','CVR','CPA','PCVR','PCPA','Impression','Clicks','Prime','Install','SCPA']
Y.columns=['Sub']

#岭回归测试
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=2)
from sklearn.linear_model import Ridge
ridge=Ridge(alpha=3)
ridge.fit(xtrain,ytrain)
print('标准化系数:',ridge.coef_)
print('截距:',ridge.intercept_)
print('R^2:',round(ridge.score(xtest,ytest),3))
pre_y=ridge.predict(X)
plt.subplot(2,1,1)
plt.plot(range(len(data['Prime'])),Y)
plt.plot(range(len(data['Prime'])),pre_y)
plt.title('拟合曲线')
plt.subplot(2,1,2)
plt.scatter(data['Prime'],Y)
plt.scatter(data['Prime'],pre_y)
plt.title('拟合分布')
plt.tight_layout()
plt.show()

#交叉验证
from sklearn.linear_model import RidgeCV
ridgecv=RidgeCV(alphas=[0.01,0.1,0.2,0.5,1,2,3,5,7,10,20,50,100,200,300,500])
ridgecv.fit(xtrain,ytrain)
print('合适的α：',ridgecv.alpha_)

#岭迹图
alphas = [0.01,0.1,0.2,0.5,1,3,5,7,10,20,50,100,200,300,500,1000]
scores = []
for i, alpha in enumerate(alphas):
    ridgeRegression = linear_model.Ridge(alpha=alpha)
    ridgeRegression.fit(xtrain,ytrain)
    scores.append(ridgeRegression.score(xtest,ytest))
plt.plot(alphas,scores)
plt.show()

#保存模型
joblib.dump(ridge,r'C:\Users\ant.zheng\Desktop\ridge.pkl')





