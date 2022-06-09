#pip install numpy pandas seaborn scipy matplotlib arviz

import numpy   as np
import pandas  as pd
from   scipy   import stats
import seaborn as ans
import arviz   as az
import matplotlib.pyplot as plt

import pymc3 as pm

''' 2.3.4 pyMC3によるモデリング

    pyMC3 を使い, MCMCによってベルヌーイ分布のパラメータを予測するプログラム
    動かしてみると, たしかに真の分布（ベルヌーイ分布）っぽい定常分布が得られる！

    Variables:
        theta: 推論対象の事前分布
        y    : 尤度関数

'''
def chap2():
    y_obs = [1, 0, 0, 1, 1, 1, 0, 1, 1, 0]

    ## モデルの構築
    with pm.Model() as model:
        theta = pm.Uniform("theta", lower=0, upper=1)
        y = pm.Bernoulli("y", p=theta, shape=len(y_obs), observed=y_obs)

    print(model.basic_RVs)


    ## 統計量の可視化
    ## まずはMCMCで推論

    with model:
        trace = pm.sample(
            draws =6000,
            tune  =2000,
            step  =pm.NUTS(),
            chains=3,
            random_seed=1,
            return_inferencedata=True
        )

        az.plot_trace(trace)
        az.summary   (trace)
chap2()
plt.show()