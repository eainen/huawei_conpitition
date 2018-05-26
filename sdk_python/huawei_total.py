#encoding=utf-8
from operator import mul,add,sub
from math import isinf,fabs,exp
from copy import copy,deepcopy
import predictor

def fit_ar1_notrend(parama,*args):
    global error_list,data
    y_fit=map(float,list(data))
    mu=sum(y_fit)/(1.0*len(y_fit))
    y_fit=[i-mu for i in y_fit]
    error_list=[0]*len(y_fit)
    for i in range(1,len(y_fit)):
        ar_coff_sum=reduce(lambda sum_ar,x:sum_ar+x,map(mul,y_fit[i-1:i][::-1],parama[:i]),0)
        error_list[i]=y_fit[i]-ar_coff_sum
    return sum(map(lambda x: pow(x,2),error_list))


def DF_TEST():
    global data,error_list
    star_ar1_parama=[0.1]
    paramas,fvale=newdun_bfgs(fit_ar1_notrend,star_ar1_parama)
    return fabs(paramas[0])

def diff(flavors): 
    return map(sub,flavors[1:],flavors[:-1])

def value_after_diff(flavors,fore_value):
    diff_value=[0]*len(fore_value)
    for i in range(1,len(fore_value)+1):
        diff_value[i-1]=flavors[-1]+sum(fore_value[:i])
    return diff_value  

def mat_dot(A, B):
    return [[reduce(lambda sum_, (x, y): sum_ + x * y, zip(a, b), 0) for b in zip(*B)] for a in A]

def mat_T(A, ndim=2):
    if ndim == 1:
        return map(list, zip(A))
    else:
        return map(list, zip(*A))

def mat_add(A, B):
    return [map(add, i, j) for i, j in zip(A, B)]

def mat_sub(A, B):
    return [map(sub, i, j) for i, j in zip(A, B)]

def pk_find(A, B, ndim=1):
    return [[-j for j in i] for i in mat_dot(A, mat_T(B, ndim))]

def vecnorm(vec):
    return reduce(lambda sum_, x: sum_ + x, [i ** 2 for i in vec], 0) ** (1.0 / 2)

def approx_gradien(f, xk):
    #  epsilon:interval of gradient
    epsilon = [1.4901161193847656e-08] * len(xk)
    ei = [0.0] * len(xk)
    grad = [0.0] * len(xk)
    for i in range(len(xk)):
        ei[i] = 1.0
        ei = map(mul, ei, epsilon)
        xk_ = map(add, ei, xk)
        grad[i] = (f(xk_) - f(xk)) / epsilon[i]
    return grad

def newdun_bfgs(fun,x0):
    #ps,gtol Control gradient,maxk:iterator,
    #input:fun,star_parama
    #output：x* ,f_value
    gtol = 1.0000000000000001e-05
    maxk = len(x0)*200
    gfk=approx_gradien(fun,x0)
    k = 0
    N = len(x0)
    I=[[1 if i==j else 0 for j in range(N) ] for i in range(N)]
    HK=I
    xk=x0
    sk=[2*gtol]
    gnorm = vecnorm(gfk)
    rho = 0.55
    sigma = 0.4
    gama = 0.7


    while (gnorm > gtol) and (k < maxk) :
        pk = pk_find(HK, gfk)
        m = 0
        mk = 0
        # 用Wolfe方法求步长
        try :
            while m < 30:
                xkppk=map(add,xk ,[rho**m*i for i in list(zip(*pk)[0])])
                gfkp1 = approx_gradien(fun,xkppk)
                if fun(xkppk) < fun(xk)+sigma*rho**m*mat_dot([gfk],pk)[0][0] and \
                mat_dot([xkppk], pk)[0][0] >=  gama*mat_dot([gfk],pk)[0][0]:
                    mk = m
                    break
                m += 1
        except :
            break

        #start iterable
        xkp1 = map(add,xk ,[rho**mk*i for i in list(zip(*pk)[0])])
        #print "the"+str(k)+"result of iterable is ："+str(xkp1)
        sk = map(sub,xkp1,xk)
        xk=xkp1
        gfkp1=approx_gradien(fun,xkp1)
        yk = map(sub,gfkp1,gfk)
        gfk=gfkp1
        k+=1
        gnorm = vecnorm(gfk)
        if (gnorm <=gtol):
            break
        try :
            skTyk=1.0/mat_dot([sk],mat_T(yk,1))[0][0]
        except ZeroDivisionError:
            skTyk = 1000.0
        if isinf(skTyk):
            skTyk = 1000.0
        if skTyk > 0:
            A1 = mat_sub(I,[[j*skTyk for j in i] for i in mat_dot(mat_T(sk,1),[yk])])
            A2 = mat_T(A1)
            A3 =[[ j*skTyk for j in i]for i in mat_dot(mat_T(sk,1),[sk])]
            HKP1 =mat_add(mat_dot(A1,mat_dot(HK,A2)),A3)
            HK=HKP1
    return xk,fun(xk)

#mean
def haha_predict(flavors,steps,mean_step):
    ha_data=copy(flavors)
    fore_value=[0]*steps
    for i in range(steps):
        fore_value[i]=sum(ha_data[-mean_step:])/(1.0*mean_step)
        ha_data.append(fore_value[i])
    return fore_value

# merge model via LinesRegreesion
#parama start_num for lwlr model
def standRegres_cost(paramas,*args):
    k_constant=paramas[0]
    coff=paramas[1:]
    sr_errlist=[0]*len(sr_label)
    for i in range(len(sr_label)):
        sr_errlist[i]=sr_label[i]-k_constant-reduce(lambda sum_sr,x:sum_sr+x,map(mul,sr_data[i],coff),0)
    return  sum(map(lambda x: pow(x,2),sr_errlist))/(1.0*len(sr_label))

def standRegres(xArr,yArr,model_num):
    global sr_data,sr_label
    sr_data=deepcopy(xArr)
    sr_label=copy(yArr)
    sr_star_parama=[0.1]*(model_num+1)
    sr_ws,sr_fvale=newdun_bfgs(standRegres_cost,sr_star_parama)
    return sr_ws

def sR_pre_insample(xArr,yArr,model_num,start_num=1):
    sr_yArr=yArr[start_num:]
    lenth_num=len(sr_yArr)
    sr_xArr=[ i[-lenth_num:] for i in xArr]
    sr_xArr=[list(i) for i in zip(*sr_xArr)]
    sr_ws=standRegres(sr_xArr,sr_yArr,model_num)
    sr_predict=[0]*lenth_num
    for i in range(lenth_num):
        sr_predict[i]=sr_ws[0]+reduce(lambda sum_sr,x:sum_sr+x,map(mul,sr_xArr[i],sr_ws[1:]),0)
    return sr_predict

def sR_pre_forecast(xArr,yArr,fc_xArr,steps,model_num,start_num=0):
    sr_yArr=yArr[start_num:]
    lenth_num=len(sr_yArr)
    sr_xArr=[ i[-lenth_num:] for i in xArr]
    sr_xArr=[list(i) for i in zip(*sr_xArr)]
    sr_fc_xArr=[list(i) for i in zip(*fc_xArr)]
    sr_ws=standRegres(sr_xArr,sr_yArr,model_num)
    sr_fc_value=[0]*steps
    for i in range(steps):
        sr_fc_value[i]=sr_ws[0]+reduce(lambda sum_sr,x:sum_sr+x,map(mul,sr_fc_xArr[i],sr_ws[1:]),0)
    return sr_fc_value

#LWLR
def creat_data(flavors,var_num):
    x_data=[]
    y_label=flavors[var_num:]
    for i in range(var_num,len(flavors)):
        tmp=flavors[i-var_num:i][::-1]
        tmp.insert(0,1.0)
        x_data.append(tmp)
    return x_data,y_label


def lwlr_cost_function(paramas,*args):
    error_list=[0]*len(y_label)
    for i in range(len(y_label)):
        error_list[i]=weights[i][i]*pow((y_label[i]-reduce(lambda sum_x,x:sum_x+x,map(mul,x_data[i],paramas),0)),2)
    return sum(error_list)


def lwlr(testPoint,xArr,yArr,paramas,k=1.0):
    global weights
    m = len(xArr)
    weights = [[0]*m for i in range(m)]
    for j in range(m):                      
        diffMat = map(sub,testPoint ,xArr[j])
        weights[j][j] = exp(mat_dot([diffMat],mat_T(diffMat,1))[0][0]/(-2.0*k**2))
    ws,fvale=newdun_bfgs(lwlr_cost_function,paramas)
    return reduce(lambda sum_x,x:sum_x+x,map(mul,testPoint,ws),0)

def lwlr_predict_insample(flavors,var_num,k=1.0):
    global x_data,y_label
    x_data,y_label=creat_data(flavors,var_num)
    y_label_copy=copy(y_label)
    y_forecast=[0]*len(x_data)
    start_paramas=[0.1]*(var_num+1)    
    for i in range(len(x_data)):       
        y_forecast[i]=lwlr(x_data[i],x_data,y_label,start_paramas,k)
    for i in range(var_num):
        y_forecast.insert(0,None)
    return y_forecast


def lwlr_predict(flavors,var_num,steps,k=1.0):
    global x_data,y_label
    x_data,y_label=creat_data(flavors,var_num)
    y_label_copy=copy(y_label)
    y_forecast=[0]*steps
    start_paramas=[0.1]*(var_num+1)
    for i in range(steps):
        testArr=y_label_copy[-var_num:][::-1]
        testArr.insert(0,1.0)
        y_forecast[i]=lwlr(testArr,x_data,y_label,start_paramas,k)
        y_label_copy.append(y_forecast[i])
    return y_forecast

def armamodel_trend_OLS(parama,*args):
    global order,data,error_list
    order_=order
    y_fit=map(float,list(data))
    parama=list(parama)
    k_ar,k_ma=order_
    k_constant=parama[0]
    coff=parama[1:]
    error_list=[0]*len(y_fit)
    #fit ar
    if k_ar>0 and k_ma==0:
        for i in range(k_ar,len(y_fit)):
            ar_coff_sum=reduce(lambda sum_ar,x:sum_ar+x,map(mul,y_fit[i-k_ar:i][::-1],coff[:i]),0)
            error_list[i]=y_fit[i]-k_constant-ar_coff_sum
    #fit arma
    if k_ar>0 and k_ma>0:
        for i in range(len(y_fit)):
            ar_coff_sum=reduce(lambda sum_ar,(x,y):sum_ar+(x*y),zip(y_fit[:i][::-1],coff[:k_ar]),0)
            ma_coff_sum=reduce(lambda sum_ma,(x,y):sum_ma+(x*y),zip(error_list[:i][::-1],coff[k_ar:k_ar+k_ma]),0)
            error_list[i]=y_fit[i]-k_constant-ar_coff_sum+ma_coff_sum
    #fit ma
    if k_ar==0 and k_ma>0:
        for i in range(len(y_fit)):
            ma_coff_sum=reduce(lambda sum_ma,(x,y):sum_ma+(x*y),zip(error_list[:i][::-1],coff[:i]),0)
            error_list[i]=y_fit[i]-k_constant+ma_coff_sum
    return sum(map(lambda x: pow(x,2),error_list))

def armamodel_notrend_OLS(parama,*args):
    global order,data,error_list
    order_=order
    y_fit=map(float,list(data))
    mu=sum(y_fit)/(1.0*len(y_fit))
    y_fit=[i-mu for i in y_fit]
    parama=list(parama)
    k_ar,k_ma=order_
    error_list=[0]*len(y_fit)
    #fit ar
    if k_ar>0 and k_ma==0:
        for i in range(k_ar,len(y_fit)):
            ar_coff_sum=reduce(lambda sum_ar,x:sum_ar+x,map(mul,y_fit[i-k_ar:i][::-1],parama[:i]),0)
            error_list[i]=y_fit[i]-ar_coff_sum

    #fit arma
    if k_ar>0 and k_ma>0:
        for i in range(len(y_fit)):
            ar_coff_sum=reduce(lambda sum_ar,(x,y):sum_ar+(x*y),zip(y_fit[:i][::-1],parama[:k_ar]),0)
            ma_coff_sum=reduce(lambda sum_ma,(x,y):sum_ma+(x*y),zip(error_list[:i][::-1],parama[k_ar:k_ar+k_ma]),0)
            error_list[i]=y_fit[i]-ar_coff_sum+ma_coff_sum
    #fit ma
    if k_ar==0 and k_ma>0:
        for i in range(len(y_fit)):
            ma_coff_sum=reduce(lambda sum_ma,(x,y):sum_ma+(x*y),zip(error_list[:i][::-1],parama[:i]),0)
            error_list[i]=y_fit[i]+ma_coff_sum
    return sum(map(lambda x: pow(x,2),error_list))




def armaorma_notrend_with_ar(parama,*args):
    global order,data,err_pre_2
    order_=order
    y_fit=map(float,list(data))
    mu=sum(y_fit)/(1.0*len(y_fit))
    y_fit=[i-mu for i in y_fit]
    parama=list(parama)
    k_ar,k_ma=order_
    L=max(k_fit_ar,k_ar,k_ma)
    error_list=[0]*len(y_fit)
    if k_ar>0 and k_ma>0:
        for  i in range(L,len(y_fit)):
            ar_coff_sum=reduce(lambda sum_ar,(x,y):sum_ar+(x*y),zip(y_fit[:i][::-1],parama[:k_ar]),0)
            ma_coff_sum=reduce(lambda sum_ma,(x,y):sum_ma+(x*y),zip(err_pre_2[:i][::-1],parama[k_ar:k_ar+k_ma]),0)
            error_list[i]=y_fit[i]-ar_coff_sum+ma_coff_sum
    if k_ar==0 and k_ma>0:
        for i in range(L,len(y_fit)):
            ma_coff_sum=reduce(lambda sum_ma,(x,y):sum_ma+(x*y),zip(err_pre_2[:i][::-1],parama[k_ar:k_ar+k_ma]),0)
            error_list[i]=y_fit[i]+ma_coff_sum
    return sum(map(lambda x: pow(x,2),error_list))

def armaorma_trend_with_ar(parama,*args):
    global order,data,err_pre_2
    order_=order
    y_fit=map(float,list(data))
    parama=list(parama)
    k_constant=parama[0]
    coff=parama[1:]
    k_ar,k_ma=order_
    L=max(k_fit_ar,k_ar,k_ma)
    error_list=[0]*len(y_fit)
    if k_ar>0 and k_ma>0:
        for  i in range(L,len(y_fit)):
            ar_coff_sum=reduce(lambda sum_ar,(x,y):sum_ar+(x*y),zip(y_fit[:i][::-1],coff[:k_ar]),0)
            ma_coff_sum=reduce(lambda sum_ma,(x,y):sum_ma+(x*y),zip(err_pre_2[:i][::-1],coff[k_ar:k_ar+k_ma]),0)
            error_list[i]=y_fit[i]-k_constant-ar_coff_sum+ma_coff_sum
    if k_ar==0 and k_ma>0:
        for i in range(L,len(y_fit)):
            ma_coff_sum=reduce(lambda sum_ma,(x,y):sum_ma+(x*y),zip(err_pre_2[:i][::-1],coff[k_ar:k_ar+k_ma]),0)
            error_list[i]=y_fit[i]-k_constant+ma_coff_sum
    return sum(map(lambda x: pow(x,2),error_list))

def fit_ar_notrend(parama,*args):
    global error_list,k_fit_ar
    y_fit=map(float,list(data))
    mu=sum(y_fit)/(1.0*len(y_fit))
    y_fit=[i-mu for i in y_fit]
    error_list=[0]*len(y_fit)
    for i in range(k_fit_ar,len(y_fit)):
        ar_coff_sum=reduce(lambda sum_ar,x:sum_ar+x,map(mul,y_fit[i-k_fit_ar:i][::-1],parama[:i]),0)
        error_list[i]=y_fit[i]-ar_coff_sum
    return sum(map(lambda x: pow(x,2),error_list))

def fit_ar_trend(parama,*args):
    global error_list,k_fit_ar
    y_fit=map(float,list(data))
    k_constant=parama[0]
    coff=parama[1:]
    error_list=[0]*len(y_fit)
    for i in range(k_fit_ar,len(y_fit)):
        ar_coff_sum=reduce(lambda sum_ar,x:sum_ar+x,map(mul,y_fit[i-k_fit_ar:i][::-1],coff[:i]),0)
        error_list[i]=y_fit[i]-k_constant-ar_coff_sum
    return sum(map(lambda x: pow(x,2),error_list))

def arma_predict_insample(parama,error):
    #必须带constant参数
    global order,data
    order_=order
    y_fit=list(data)
    parama=list(parama)
    k_ar,k_ma=order_
    k_constant=parama[0]
    k_ar_cof=parama[1:k_ar+1]
    k_ma_cof=parama[k_ar+1:]
    pre_value=[0]*len(y_fit)
    if k_ar>0:
        for i in range(len(y_fit)):
            ar_coff_sum=reduce(lambda sum_ar,(x,y):sum_ar+x*y,zip(y_fit[:i][::-1],k_ar_cof[:i]),0)
            if k_ma>0:
                ma_coff_sum=reduce(lambda sum_ma,(x,y):sum_ma+x*y,zip(error[:i][::-1],k_ma_cof[:i]),0)
            else:
                ma_coff_sum=0
            pre_value[i]=k_constant+ar_coff_sum-ma_coff_sum
    if k_ar==0 and k_ma>0:
        for i in range(len(y_fit)):
            ma_coff_sum=reduce(lambda sum_ma,(x,y):sum_ma+x*y,zip(error[:i][::-1],k_ma_cof[:i]),0)
            pre_value[i]=k_constant-ma_coff_sum
    return pre_value

def arma_forecast(steps,paramas,predict_method='OLS'):
    #have constant
    global order,data
    k_ar,k_ma=order
    k_constant=paramas[0]
    coff = paramas[1:]
    if predict_method=='OLS':
        #err_pre=map(sub,data,arma_predict_insample(paramas,error_list))
        y_pre=copy(data)
        forecast=[0]*steps
        if k_ar>0:
            for i in range (steps):
                ar_coff_sum=reduce(lambda sum_ar,x:sum_ar+x, map(mul,y_pre[-k_ar:][::-1],coff[:k_ar]),0)
                if k_ma>0:
                    ma_coff_sum=reduce(lambda sum_ma,x:sum_ma+x,map(mul,err_pre_2[-k_ma:][::-1],coff[k_ar:k_ar+k_ma]),0)
                else:
                    ma_coff_sum=0
                forecast[i]=k_constant+ar_coff_sum-ma_coff_sum
                y_pre.append(forecast[i])
                if k_ma>0:
                    err_pre_2.append(0)
            return forecast
        if k_ar==0 and k_ma>0:
            for i in range (steps):
                ma_coff_sum=reduce(lambda sum_ma,x:sum_ma+x,map(mul,err_pre_2[-k_ma:][::-1],coff[:k_ma]),0)
                forecast[i]=k_constant-ma_coff_sum
                err_pre_2.append(0)
    return forecast

def fit_arma_model(flavors,orders,trend='nc',star_para=None,k_fit_ars=1):
    #trend 
    #k_fit_ars: how many ar control arma
    #only one diff
    global data,order,err_pre_2,k_fit_ar
    k_fit_ar=k_fit_ars
    data=copy(flavors)
    #DF test
    order=copy(orders)
    k_ar,k_ma=order
    mu_=sum(flavors)/(1.0*len(flavors))
    if star_para :
        star_para=star_para
    elif star_para== None and trend=='nc':
        star_para=[0.1]*(k_ar+k_ma)
    else :
        star_para=[mu_]+[0.1]*(k_ar+k_ma)
    if k_ar>0 and k_ma==0:
        if trend=='nc':
            paramas,fvalue=newdun_bfgs(armamodel_notrend_OLS,star_para)
            paramas.insert(0,mu_)
            #paramas.insert(0,mu_/(1-1.0*sum(paramas[:k_ar])))
        if trend=='c':
            paramas, fvalue = newdun_bfgs(armamodel_trend_OLS, star_para)
    if (k_ar>0 and k_ma>0) or (k_ar==0 and k_ma>0):
        if trend=='nc':
            #k_fit_ar用来拟合多少阶的ar模型来逼近arma模型
            star_k_fit_ar_parama=[0.1]*k_fit_ar
            paramas,fvale=newdun_bfgs(fit_ar_notrend,star_k_fit_ar_parama)
            paramas.insert(0,mu_)
            #paramas.insert(0,mu_/(1-1.0*sum(paramas[0:k_ar])))
            #paramas.insert(0,mu_/(1+1.0*sum(paramas[0:k_ar])))
            #先修改order的值，再改回来
            order=(k_fit_ar,0)
            err_pre_2=map(sub,data,arma_predict_insample(paramas,error_list))
            order=copy(orders)
            star_k_fit_ar_parama=[0.1]*(k_ar+k_ma)
            paramas,fvale=newdun_bfgs(armaorma_notrend_with_ar,star_k_fit_ar_parama)
            paramas.insert(0,mu_)          
        if trend=='c':
            #k_fit_ar用来拟合多少阶的ar模型来逼近arma模型
            star_k_fit_ar_parama=[0.1]*(k_fit_ar+1)
            paramas,fvale=newdun_bfgs(fit_ar_trend,star_k_fit_ar_parama)
            #先修改order的值，再改回来
            order=(k_fit_ar,0)
            err_pre_2=map(sub,data,arma_predict_insample(paramas,error_list))
            order=copy(orders)
            star_k_fit_ar_parama=[0.1]*(k_ar+k_ma+1)
            paramas,fvale=newdun_bfgs(armaorma_trend_with_ar,star_k_fit_ar_parama)         
    return paramas

def fit_arima_model(flavors,orders,trend='nc',star_para=None,k_fit_ars=1):
    global data,order,err_pre_2,k_fit_ar,initial_data
    k_fit_ar=k_fit_ars
    initial_data=copy(flavors)
    data=diff(flavors)
    k_ar,diff_count,k_ma=orders
    order=(k_ar,k_ma)
    mu_=sum(data)/(1.0*len(data))
    if star_para :
        star_para=star_para
    elif star_para== None and trend=='nc':
        star_para=[0.1]*(k_ar+k_ma)
    else :
        star_para=[mu_]+[0.1]*(k_ar+k_ma)        
    if k_ar>0 and k_ma==0:
        if trend=='nc':
            paramas,fvalue=newdun_bfgs(armamodel_notrend_OLS,star_para)
            paramas.insert(0,mu_)
            #paramas.insert(0,mu_/(1-1.0*sum(paramas[:k_ma])))  
        if trend=='c':
            paramas, fvalue = newdun_bfgs(armamodel_trend_OLS, star_para)
    if (k_ar>0 and k_ma>0) or (k_ar==0 and k_ma>0):
        if trend=='nc':
            #k_fit_ar用来拟合多少阶的ar模型来逼近arma模型
            star_k_fit_ar_parama=[0.1]*k_fit_ar
            paramas,fvale=newdun_bfgs(fit_ar_notrend,star_k_fit_ar_parama)
            paramas.insert(0,mu_)
            #paramas.insert(0,mu_/(1-1.0*sum(paramas[0:k_ar])))
            #先修改order的值，再改回来
            order=(k_fit_ar,0)
            err_pre_2=map(sub,data,arma_predict_insample(paramas,error_list))
            order=(k_ar,k_ma)
            star_k_fit_ar_parama=[0.1]*(k_ar+k_ma)
            paramas,fvale=newdun_bfgs(armaorma_notrend_with_ar,star_k_fit_ar_parama)
            paramas.insert(0,mu_)
            #paramas.insert(0,mu_/(1-1.0*sum(paramas[0:k_ar])))
            #paramas.insert(0,mu_/(1+1.0*sum(paramas[0:k_ar])))      
        if trend=='c':
            #k_fit_ar用来拟合多少阶的ar模型来逼近arma模型
            star_k_fit_ar_parama=[0.1]*(k_fit_ar+1)
            paramas,fvale=newdun_bfgs(fit_ar_trend,star_k_fit_ar_parama)
            #先修改order的值，再改回来
            order=(k_fit_ar,0)
            err_pre_2=map(sub,data,arma_predict_insample(paramas,error_list))
            order=(k_ar,k_ma)
            star_k_fit_ar_parama=[0.1]*(k_ar+k_ma+1)
            paramas,fvale=newdun_bfgs(armaorma_trend_with_ar,star_k_fit_ar_parama)
    return paramas  
   
def auto_fit_forecast(flavors,orders,orders_diff,trend='nc',steps=1,star_para=None,k_fit_ars=1):
    #trend ：  
    #k_fit_ars: how many ar control arma
    #only one diff
    global data,order,err_pre_2,k_fit_ar
    #DF test
    data=copy(flavors)
    df_test=DF_TEST()
    if df_test<1:
        paramas=fit_arma_model(flavors,orders,trend,star_para,k_fit_ars)
        forecast_value = arma_forecast(steps, paramas)          
        return forecast_value
    if df_test>=1 :       
        paramas=fit_arima_model(flavors,orders_diff,trend,star_para,k_fit_ars)
        forecast_value = arma_forecast(steps, paramas)
        forecast_value_2=value_after_diff(initial_data,forecast_value)
        return forecast_value_2

#merge model via linesRegression
      
def merge_model(flavors,trend='nc',steps=1,star_para=None,k_fit_ars=1,start_num=0):
    global data,order,err_pre_2,k_fit_ar
    data=copy(flavors)
    #model_type=[(1,0),(1,1),(2,0)]
    model_type=[(1,1),(1,0),(2,0)]
    is_diff_model=[(1,1,1),(1,1,0),(2,1,0)]
    #DF test
    df_test=DF_TEST()
    if df_test<1:
        sr_fit_data=[]
        st_fore_data=[]
        for i in model_type:
            paramas=fit_arma_model(flavors,i,trend,star_para,k_fit_ars)
            if (i[0]>0 and i[1]>0) or (i[0]==0 and i[1]>0):
                mm_predict_value=arma_predict_insample(paramas,err_pre_2)
            if i[0]>0 and i[1]==0 :
                mm_predict_value=arma_predict_insample(paramas,error_list)
            forecast_value = arma_forecast(steps, paramas)
            sr_fit_data.append(mm_predict_value)
            st_fore_data.append(forecast_value)
        mm_fc_value=sR_pre_forecast(sr_fit_data,flavors,st_fore_data,steps,len(model_type),start_num)
        return mm_fc_value
    if df_test>=1:
        sr_fit_data=[]
        st_fore_data=[]
        for i in is_diff_model:
            paramas=fit_arima_model(flavors,i,trend,star_para,k_fit_ars)
            if (i[0]>0 and i[2]>0) or (i[0]==0 and i[2]>0):
                mm_predict_value=arma_predict_insample(paramas,err_pre_2)
            if i[0]>0 and i[2]==0 :
                mm_predict_value=arma_predict_insample(paramas,error_list)
            forecast_value = arma_forecast(steps, paramas)
            sr_fit_data.append(mm_predict_value)
            st_fore_data.append(forecast_value)
        mm_fc_value=sR_pre_forecast(sr_fit_data,data,st_fore_data,steps,len(is_diff_model),start_num)
        forecast_value_2=value_after_diff(initial_data,mm_fc_value)
        return forecast_value_2

 
def best_arima_model(f,arma,arima,merge_day,outlier,flavor_list,sort_list,forecast_stat,forecast_end,trend='nc',k_fit_ars=[1,2,3,4],threshold=0):
    #arma:list,[(0,1),(1,0),(1,1)]
    #arima;list,[(0,1,1),(1,1,0),(1,1,1)]   
    global data,order,err_pre_2,k_fit_ar
    best_order_arma=0
    best_order_arima=0
    best_day=0
    best_outlier=0
    best_r=0
    jude_err=1e10
    #df_test=DF_TEST()
    #if df_test <1:
    for j in merge_day:
        for k in outlier:
            fg=predictor.agg_by_step(f, j, threshold, flavor_list , sort_list, outlier_type =k)
            div_end,mod_end=divmod(forecast_end,j)
            div_start,mod_start=divmod(forecast_stat,j)
            steps=div_end+1  if  mod_end else div_end
            data=fg[:-steps]
            vilia_data=fg[-steps:]
            if DF_TEST()<1:
                for i in arma:
                    for r in k_fit_ars:
                        paramas=fit_arma_model(data,i,trend,star_para=None,k_fit_ars=r)
                        forecast_value = arma_forecast(steps, paramas)
                        forecast_value[-1]= (mod_end/(1.0*j))*forecast_value[-1] if mod_end else forecast_value[-1]
                        forecast_value[div_start]=(1-mod_start/(1.0*j))*forecast_value[div_start] if mod_start else forecast_value[div_start]
                        forecast_value=forecast_value[div_start:]
                        cur_fore= int(round(sum([ 0 if t <0 else t for t in forecast_value])))
                        err_best=pow((sum(vilia_data)-cur_fore),2)
                        if  err_best <= jude_err:
                            jude_err=err_best
                            best_order_arma=i
                            best_day=j
                            best_outlier=k
                            best_r=r
            else :
                initial_data=copy(data)
                data=diff(data)
                for i in arima:
                    for r in k_fit_ars:
                        paramas=fit_arima_model(data,i,trend,star_para=None,k_fit_ars=r)
                        forecast_value = arma_forecast(steps, paramas)
                        forecast_value_2=value_after_diff(initial_data,forecast_value)
                        forecast_value_2[-1]= (mod_end/(1.0*j))*forecast_value_2[-1] if mod_end else forecast_value_2[-1]
                        forecast_value_2[div_start]=(1-mod_start/(1.0*j))*forecast_value_2[div_start] if mod_start else forecast_value_2[div_start]
                        forecast_value_2=forecast_value_2[div_start:]
                        cur_fore= int(round(sum([ 0 if t <0 else t for t in forecast_value_2])))
                        err_best=pow((sum(vilia_data)-cur_fore),2)
                        if err_best <= jude_err:
                            jude_err=err_best
                            best_order_arima=i
                            best_day=j
                            best_outlier=k
                            best_r=r

    data=predictor.agg_by_step(f, best_day, threshold, flavor_list , sort_list, outlier_type =best_outlier)
    if best_order_arma:
        order=best_order_arma
    else:
        order=(1,1)
    if best_order_arima:
        orders_diff=best_order_arima
    else:
        orders_diff=(1,1,1)
    div_end,mod_end=divmod(forecast_end,best_day)
    div_start,mod_start=divmod(forecast_stat,best_day)
    steps=div_end+1  if  mod_end else div_end
    forecast_value=auto_fit_forecast(data,order,orders_diff,trend,steps=steps,star_para=None,k_fit_ars=best_r)
    forecast_value[-1]= (mod_end/(1.0*best_day))*forecast_value[-1] if mod_end else forecast_value[-1]
    forecast_value[div_start]=(1-mod_start/(1.0*best_day))*forecast_value[div_start] if mod_start else forecast_value[div_start]
    forecast_value=forecast_value[div_start:]
    return forecast_value






