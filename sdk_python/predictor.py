#encoding=utf-8
import datetime
import huawei_total
from operator import mul,add,sub
from math import isinf,fabs,log
from copy import copy
import random


def find_link_break(array):
    for item in array:
        values = item.split(" ")
        for i in range(len(values)):
            str_temp =str(values[i])
            if str_temp.find('\r\n') != -1:
                return 1
            if str_temp.find('\n') != -1:
                return 2

def get_params(array):
    dict={'params_0':[],'params_1':[],'params_2':[],'params_3':[],'params_4':[]}
    list=[]
    link = find_link_break(array)
#    print link
    if link == 1:
        link_char = "\r\n"
    else:
        link_char ="\n"
    for item in  array:
        values = item.split(" ")
#        print("test")
#        print values
        #judge huan hang fu
        for i in range(len(values)):
            if values[i] == link_char:
                continue
            list.append(values[i].replace(link_char,""))
#        print list
    for i in range(5):
        current_key = 'params_%d' % i
        if current_key == 'params_0':
            dict[current_key] = list[0:2]
        if current_key == 'params_1':
            current  = 3
            dict[current_key] = list[current]
        # record current positon
        if current_key == 'params_2':
            current+=1
            for j in range(int(dict.get('params_1'))):
                dict[current_key].append(list[(current):(current+3)])
                current += 3
        if current_key == 'params_3':
            dict[current_key] = list[current]
        if current_key == 'params_4':
            current += 1
            for j in range(2):
                dict[current_key].append(list[(current):(current+2)])
                current += 2
    if current == len(list) :
        print("params read success ")
    return dict


def find_right(dict,res_info,res_cpu,res_mem,flag_res):
    min = float("inf")
    minkey = ""
 #   print("current======== %.3f"%float(res_info))
    if res_info == 0:
        return minkey
    for key in dict.keys():
        if int(dict[key][0]) <=int(res_cpu) and int(dict[key][1]) <= int(res_mem):

            if abs(float(dict[key][2]) - float(res_info)) == min:
                if flag_res == 0:
                    if int(dict[key][0]) > int(dict[minkey][0]) :
                        minkey = key
                if flag_res == 1:
                    if int(dict[key][1]) > int(dict[minkey][1]) :
                        minkey = key
            if abs(float(dict[key][2]) - float(res_info)) < min :
                min = abs(float(dict[key][2]) - float(res_info))
                minkey = key
    return minkey

def dict_sort_key_print(dict,array):
    keys_list = list(dict.keys())
    list_num =[]
    for i in range(len(keys_list)):
        num = int(keys_list[i][6:])
        list_num.append(num)
    list_num.sort()
    for item in list_num:
        cur_key  = 'flavor%d' % item
        array.append(cur_key)
        array.append(str(dict[cur_key]))

def dict_count(dict):
    num=0
    for key in dict.keys():
        num += int(dict[key])
    return num
def put_fun(dict,list):
    total =list[0]
    dict_put={}
    count = 0
    res_cpu = dict['params_0'][0]
    res_mem = dict['params_0'][1]
    flag_res =0
    if dict['params_3'] == 'CPU':
#        print("cpu")
        phy_info = ("%.3f" % (float(res_mem) / float(res_cpu)))
        for list_par in dict['params_2']:
            count += 2
            dict_put[list_par[0]] = [int(list_par[1]) ,int(list_par[2])/1024,(float(list_par[2])/1024)/float(list_par[1]),int(list[count])]
    if dict['params_3']  == 'MEM':
#        print("mem")
        flag_res = 1
        phy_info = ("%.3f" % (float(res_cpu) / float(res_mem)))
        for list_par in dict['params_2']:
            count += 2
            dict_put[list_par[0]] = [int(list_par[1]),int(list_par[2])/1024,float(list_par[1])/(float(list_par[2])/1024),int(list[count])]
    print(dict_put)
#    print (phy_info)
#    print(res_mem)
#    print(res_cpu)
    phy_use_count = 1
    cpu_remain = res_cpu
    mem_remain = res_mem

    print(total)

    for key in dict_put.keys():
        if int(dict_put[key][3]) == 0:
            dict_put.pop(key)
    print dict_put
    res_info = phy_info
    dict_tmp = {}

    array = []
    index=0
    while total != 0:
        minkey = find_right(dict_put,res_info,cpu_remain,mem_remain,flag_res)
        if minkey == "":

            if flag_res == 0:
                rate = 1 - (float(cpu_remain) / float(res_cpu))
            else:
                rate = 1 - (float(mem_remain) / float(res_mem))
            print("**** physical rate =========== %f" % rate)

            res_info = phy_info
            cpu_remain = res_cpu
            mem_remain = res_mem
            list_tmp=[]
            list_tmp.append(str(phy_use_count))
            array.append(list_tmp)
            dict_sort_key_print(dict_tmp,array[index])
            dict_tmp = {}
            phy_use_count +=1
            index +=1

        else:
            if minkey in dict_tmp.keys():
                dict_tmp[minkey] +=1
            else:
                dict_tmp[minkey] =1
            total =int(total)- 1
            cpu_remain = int(cpu_remain) - int(dict_put[minkey][0])
            mem_remain = int(mem_remain)- int(dict_put[minkey][1])
            if cpu_remain == 0 or mem_remain == 0:
                res_info=0
            else:
                if flag_res == 0:
                    res_info = ("%.3f" % (float(mem_remain) / float(cpu_remain)))
                else:
                    res_info = ("%.3f" % (float(cpu_remain) / float(mem_remain)))

            dict_put[minkey][3] -= 1
            if int(dict_put[minkey][3]) == 0:
                dict_put.pop(minkey)
    if dict_tmp !={} :
        list_tmp=[]
        list_tmp.append(str(phy_use_count))
        array.append(list_tmp)
        dict_sort_key_print(dict_tmp,array[index])
        #need to judge if need to del
        if flag_res == 0:
            rate =1- (float(cpu_remain) / float(res_cpu))
        else:
            rate = 1- (float(mem_remain) / float(res_mem))
        print("****last physical rate =========== %f"%rate)
        num = dict_count(dict_tmp)
        print("the last num *********** %d"%num)
        if(num < 14):
            dict_del = dict_tmp
        else:
            dict_del ={}
    else :
        dict_del ={}
    return array,phy_use_count,dict_del


def params_check(list):
    total= list[0]
    count_num=int(0)
    for count in range(2,len(list),2):
        count_num += int(list[count])
    if int(total) == count_num:
        return 0
    else:
        return 1

def cal_day_count(start,end):
    start = start.split("-")
    end = end.split("-")
    index = datetime.date(int(end[0]),int(end[1]),int(end[2]))-datetime.date(int(start[0]),int(start[1]),int(start[2]))
    return index


def Days_process(array):
    cut_start =0
    cut_end =0
    for i in range(len(array)):
        values = array[0].replace("\t", " ")
        values = values.split(" ")
        values = int(values[1][6:])
        if values < 16:
            cut_start = i
            break;
    for i in range(len(array)):
        values = array[len(array)-i-1].replace("\t", " ")
        values = values.split(" ")
        values = int(values[1][6:])
        if values < 16:
            cut_end =len(array)-i
            break
    return array[cut_start:cut_end]

def read_train_data(array,dict):
    array = Days_process(array)
    print len(array)

    start = array[0].replace("\t", " ")
    start = start.split(" ")
    start = start[2]

    end = array[len(array)-1].replace("\t", " ")
    end = end.split(" ")
    end= end[2]

    Day_count = cal_day_count(start,end)
    pre_start_train_end=cal_day_count(end,dict["params_4"][0][0])
    pre_end_train_end=cal_day_count(end,dict["params_4"][1][0])
 #   print Day_count
 #   flavor_list = [[0 for col in range(15)]for row in range(Day_count.days+1)]
    flavor_list = [[0 for col in range(Day_count.days + 1)]for row in range(15)]
    sort_list=[[] for row in range(15)]
    last_index_day =0

    last_count =[0 for col in range(15)]

    for item in array:
        values = item.replace("\t", " ")
      #  values = values.replace("\r\n", "")
        values = values.split(" ")
        index_flavor = int(values[1][6:])
        if index_flavor >= 16:
            continue
        index_day = cal_day_count(start,values[2]).days

     #   flavor_list[index_day][index_flavor-1] +=1
        flavor_list[index_flavor - 1][index_day]  += 1
        if index_day != last_index_day:
            for j in range(15):
                if last_count[j]!= 0:
                    sort_list[j].append(last_count[j])
                    last_count[j]=0

        last_count[index_flavor-1] +=1
        last_index_day= index_day

    for j in range(15):
        if last_count[j] != 0:
            sort_list[j].append(last_count[j])
            last_count[j] = 0
    for j in range(len(sort_list)):
        sort_list[j].sort()

    return flavor_list,sort_list,pre_start_train_end.days-1,pre_end_train_end.days-1


## forecast
def get_outlier(li, k=1.5):
    # =============================================================================
    # li: list
    # k : params control boundary of outliers
    # return a threshold of outlier
    # =============================================================================
    length = len(li)
    if length < 4:
        return 0
    else:
        q25_index = 1.0 * length / 4
        q25 = 1.0 * (li[int(q25_index - q25_index % 1)] + li[
            int(q25_index - q25_index % 1)] - 1) / 2 if q25_index % 1 > 0 else li[int(q25_index) - 1]
        q75_index = 3.0 * length / 4
        q75 = 1.0 * (li[int(q75_index - q75_index % 1)] + li[
            int(q75_index - q75_index % 1)] - 1) / 2 if q75_index % 1 > 0 else li[int(q75_index) - 1]
        return q75 + k * (q75 - q25)

def agg_by_step(flavor, step, threshold, flavor_list , sort_list, outlier_type = 'repByThres'):
    # =============================================================================
    # int flavor: flavor type
    # step: days to aggregate
    # threshold: threshold to delete or fill the residual data
    # outlier_type: repByThres, 2nn, 3mean,5mean,7mean
    # =============================================================================
    li = flavor_list[flavor-1][:]
    con = []
    outlier_threshold = get_outlier(sort_list[flavor-1])
    
    if outlier_type == 'repByThres':
        for i in range(len(li) - 1, len(li) % step - 1, -step):
            sub_li = [x if x < outlier_threshold else outlier_threshold for x in li[i:i - step:-1]]
            con.append(sum(sub_li))
    elif outlier_type == '2nn':
        for i in range(len(li) - 1, len(li) % step - 1, -step):
            tmp = li[i:i - step:-1]
            sub_li = []
            for num in range(len(tmp)):
                if tmp[num] > outlier_threshold:
                    if num == 0:
                        sub_li.append((tmp[num+1]+tmp[num+2]) / 2.0)
                    elif num == (len(tmp)-1):
                        sub_li.append((tmp[num-1] + tmp[num-2]) / 2.0 )
                    else:
                        sub_li.append((tmp[num-1]+tmp[num+1]) / 2.0)
                else:
                    sub_li.append(tmp[num])
            con.append(sum(sub_li))
    elif outlier_type == '3mean':
        k = 3
        len_li = len(li)
        for num in range(len_li):
            if li[num] > outlier_threshold:
                if num < k/2:
                    li[num] = 1.0 * sum(li[:k]) / k
                elif num < (len_li-k/2-1):
                    li[num] = 1.0 * sum(li[(num-k/2):(num+k/2+1)]) / k
                else:
                    li[num] =  1.0 * sum(li[(len_li-k):len_li]) / k
        
        for i in range(len(li) - 1, len(li) % step - 1, -step):
            con.append(sum(li[i:i - step:-1]))
    elif outlier_type == '5mean':
        k = 5
        len_li = len(li)
        for num in range(len_li):
            if li[num] > outlier_threshold:
                if num < k/2:
                    li[num] = 1.0 * sum(li[:k]) / k
                elif num < (len_li-k/2-1):
                    li[num] = 1.0 * sum(li[(num-k/2):(num+k/2+1)]) / k
                else:
                    li[num] =  1.0 * sum(li[(len_li-k):len_li]) / k
        for i in range(len(li) - 1, len(li) % step - 1, -step):
            con.append(sum(li[i:i - step:-1]))
    elif outlier_type == '7mean':
        k = 7
        len_li = len(li)
        for num in range(len_li):
            if li[num] > outlier_threshold:
                if num < k/2:
                    li[num] = 1.0 * sum(li[:k]) / k
                elif num < (len_li-k/2-1):
                    li[num] = 1.0 * sum(li[(num-k/2):(num+k/2+1)]) / k
                else:
                    li[num] =  1.0 * sum(li[(len_li-k):len_li]) / k
        for i in range(len(li) - 1, len(li) % step - 1, -step):
            con.append(sum(li[i:i - step:-1]))
    elif outlier_type is None:
        for i in range(len(li) - 1, len(li) % step - 1, -step):
            con.append(sum(li[i:i - step:-1]))
            
    if 1.0 * (i - step) / step > threshold:
        con.append(sum(li[0:step]))

    con.reverse()
    return con[:]



def xuanxue_forecast(data, forecast_days, step):
    li = data[:]
    end = len(li)-1
    forecast_step = forecast_days / step
    res =  forecast_days % step

    for i in range(1,forecast_step+1):
        li.append(0.7 * li[-1] + 0.3 * li[-2])

    result = sum(li[(end+1):(end+forecast_step+1)]) + li[end+forecast_step] * 1.0 * res/step
    return result



def get_flavor_to_forecast(dict):
    params = dict['params_2']
    flavors = []
    for i in range(len(params)):
        index_flavor = int(params[i][0][6:])
        flavors.append(index_flavor)
    print flavors
    return flavors[:]

def test_outlier(data):
    mean = 1.0 * sum(data) / len(data)
    outlier_thres = get_outlier(data)
    li = [mean if x > outlier_thres else x for x in data]
    return li[:]


def cv_agg_by_step(data, step, threshold=0.6, outlier_type = 'repByThres'):
    # =============================================================================
    # int flavor: flavor type
    # step: days to aggregate
    # threshold: threshold to delete or fill the residual data
    # outlier_type: repByThres, 2nn, 3mean
    # =============================================================================
    li = data[:]
    con = []
    sort_li = [x for x in li if x !=0]
    sort_li.sort()
    outlier_threshold = get_outlier(sort_li)
    
    if outlier_type == 'repByThres':
        for i in range(len(li) - 1, len(li) % step - 1, -step):
            sub_li = [x if x < outlier_threshold else outlier_threshold for x in li[i:i - step:-1]]
            con.append(sum(sub_li))
    elif outlier_type == '2nn':
        for i in range(len(li) - 1, len(li) % step - 1, -step):
            tmp = li[i:i - step:-1]
            sub_li = []
            for num in range(len(tmp)):
                if tmp[num] > outlier_threshold:
                    if num == 0:
                        sub_li.append((tmp[num+1]+tmp[num+2]) / 2.0)
                    elif num == (len(tmp)-1):
                        sub_li.append((tmp[num-1] + tmp[num-2]) / 2.0 )
                    else:
                        sub_li.append((tmp[num-1]+tmp[num+1]) / 2.0)
                else:
                    sub_li.append(tmp[num])
            con.append(sum(sub_li))
    elif outlier_type == '3mean':
        k = 3
        len_li = len(li)
        for num in range(len_li):
            if li[num] > outlier_threshold:
                if num < k/2:
                    li[num] = 1.0 * sum(li[:k]) / k
                elif num < (len_li-k/2-1):
                    li[num] = 1.0 * sum(li[(num-k/2):(num+k/2+1)]) / k
                else:
                    li[num] =  1.0 * sum(li[(len_li-k):len_li]) / k
        
        for i in range(len(li) - 1, len(li) % step - 1, -step):
            con.append(sum(li[i:i - step:-1]))
    elif outlier_type == '5mean':
        k = 5
        len_li = len(li)
        for num in range(len_li):
            if li[num] > outlier_threshold:
                if num < k/2:
                    li[num] = 1.0 * sum(li[:k]) / k
                elif num < (len_li-k/2-1):
                    li[num] = 1.0 * sum(li[(num-k/2):(num+k/2+1)]) / k
                else:
                    li[num] =  1.0 * sum(li[(len_li-k):len_li]) / k
        for i in range(len(li) - 1, len(li) % step - 1, -step):
            con.append(sum(li[i:i - step:-1]))
    elif outlier_type == '7mean':
        k = 7
        len_li = len(li)
        for num in range(len_li):
            if li[num] > outlier_threshold:
                if num < k/2:
                    li[num] = 1.0 * sum(li[:k]) / k
                elif num < (len_li-k/2-1):
                    li[num] = 1.0 * sum(li[(num-k/2):(num+k/2+1)]) / k
                else:
                    li[num] =  1.0 * sum(li[(len_li-k):len_li]) / k
        for i in range(len(li) - 1, len(li) % step - 1, -step):
            con.append(sum(li[i:i - step:-1]))
    elif outlier_type is None:
        for i in range(len(li) - 1, len(li) % step - 1, -step):
            con.append(sum(li[i:i - step:-1]))
            
    if 1.0 * (i - step) / step > threshold:
        con.append(sum(li[0:step]))

    con.reverse()
    return con[:]

def exponential_smoothing_forecast(data, step, alpha=0.5, cut=20):
    li = data[:]
    end = len(li)-1
    
    for i in range(1,step+1):
        li.append(sum([li[len(li)-p] * pow(alpha, p) for p in range(1,min(len(li)+i+1, cut))]))
    result = li[(end+1):(end+step+1)]
    return result

def cv_exponential_smoothing_forecast(data, forecast_steps, alpha=0.45, cut=20):

    li = data[:]
    end = len(li)-1
    for i in range(1,forecast_steps+1):
        li.append(sum([li[len(li)-p] * pow(alpha, p) for p in range(1,min(len(li)+i+1, cut))]))

    return li[(end+1):(end+forecast_steps+1)]

def cv_exp_smoothing_estimate(cv_data, alpha, forecast_end, forecast_stat, other_params):
    cv_score = {}
    cv_score['best_ind'] = 0
    cv_score['best_score'] = 9999999999
    cv_score['fit'] = []
    
    cnt = 0
    
    data_merge = other_params['data_merge']
    outier = other_params['outier']
    
    for out in outier:
        for day_step in data_merge:
            for p1 in alpha:
                score = []
                for train, test in cv_data:
                    
                    div_end,mod_end=divmod(forecast_end,day_step)
                    div_start,mod_start=divmod(forecast_stat,day_step)
                    steps=div_end+1  if  mod_end else div_end
                    
                    agg_data = cv_agg_by_step(data=train, step=day_step, threshold=0.6,outlier_type=out)
                    pre_value = cv_exponential_smoothing_forecast(data = agg_data, forecast_steps=steps, alpha=p1)
                    
                    pre_value[-1]= (mod_end/(1.0*day_step))*pre_value[-1] if mod_end else pre_value[-1]
                    pre_value[div_start]=(1-mod_start/(1.0*day_step))*pre_value[div_start] if mod_start else pre_value[div_start]
                    pre_value=pre_value[div_start:]
                    result = int(round(sum(pre_value)) if sum(pre_value)>=0 else 0)
                    
                    score.append(abs(result - sum(test)))
                score = sum(score)
                if score < cv_score['best_score']:
                    cv_score['best_score'] = score
                    cv_score['best_ind'] = cnt
                cv_score['fit'].append({'outlier':out, 'data_merge':day_step,'params':p1, 'score':score})
                cnt += 1
    
#    print cv_score['fit'][cv_score['best_ind']]
    return cv_score

def cv_estimate(train_data, test_len, forecast_stat, K=10, other_params={'data_merge':[3,4,5,6,7,8],
                                                                        'outier':['repByThres','2nn','3mean','5mean','7mean']}):
#    if(len(train_data) < train_lower):
#        train_lower = len(train_data) - test_len - 10
    
    all_len = len(train_data)
    train_lower = round((all_len - test_len) * 0.7)
    train_upper = all_len - test_len
    
    cv_len = [(random.randrange(train_lower, train_upper), test_len) for x in range(K)]
    cv_bag = [(random.randrange(0, all_len - train_len - test), train_len, test) for train_len, test in cv_len]
    cv_data = [(train_data[start:(start+train_len)], test_outlier(train_data[(start+train_len):(start+train_len+test)])) for start, train_len, test in cv_bag]
    
    alpha = map(lambda x: x*0.1, range(1,10))
    cv_exp = cv_exp_smoothing_estimate(cv_data, alpha, test_len, forecast_stat,other_params)
    sec_optim_param = cv_exp['fit'][cv_exp['best_ind']]
    
    other_params['data_merge'] = [sec_optim_param['data_merge']]
    cv_exp2 = cv_exp_smoothing_estimate(cv_data, map(lambda x:x/100.0, range(int(sec_optim_param['params'] * 100 -9),
                int(sec_optim_param['params']*100+10))), test_len, forecast_stat, other_params)
    optim_param = cv_exp2['fit'][cv_exp2['best_ind']]
    
    # forecast
    day_step = optim_param['data_merge']
    div_end,mod_end=divmod(test_len,day_step)
    div_start,mod_start=divmod(forecast_stat,day_step)
    steps=div_end+1  if  mod_end else div_end    
    
    agg_data = cv_agg_by_step(train_data, day_step, 0.6, optim_param['outlier'])
    pre_value = cv_exponential_smoothing_forecast(agg_data,steps,optim_param['params'])
    return pre_value, optim_param


def del_dict(list,del_dict):

    tmplist =list
    for key in del_dict.keys():
        for i in range(len(tmplist)):
            if tmplist[i] == key :
                tmplist[i+1] -= int(del_dict[key])
                list[0] -= int(del_dict[key])
    return tmplist



def predict_vm(ecs_lines, input_lines):
    # Do your work from here#
    result = []
    if ecs_lines is None:
        print 'ecs information is none'
        return result
    if input_lines is None:
        print 'input file information is none'
        return result
    dict = get_params(input_lines)
    #judge forecast days
    forecast_day= cal_day_count( dict['params_4'][0][0],dict['params_4'][1][0]).days
    flavor_list, sort_list,forecast_stat,forecast_end = read_train_data(ecs_lines,dict)
    count = 0
    li = []
    li.append(count)

    for f in get_flavor_to_forecast(dict):
        #input merge day
        #data_merge_by_day=forecast_day
        #fg = agg_by_step(f, data_merge_by_day, 0, flavor_list , sort_list, outlier_type = '3mean')
        #fg = agg_by_step(f, data_merge_by_day, 0, flavor_list , sort_list, outlier_type = '2nn')
        #fg = agg_by_step(f, data_merge_by_day, 0, flavor_list , sort_list, outlier_type = 'repByThres')
        #judge forecast days

        #div_end,mod_end=divmod(forecast_end,data_merge_by_day)
       #div_start,mod_start=divmod(forecast_stat,data_merge_by_day)
        #steps=div_end+1  if  mod_end else div_end

        #input model
        #ordesr=(1,1)
        #ordesr_diff=(1,1,1)
        #pre_value=huawei_total.auto_fit_forecast(fg,ordesr,ordesr_diff,trend='nc',steps=steps,star_para=None,k_fit_ars=1)
        #pre_value=huawei_total.lwlr_predict(fg,2,steps=steps,k=8)
        #pre_value=huawei_total.haha_predict(fg,steps,1)
        #pre_value=huawei_total.merge_model(fg,trend='nc',steps=steps,k_fit_ars=4,start_num=0)
        pre_value_qt=huawei_total.best_arima_model(f,[(1,0),(2,0),(1,1),(0,1),(3,0),(4,0)],\
                                                   [(1,1,0),(1,1,1),(0,1,1),(2,1,0)],\
                                                   [3,4,5,6,7],['3mean','2nn','repByThres'],flavor_list,sort_list,forecast_stat,forecast_end,trend='nc',k_fit_ars=[4],threshold=0)
        #merge 
        
        cv_K = 50
        other_params={'data_merge':[3,4,5,6,7,8],
                     'outier':['repByThres','2nn','3mean','5mean','7mean']}
        
        pre_value, optim_params = cv_estimate(flavor_list[f-1], forecast_end, forecast_stat, K=cv_K,other_params=other_params)
        
        data_merge_by_day = optim_params['data_merge']



        div_end,mod_end=divmod(forecast_end,data_merge_by_day)
        div_start,mod_start=divmod(forecast_stat,data_merge_by_day)
        steps=div_end+1  if  mod_end else div_end
        pre_value[-1]= (mod_end/(1.0*data_merge_by_day))*pre_value[-1] if mod_end else pre_value[-1]
        pre_value[div_start]=(1-mod_start/(1.0*data_merge_by_day))*pre_value[div_start] if mod_start else pre_value[div_start]
        pre_value=pre_value[div_start:]
        
        
        #accumulate result
        cur_num= int(round(0.35*sum([ 0 if i <0 else i for i in pre_value_qt])+0.65*sum([ 0 if i <0 else i for i in pre_value])))
        #cur_num= int(round(sum([ 0 if i <0 else i for i in pre_value_qt])))
        #cur_num=int(round(sum(pre_value)) if sum(pre_value)>=0 else 0)
        cur_flavor = 'flavor' + str(f)
        li.append(cur_flavor)
        li.append(cur_num)
        count += cur_num
    li[0] = count
#    print count
#    print li

    #del the last physical meachines
    arr,phy_use_count,dict_del = put_fun(dict, li)
    if (del_dict != {}):
        li = del_dict(li,dict_del)
        arr, phy_use_count, dict_del= put_fun(dict, li)

    print phy_use_count
    list_prin = []
    # put in list_prin
    list_prin.append(li[0])
    for i in range(len(li) / 2):
        str_tem = li[i * 2 + 1] + ' ' + str(li[i * 2 + 2])
        list_prin.append(str_tem)
    list_prin.append("")

    list_prin.append(phy_use_count)
    for i in range(len(arr)):
        str_convert = " ".join(arr[i])
        list_prin.append(str_convert)
    result = list_prin
#    print(result)
    return result



