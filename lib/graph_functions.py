import numpy as np
import scipy as sp
import scipy.stats as stats

# b[0] * x ** 0 + b[1] * x ** 1 + b[2] * x ** 2 + ...
def polynomial (beta, x):
    return sum([b * x ** i for i,b in enumerate(beta)])

def gaussian(x, mean, std):
    return 1.0/(std * np.sqrt(2*np.pi)) * np.exp( -1/2 * ( ( (x - mean)/std ) ** 2) )


def log_likelihood_gauss_regression(f, beta, gaussian_std, x, y):
    return sum([np.log(sp.special.erf(abs(gaussian(y_i, f(beta, x_i), gaussian_std))))  for x_i, y_i in zip(x,y)])

def chi_likelihood_combined(f, beta, x, y, pearson=False, sdy=[]):
    
    if len(sdy) > 0:
        statistic = sum( ((y - polynomial(beta, x)) / sdy) ** 2)
        log_likelihood = sum( np.log(sp.special.erf(abs(gaussian(y, f(beta, x), sdy)))) )
        return statistic, log_likelihood

    bin_dict = {}
    current = []
    last_d = 0
    d_sorted = sorted(list(zip(x, y)))
    for d,c in d_sorted:
        if d > last_d:
            bin_dict[last_d] = current
            current = []
            last_d = d
        current.append(c)
    statistic = 0
    log_likelihood = 0
    for d,c in bin_dict.items():
        std = np.std(c)
        if len(c) > 5 and not std == 0 and not pearson:
            # print('1')
            xx = sum( ((c - polynomial(beta, d)) / std) ** 2)
            statistic += xx
            log_likelihood += sum( np.log(sp.special.erf(abs(gaussian(c, f(beta, d), std)))) )
        else:            
            # Resort to pearson when there is only one point
        
            # approximate that the log likelihodd is irrelevant when std = 0?
            # log_likelihood += sum( np.log(sp.special.erf(abs(gaussian(c, f(beta, d), std)))) )
            
            statistic += sum( ((c - polynomial(beta, d))) ** 2 / polynomial(beta, d))

    if np.isnan(log_likelihood):
        print('is nan...')
    return statistic, log_likelihood


def closeness_and_degree_to_x_y_bins(closeness_values, degree_values):
    # convert to arrays
    closeness_values = np.array(closeness_values)
    degree_values = np.array(degree_values)

    # converts to 1/closness
    y_val = 1 / closeness_values
    x_val = degree_values

    # Varaible delcarations to be used later
    y_sub_total = 0
    x_sub_total = 0
    count = 0
    y_avg = []
    x_avg = []
    y_avg_data = []
    x_avg_data = []
    y_avg_error = []
    x_avg_error = []
    error_sum_y = 0
    error_sum_x = 0
    ordered_degree = np.sort(degree_values)

    # Loops through all the possible degrees
    for i in range(int(max(x_val)) + 1):

        # for each degree adds the values to a total and appends the item to a list
        for j in range(np.size(x_val)):
            if x_val[j] == i:
                y_sub_total += y_val[j]
                y_avg_data.append(y_val[j])
                count += 1
                x_sub_total += x_val[j]
                x_avg_data.append(x_val[j])

        # if there are 5 or more items inspected then calculate averages and errors
        # if there are more than 4 degrees left to process
        if count > 4 and i < ordered_degree[-4]:

            # calculate average
            y_avg_value = y_sub_total / count
            x_avg_value = x_sub_total / count

            # calculate sum of errors
            for k in range(count):
                error_sum_y += (y_avg_data[k] - y_avg_value) ** 2
                error_sum_x += (x_avg_data[k] - x_avg_value) ** 2

            # calculate standard deviation in x and y with error propagation
            sd_y = np.sqrt((1 / (count - 1)) * error_sum_y)
            sd_x = np.sqrt(((1 / x_avg_value) ** 2) * (1 / (count - 1)) * error_sum_x)

            # append to list
            y_avg.append(y_avg_value)
            x_avg.append(x_avg_value)
            y_avg_error.append(sd_y)

            # if x error is 0 as all same degree make error very small
            if sd_x == 0:
                sd_x = 0.0000000000001
            x_avg_error.append(sd_x)

            # reset variables for next loop
            y_sub_total = 0
            x_sub_total = 0
            count = 0
            error_sum_x = 0
            error_sum_y = 0
            y_avg_data = []
            x_avg_data = []

        # if final loop then do as above
        elif i == max(x_val):

            # calculate average
            y_avg_value = y_sub_total / count
            x_avg_value = x_sub_total / count

            # calculate sum of errors
            for k in range(count):
                error_sum_y += (y_avg_data[k] - y_avg_value) ** 2
                error_sum_x += (x_avg_data[k] - x_avg_value) ** 2

            # calculate standard deviation in x and y with error propagation
            sd_y = np.sqrt((1 / (count - 1)) * error_sum_y)
            sd_x = np.sqrt(((1 / x_avg_value) ** 2) * (1 / (count - 1)) * error_sum_x)

            # append to list
            y_avg.append(y_avg_value)
            x_avg.append(x_avg_value)
            y_avg_error.append(sd_y)

            # if x error is 0 as all same degree make error very small
            if sd_x == 0:
                sd_x = 0.0000000000001
            x_avg_error.append(sd_x)

            # reset variables for next loop
            y_sub_total = 0
            x_sub_total = 0
            count = 0
            error_sum_x = 0
            error_sum_y = 0
            y_avg_data = []
            x_avg_data = []

    # convert final values and make x into log x, error propagation done above
    y = np.array(y_avg)
    x = np.log(np.array(x_avg))
    sd_y = np.array(y_avg_error)
    sd_x = np.array(x_avg_error)

    # return x, y, sd_x, sd_y
    return x, y, sd_x, sd_y


def chi_squared(x,y,f,beta, pearson=False):
    bin_dict = {}
    current = []
    last_d = 0
    d_sorted = sorted(list(zip(x, y)))
    # print(d_sorted)
    for d,c in d_sorted:
        if d > last_d:
            if len(current) < 1 : print(current)
            bin_dict[last_d] = current
            current = []
            last_d = d
        current.append(c)
    statistic = 0
    statistic_pears = 0
    statistic_stat = 0
    for d,c in bin_dict.items():
        # if len (c) < 1:
            # print(c)
        if len(c) > 5 and not np.std(c) == 0 and not pearson:
            xx = sum( ((c - polynomial(beta, d)) / np.std(c)) ** 2)
            statistic += xx
            # if xx > 2*len(c):
            #     print('large', xx, len(c), np.std(c), c, polynomial(beta, d))
        else:
            # Resort to pearson when there is only one point
            statistic += sum( ((c - polynomial(beta, d))) ** 2 / polynomial(beta, d))

        statistic_pears += sum( ((c - polynomial(beta, d))) ** 2 / polynomial(beta, d))
        statistic_stat += stats.chisquare(c, f_exp=polynomial(beta, d))[0]
    return statistic


def print_metrics(betas, zs, chis, chis_r, log_Ls, bics, raw_bic_list):
    spacer, raw, binn, rawbin = ' ', 'Raw', 'Bin', 'Raw/Bin'
    d = 24
    print(f'{spacer:{d}}{raw:{d}}{binn:{d}}{rawbin:{d}}')

    g = 'Gradient'
    print(f'{g:{d}}',end='')
    for b in betas:
        print(f'{b[0][1]:+.3g} ± {b[1][1]:<15.3g}', end='')

    z_exp = 1
    z = 'z'
    print(f'\n{z:{d}}',end='')
    for z in zs:
        try:
            print(f'{z[0]:+.4g} ± {z[1]:<15.4g}', end='')
        except:
            z_exp = z[0]
            print(f'Predicted: {z[0]:<24.4g}', end='')
    
    z_fr = 'z fraction'
    print(f'\n{z_fr:{d}}',end='')
    for z in zs:
        try:
            print(f'{z[0]/z_exp:+.4g} ± {z[1]/z_exp:<14.4g}', end='')
        except:
            pass
    
    c = 'Chi-squared'
    print(f'\n{c:{d}}',end='')
    for chi in chis:
        print(f'{chi[0]:<24.3g}', end='')
    

    cr = 'Chi-squared (R)'
    print(f'\n{cr:{d}}',end='')
    for chi in chis_r:
        print(f'{chi[0]:<24.3g}', end='')
    
    b = 'BIC'
    print(f'\n{b:{d}}',end='')
    for bic in bics:
        print(f'{bic[0]:<24.3g}', end='')

    for i, bic in enumerate(raw_bic_list):
        b = f'BIC {i:01}'
        print(f'\n{b:{d}}',end='')
        # for bic in bics:
        print(f'{bic:<24.5g}', end='')
    
    l = 'Log Likelihood'
    print(f'\n{l:{d}}',end='')
    for L in log_Ls:
        print(f'{L[0]:<+24.3g}', end='')

def bic_value_chi(chi, n, k):
    return chi + k * np.log(n)

def bic_value_var(residual_variance, n, k):
    return n * np.log(residual_variance) + k * np.log(n)

def bic_value_log(log_L_max, n, k):
    return k * np.log(n) - 2 * log_L_max