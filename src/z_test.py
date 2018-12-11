
from numpy import sqrt
import scipy.stats as scs


def z_test(ctr_old, ctr_new, nobs_old, nobs_new,
           effect_size=0., two_tailed=True, alpha=.05):
    """Perform z-test to compare two proprtions (e.g., click through rates (ctr)).

        Note: if you set two_tailed=False, z_test assumes H_A is that the effect is
        non-negative, so the p-value is computed based on the weight in the upper tail.

        Arguments:
            ctr_old (float):    baseline proportion (ctr)
            ctr_new (float):    new proportion
            nobs_old (int):     number of observations in baseline sample
            nobs_new (int):     number of observations in new sample
            effect_size (float):    size of effect
            two_tailed (bool):  True to use two-tailed test; False to use one-sided test
                                where alternative hypothesis if that effect_size is non-negative
            alpha (float):      significance level

        Returns:
            z-score, p-value, and whether to reject the null hypothesis
    """
    conversion = (ctr_old * nobs_old + ctr_new * nobs_new) / \
                 (nobs_old + nobs_new)

    se = sqrt(conversion * (1 - conversion) * (1 / nobs_old + 1 / nobs_new))

    z_score = (ctr_new - ctr_old - effect_size) / se

    if two_tailed:
        p_val = (1 - scs.norm.cdf(abs(z_score))) * 2
    else:
        # Because H_A: estimated effect_size > effect_size
        p_val = 1 - scs.norm.cdf(z_score)

    reject_null = p_val < alpha
    print('z-score: %s, p-value: %s, reject null: %s' % (z_score, p_val, reject_null))
    return z_score, p_val, reject_null

if __name__ == '__main__':
    # Testing the z-test on a known example
    old_p = 100. / 1000
    new_p = 105. / 1000
    old_row = 1000.
    new_row = 1000.
    z_test(old_p, new_p, old_row, new_row)

    # p-value should be << 1
    z_test(old_p, new_p, old_row, new_row, two_tailed=False)

    # p-value should be 1 because z-score < 0
    z_test(new_p, old_p, old_row, new_row, two_tailed=False)
