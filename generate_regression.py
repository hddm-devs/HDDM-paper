import hddm
import kabuki
import numpy as np
import scipy
import pandas as pd
from copy import deepcopy

def gen_regression_data(params, subj_noise, share_noise = ('a','v','t','st','sz','sv','z', 'v_slope', 'v_inter'), size=50,
                        subjs=1, exclude_params=(), **kwargs):

    """Generate simulated RTs with random parameters.

       :Optional:
            params : dict
                Parameter names and values. If not
                supplied, takes random values.
            method : string
                method to generate samples
            the rest of the arguments are forwarded to kabuki.generate.gen_rand_data

       :Returns:
            data array with RTs
            parameter values

    """

    from numpy import inf

    # set valid param ranges
    bounds = {'a': (0, inf),
              'z': (0, 1),
              't': (0, inf),
              'st': (0, inf),
              'sv': (0, inf),
              'sz': (0, 1)
    }


    # Create RT data
    group_params = []
    for i_subj in range(subjs):
      subj_params = kabuki.generate._add_noise({'none': params}, noise=subj_noise, share_noise=share_noise,
                                        check_valid_func=hddm.utils.check_params_valid,
                                        bounds=bounds,
                                        exclude_params=exclude_params)['none']
      group_params.append(subj_params)

      #generate v
      wfpt_params = deepcopy(subj_params)
      wfpt_params.pop('v_inter')
      effect = wfpt_params.pop('v_slope')
      x1 = np.random.randn(size);
      x2 = np.random.randn(size);
      wfpt_params['v'] = (effect*x1 + np.sqrt(1-effect**2)*x2) + subj_params['v_inter'];

      #generate data
      subj_data, _ = kabuki.generate.gen_rand_data(hddm.models.hddm_regression.wfpt_reg_like, wfpt_params,
                                                        size=size,
                                                        check_valid_func=hddm.utils.check_params_valid,
                                                        bounds=bounds, share_noise=share_noise, **kwargs)

      #fix data a little bit
      subj_data = pd.DataFrame(hddm.generate.kabuki_data_to_hddm_data(subj_data))
      subj_data['cov'] = x1
      subj_data['subj_idx'] = i_subj

      #concatante subj_data to group_data
      if i_subj == 0:
        data = subj_data
      else:
        data = pd.concat((data, subj_data), ignore_index=True)

    return data, group_params


def gen_reg_params(effect, **kwargs):

  params = hddm.generate.gen_rand_params(**kwargs)
  params['v_slope'] = effect
  params['v_inter'] = 1
  params['sv'] = 0
  del params['v']

  return params
