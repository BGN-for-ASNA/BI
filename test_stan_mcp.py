import sys
import os

# Ensure BI is in path
sys.path.insert(0, r"c:\Users\Sosa\Documents\BI")

from mcp_server.tools import convert_stan_to_bi, validate_bi_model
from mcp_server.resources import get_stan_conversion_examples

print("Testing Stan examples read...")
examples_json = get_stan_conversion_examples()
print(f"Loaded length: {len(examples_json)}")

test_stan = """
data{
    vector[346] height;
    vector[346] weight;
}
parameters{
    real a;
    real<lower=0> b;
    real<lower=0,upper=50> s;
}
model{
    vector[346] mu;
    s ~ uniform( 0 , 50 );
    b ~ lognormal( 0 , 1 );
    a ~ normal( 178 , 20 );
    for ( i in 1:346 ) {
        mu[i] = a + b* weight[i] ;
    }
    height ~ normal( mu , s );
}
"""

print("\n--- Testing convert_stan_to_bi ---")
res = convert_stan_to_bi(test_stan)
if res['success']:
    print(res['bi_code'])
else:
    print("Error:", res['error'])

print("\n--- Testing validate_bi_model ---")
if res['success']:
    val = validate_bi_model(res['bi_code'])
    print("Valid Python:", val.get('is_valid_python'))
    if val.get('warnings'):
        print("Warnings:", val['warnings'])
else:
    print("Skipped validation due to failed conversion.")
