// this model has a b value for each participant and a general probability of segment for all.

data {
  int<lower=1> k; //total participants
  int<lower=1> n; //number of obervations
  vector[n] correct_ans; 
  int participant[n]; //participant ID
  vector[n] response; //participant's response
  int<lower=1> number_segments;
  vector[number_segments] segments; //number of markings (1 implies no other marks, 2 implies 3 points - 0, half, full ) r[n] n = segments
}

parameters {
  vector[k] log_b_z;
  real mu_b;
  real<lower=0> sigma_b;
  real<lower=0> phi;
  simplex[number_segments] prob_segment;
}

transformed parameters {
  matrix[number_segments, n] p;
  matrix[number_segments, n] riminus_value; //ri minus one
  matrix[number_segments, n] ri_value;
  
  vector<lower=0>[k] b;
  
  for (m in 1:k) { //k is number of participants
    //b for participant ID: m
    b[m] = exp(log_b_z[m] * sigma_b + mu_b);
  }

  for (j in 1:number_segments) {
    for (i in 1:n){
    //figure out which segment correct_ans lies in 
    riminus_value[j][i] = floor(correct_ans[i]*segments[j])*(1/segments[j]);
    ri_value[j][i] = riminus_value[j][i] + 1/segments[j];
    // person's estimate
    p[j][i] = (pow((correct_ans[i] - riminus_value[j][i]),b[participant[i]])/(pow((correct_ans[i] - riminus_value[j][i]),b[participant[i]])+pow((ri_value[j][i] - correct_ans[i]),b[participant[i]])))*(1/segments[j]) + (riminus_value[j][i]);
    } 
  }
} 

model {
  
  //prior for phi,b
  phi ~ cauchy(0,5);
  
  mu_b ~ normal(0,1);
  sigma_b ~ cauchy(0,1);
  
  //model
  
  for (m in 1:k){
    log_b_z[m] ~ normal(0, 1);
  }
  
  prob_segment ~ uniform(0,1);
  
  for (i in 1:n){
    vector[number_segments] test;
    
    for (j in 1:number_segments) {
      test[j] = beta_lpdf(response[i] | p[j][i]*phi, (1-p[j][i])*phi) + log(prob_segment[j]);
    }
    
    target += log_sum_exp(test);
  }
}

generated quantities {
  vector[n] log_lik;
  
  for (i in 1:n) {
    vector[number_segments] test;
    for (j in 1:number_segments) {
      test[j] = beta_lpdf(response[i] | p[j][i]*phi, (1-p[j][i])*phi) + log(prob_segment[j]);
    }
    log_lik[i] = log_sum_exp(test);
  }
}
