// this model has a general b value for all and a probability of segment for each participant.

data {
  //Training Data
  int<lower=1> nt; //number of obervations
  vector[nt] correct_ans_t; 
  int<lower=1> kt; //number of participants
  int participant_t[nt]; //participant ID
  vector[nt] response_t; //participant's response
  
  //Testing Data
  int<lower=1> nh; //number of obervations
  vector[nh] correct_ans_h; 
  int<lower=1> kh; //number of participants
  int participant_h[nh]; //participant ID
  vector[nh] response_h; //participant's response
  
  int<lower=1> k; //number of participants
  int<lower=1> number_segments;
  vector[number_segments] segments; //number of markings (1 implies no other marks, 2 implies 3 points - 0, half, full ) r[n] n = segments
}

parameters {
  real<lower=0> phi;
  real log_b_z;
  real mu_b;
  real<lower=0> sigma_b;
  vector[number_segments] mu;
  vector<lower=0>[number_segments] sigma ;
  vector[number_segments] theta_raw[k];
}

transformed parameters {
  matrix[number_segments, nt] p_t;
  matrix[number_segments, nt] riminus_value_t; //ri minus one
  matrix[number_segments, nt] ri_value_t;
  
  matrix[number_segments, nh] p_h;
  matrix[number_segments, nh] riminus_value_h; //ri minus one
  matrix[number_segments, nh] ri_value_h;
  
  real<lower=0> b;
  simplex[number_segments] prob_segment[k];
  
  
  for (m in 1:k) { //k is number of participants
      prob_segment[m] = softmax(theta_raw[m]);
  }
  
  b = exp(log_b_z * sigma_b + mu_b);

  for (j in 1:number_segments) {
    for (i in 1:nt){
    //figure out which segment correct_ans lies in 
    riminus_value_t[j][i] = floor(correct_ans_t[i]*segments[j])*(1/segments[j]);
    ri_value_t[j][i] = riminus_value_t[j][i] + 1/segments[j];
    // person's estimate
    p_t[j][i] = (pow((correct_ans_t[i] - riminus_value_t[j][i]),b)/(pow((correct_ans_t[i] - riminus_value_t[j][i]),b)+pow((ri_value_t[j][i] - correct_ans_t[i]),b)))*(1/segments[j]) + (riminus_value_t[j][i]);
    } 
  }
  
  for (j in 1:number_segments) {
    for (i in 1:nh){
    //figure out which segment correct_ans lies in 
    riminus_value_h[j][i] = floor(correct_ans_h[i]*segments[j])*(1/segments[j]);
    ri_value_h[j][i] = riminus_value_h[j][i] + 1/segments[j];
    // person's estimate
    p_h[j][i] = (pow((correct_ans_h[i] - riminus_value_h[j][i]),b)/(pow((correct_ans_h[i] - riminus_value_h[j][i]),b)+pow((ri_value_h[j][i] - correct_ans_h[i]),b)))*(1/segments[j]) + (riminus_value_h[j][i]);
    } 
  }
} 

model {
  
  //prior for phi,b
  phi ~ cauchy(0,5);
  
  mu_b ~ normal(0,1);
  sigma_b ~ cauchy(0,1);
  
  mu ~ normal(0,1);
  sigma~ cauchy(0,1);

  
  //model
  
  for (m in 1:k){
    theta_raw[m] ~ normal(mu, sigma);
  }
  
  log_b_z ~ normal(0, 1);
  
  for (i in 1:nt){
    vector[number_segments] test;
    
    for (j in 1:number_segments) {
      test[j] = beta_lpdf(response_t[i] | p_t[j][i]*phi, (1-p_t[j][i])*phi) + log(prob_segment[participant_t[i]][j]);
    }
    
    target += log_sum_exp(test);
  }
}

generated quantities {
  vector[nt] log_lik_t;
  vector[nh] log_lik_h;
  
  for (i in 1:nt) {
    vector[number_segments] test_t;
    for (j in 1:number_segments) {
      test_t[j] = beta_lpdf(response_t[i] | p_t[j][i]*phi, (1-p_t[j][i])*phi) + log(prob_segment[participant_t[i]][j]);
    }
    log_lik_t[i] = log_sum_exp(test_t);
  }
  
  for (i in 1:nh) {
    vector[number_segments] test_h;
    for (j in 1:number_segments) {
      test_h[j] = beta_lpdf(response_h[i] | p_h[j][i]*phi, (1-p_h[j][i])*phi) + log(prob_segment[participant_h[i]][j]);
    }
    log_lik_h[i] = log_sum_exp(test_h);
  }
}
