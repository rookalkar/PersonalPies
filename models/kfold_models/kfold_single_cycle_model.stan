// this model has no b value and no probability of segment for each participant.

data {
  //Training Data
  int<lower=1> nt; //number of obervations
  vector[nt] correct_ans_t; 
  vector[nt] response_t; //participant's response
  
  // Holdout Data
  int<lower=1> nh; //number of obervations
  vector[nh] correct_ans_h; 
  vector[nh] response_h; //participant's response
  
  // Common Data
  real<lower=1> segments;
}

parameters {
  real<lower=0> phi;
  real<lower=0> b;
}

transformed parameters {
  vector[nt] p_t;
  vector[nt] riminus_value_t; //ri minus one
  vector[nt] ri_value_t;
  
  vector[nh] p_h;
  vector[nh] riminus_value_h; //ri minus one
  vector[nh] ri_value_h;

  for (i in 1:nt){
  //figure out which segment correct_ans lies in 
  riminus_value_t[i] = floor(correct_ans_t[i]*segments)*(1/segments);
  ri_value_t[i] = riminus_value_t[i] + 1/segments;
  // person's estimate
  p_t[i] = (pow((correct_ans_t[i] - riminus_value_t[i]),b)/(pow((correct_ans_t[i] - riminus_value_t[i]),b)+pow((ri_value_t[i] - correct_ans_t[i]),b)))*(1/segments) + (riminus_value_t[i]);
  }
  

  for (i in 1:nh){
  //figure out which segment correct_ans lies in 
  riminus_value_h[i] = floor(correct_ans_h[i]*segments)*(1/segments);
  ri_value_h[i] = riminus_value_h[i] + 1/segments;
  // person's estimate
  p_h[i] = (pow((correct_ans_h[i] - riminus_value_h[i]),b)/(pow((correct_ans_h[i] - riminus_value_h[i]),b)+pow((ri_value_h[i] - correct_ans_h[i]),b)))*(1/segments) + (riminus_value_h[i]);
  } 
} 

model {
  
  //prior for phi,b
  phi ~ cauchy(0,5);
  b ~ lognormal(0,1);

  //model
  for (i in 1:nt){
    response_t[i] ~ beta(p_t[i]*phi, (1-p_t[i])*phi);
  }
}

generated quantities {
  vector[nt] log_lik_t;
  vector[nh] log_lik_h;
  
  for (i in 1:nt) {
    log_lik_t[i] = beta_lpdf(response_t[i] | p_t[i]*phi, (1-p_t[i])*phi);
  }
  
  for (i in 1:nh) {
    log_lik_h[i] = beta_lpdf(response_h[i] | p_h[i]*phi, (1-p_h[i])*phi);
  }
}
