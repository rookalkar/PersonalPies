// this model has a general b value for all and a probability of segment for each participant.

data {
  //Training Data
  int<lower=1> n; //number of obervations
  vector[n] correct_ans; 
  int participant[n]; //participant ID
  vector[n] response; //participant's response
  
  int<lower=1> k; //number of participants
  int<lower=1> number_segments;
  vector[number_segments] segments; //number of markings (1 implies no other marks, 2 implies 3 points - 0, half, full ) r[n] n = segments
}

parameters {
  real log_b_z;
  real<lower=0> phi;
  real mu_b;
  real<lower=0> sigma_b;
  vector[number_segments] mu;
  vector<lower=0>[number_segments] sigma ;
  vector[number_segments] theta_raw[k];
}

transformed parameters {
  matrix[number_segments, n] p;
  matrix[number_segments, n] riminus_value; //ri minus one
  matrix[number_segments, n] ri_value;
  
  real<lower=0> b;
  simplex[number_segments] prob_segment[k];
  
  b = exp(log_b_z * sigma_b + mu_b);
  
  for (m in 1:k) { //k is number of participants
      prob_segment[m] = softmax(theta_raw[m]);
  }

  for (j in 1:number_segments) {
    for (i in 1:n){
    //figure out which segment correct_ans lies in 
    riminus_value[j][i] = floor(correct_ans[i]*segments[j])*(1/segments[j]);
    ri_value[j][i] = riminus_value[j][i] + 1/segments[j];
    // person's estimate
    p[j][i] = (pow((correct_ans[i] - riminus_value[j][i]),b)/(pow((correct_ans[i] - riminus_value[j][i]),b)+pow((ri_value[j][i] - correct_ans[i]),b)))*(1/segments[j]) + (riminus_value[j][i]);
    } 
  }

} 

model {
  
  //prior for phi,b
  phi ~ cauchy(0,5);
  
  mu ~ normal(0,1);
  sigma~ cauchy(0,1);

  
  //model
  
  for (m in 1:k){
    theta_raw[m] ~ normal(mu, sigma);
  }
  
  mu_b ~ normal(0,1);
  sigma_b ~ cauchy(0,1);
  log_b_z ~ normal(0, 1);
  
  for (i in 1:n){
    vector[number_segments] test;
    
    for (j in 1:number_segments) {
      test[j] = beta_lpdf(response[i] | p[j][i]*phi, (1-p[j][i])*phi) + log(prob_segment[participant[i]][j]);
    }
    
    target += log_sum_exp(test);
  }
}

generated quantities {
  vector[n] response_pred;
  int<lower=0,upper=number_segments> segment_selected[n];
  
  for (i in 1:n) {
    segment_selected[i] = categorical_rng(prob_segment[participant[i]]);
    response_pred[i] = beta_rng(p[segment_selected[i]][i]*phi, (1 - p[segment_selected[i]][i])*phi);
  }
}
