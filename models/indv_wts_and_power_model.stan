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
  vector[k] log_b_z;
  real<lower=0> phi;
  real mu_b;
  real<lower=0> sigma_b;
  vector[number_segments] mu;
  vector<lower=0>[number_segments] sigma;
  vector[number_segments] theta_raw[k];
}

transformed parameters {
  matrix[number_segments, n] p;
  matrix[number_segments, n] riminus_value; //ri minus one
  matrix[number_segments, n] ri_value;
  
  vector<lower=0>[k] b;
  simplex[number_segments] prob_segment[k];
  
  

  for (m in 1:k) { //k is number of participants
    //b for participant ID: m
    b[m] = exp(log_b_z[m] * sigma_b + mu_b);
    prob_segment[m] = softmax(theta_raw[m]);
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
  
  mu ~ normal(0,1);
  sigma~ cauchy(0,1);

  
  //model
  
  
  for (m in 1:k){
    log_b_z[m] ~ normal(0, 1);
    theta_raw[m] ~ normal(mu, sigma);
  }
  
  for (i in 1:n){
    vector[number_segments] test;
    
    for (j in 1:number_segments) {
      test[j] = beta_lpdf(response[i] | p[j][i]*phi, (1-p[j][i])*phi) + log(prob_segment[participant[i]][j]);
    }
    
    target += log_sum_exp(test);
  }
}

