data {
  int<lower=1> n; //number of obervations
  int<lower=1> number_segments;
  vector[n] correct_ans; 
  vector[n] response; //participant's response
  vector[number_segments] segments; //number of markings (1 implies no other marks, 2 implies 3 points - 0, half, full ) r[n] n = segments
}

parameters {
  real<lower=0> b;
  real<lower=0> phi;
  simplex[number_segments] prob_segment;
}

transformed parameters {
  matrix[number_segments, n] p;
  matrix[number_segments, n] riminus_value; //ri minus one
  matrix[number_segments, n] ri_value;

  for (j in 1:number_segments) {
    for (i in 1:n){
    //For segment 1
    //figure out which segment correct_ans lies in 
    riminus_value[j][i] = floor(correct_ans[i]*segments[j])*(1/segments[j]);
    ri_value[j][i] = riminus_value[j][i] + 1/segments[j];
    
    p[j][i] = (pow((correct_ans[i] - riminus_value[j][i]),b)/(pow((correct_ans[i] - riminus_value[j][i]),b)+pow((ri_value[j][i] - correct_ans[i]),b)))*(1/segments[j]) + (riminus_value[j][i]);
    } 
  }
} 

model {
  //prior for phi,b
  phi ~ cauchy(0,5);
  b ~ lognormal(0,1);
  prob_segment ~ uniform(0,1);
  
  //model
  for (i in 1:n){
    vector[number_segments] test;
    for (j in 1:number_segments) {
      test[j] = beta_lpdf(response[i] | p[j][i]*phi, (1-p[j][i])*phi) + log(prob_segment[j]);
    }
    target += log_sum_exp(test);
  }
}

